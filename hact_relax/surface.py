from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
from typing import Sequence

import numpy as np

from frozen_bath_scf import second_order as so
from frozen_bath_scf.orbital_optimizer import (
    energy_at,
    minimize_orbitals,
    rotate_hamiltonian,
)
from frozen_bath_scf.schmidt_embedding import (
    SchmidtEmbeddedRHF,
    SchmidtEmbeddedROHF,
)
from system.pristine import build_pristine_mean_field

from hact_relax.cases import CASES
from hact_relax.errors import SCFNotConverged, SubspaceContinuityError
from hact_relax.geometry import restored_pristine_atoms
from hact_relax.tracking import (
    ActiveSolutionReference,
    PristineReference,
    align_pristine_occupied_subspace,
    defect_orbital_index,
    make_pristine_reference,
    transport_active_solution,
    transport_defect_orbital,
)


@dataclass
class SurfaceSettings:
    case_name: str
    fragment: int
    reference_kmesh: tuple[int, int, int]
    basis: str
    memory_mb: int
    bath_tol: float
    level_shift: float
    scf_max_cycle: int
    scf_conv_tol: float
    scf_conv_tol_grad: float
    min_subspace_overlap: float
    min_center_subspace_overlap: float
    verbose: int
    frozen_embedding: bool = False
    frozen_embedding_outside_fragment: bool = False
    auxbasis: str | None = None
    follow_negative_mode: bool = False
    follow_mode_tol: float = 1e-8
    rescue_unconverged_scf: bool = False
    rescue_tol_factor: float = 5.0


class ActiveHamiltonianSurface:

    def __init__(
        self,
        settings: SurfaceSettings,
        labels: Sequence[str],
        vacancy_index: int,
        lattice_bohr: np.ndarray,
    ):
        self.settings = settings
        self.case = CASES[settings.case_name]
        self.labels = list(labels)
        self.vacancy_index = int(vacancy_index)
        self.lattice_bohr = np.asarray(lattice_bohr, dtype=float)
        self.cache: dict[str, float] = {}
        self.last_accepted_key: str | None = None
        self.pristine_reference: PristineReference | None = None
        self.active_solution_reference: ActiveSolutionReference | None = None
        self.fixed_n_bath: int | None = None
        self.fixed_fragment_atoms: tuple[int, ...] | None = None
        self.last_center_followed = False

    @staticmethod
    def _key(coords_bohr: np.ndarray) -> str:
        payload = np.round(np.asarray(coords_bohr), decimals=11).tobytes()
        return hashlib.sha1(payload).hexdigest()[:16]

    def _guard_pristine_overlap(
        self, overlap: float | None, *, strict_continuation: bool
    ) -> None:
        if overlap is None or overlap >= self.settings.min_subspace_overlap:
            return
        if strict_continuation:
            raise SubspaceContinuityError(
                "pristine occupied-subspace overlap %.6f is below %.6f at a "
                "finite-difference point; refusing to switch KRHF branches"
                % (overlap, self.settings.min_subspace_overlap),
                continuity_space="pristine-occupied",
            )
        if overlap < self.settings.min_center_subspace_overlap:
            raise RuntimeError(
                "pristine occupied-subspace overlap %.6f at an accepted centre "
                "is below the centre threshold %.6f"
                % (overlap, self.settings.min_center_subspace_overlap)
            )

    @staticmethod
    def _apply_orbital_rotation(result, kappa, energy):
        h1e = np.asarray(result[5]).real
        eri = np.asarray(result[6]).real
        occ = np.asarray(result[4]).real
        u = so._expm_antisym(kappa)
        h1e_new, eri_new = rotate_hamiltonian(h1e, eri, u)
        updated = list(result)
        updated[1] = float(energy)
        updated[3] = np.asarray(result[3]).real @ u
        updated[5] = h1e_new
        updated[6] = eri_new
        dm_a, dm_b = so.densities(occ)
        v_a, v_b = so.general_veff(eri, u @ dm_a @ u.T, u @ dm_b @ u.T)
        updated[2] = np.diag(so._rotate(h1e + 0.5 * (v_a + v_b), u)).copy()
        return updated

    def _rescue_limit(self) -> float:
        return self.settings.scf_conv_tol_grad * self.settings.rescue_tol_factor

    def _rescue_unconverged_scf(self, result):
        if result[5] is None or result[6] is None:
            return result
        h1e = np.asarray(result[5]).real
        eri = np.asarray(result[6]).real
        occ = np.asarray(result[4]).real
        _, kappa = minimize_orbitals(
            h1e, eri, occ, np.zeros(int(so.rotation_mask(occ).sum()))
        )
        gradient = float(np.linalg.norm(
            so.orbital_gradient(h1e, eri, occ, kappa)
        ))
        if gradient > self._rescue_limit():
            return result
        updated = self._apply_orbital_rotation(
            result, kappa, float(energy_at(h1e, eri, occ, kappa))
        )
        updated[0] = True
        return updated

    def _converge_on_center_branch(self, solver, result, start_rotation, start_occ):
        if start_rotation is None or start_occ is None:
            return result
        u0 = np.asarray(start_rotation, dtype=float)
        occ = np.asarray(start_occ, dtype=float).copy()
        h1e = np.asarray(solver.get_hcore(solver.heff, u0)).real
        _, eri = solver.get_veff(occ, solver.eri_active, u0)
        eri = np.asarray(eri).real
        _, kappa = minimize_orbitals(
            h1e, eri, occ, np.zeros(int(so.rotation_mask(occ).sum()))
        )
        gradient = float(np.linalg.norm(
            so.orbital_gradient(h1e, eri, occ, kappa)
        ))
        if gradient > self._rescue_limit():
            return result
        seed = [
            True,
            float(energy_at(h1e, eri, occ)),
            np.zeros_like(occ),
            np.asarray(solver.mo_coeff[:, solver.active_orb]).real @ u0,
            occ,
            h1e,
            eri,
        ]
        updated = self._apply_orbital_rotation(
            seed, kappa, float(energy_at(h1e, eri, occ, kappa))
        )
        updated[0] = True
        return updated

    def _follow_negative_mode(self, result):
        h1e = np.asarray(result[5]).real
        eri = np.asarray(result[6]).real
        occ = np.asarray(result[4]).real
        report = so.stability_analysis(h1e, eri, occ)
        if report["lowest"] > -abs(self.settings.follow_mode_tol):
            return result, False

        before = energy_at(h1e, eri, occ)
        _, kappa = minimize_orbitals(
            h1e, eri, occ, 0.2 * np.asarray(report["lowest_mode"], dtype=float)
        )
        after = energy_at(h1e, eri, occ, kappa)
        if not (after < before - abs(self.settings.follow_mode_tol)):
            return result, False
        return self._apply_orbital_rotation(result, kappa, after), True

    def _transport_pristine(self, prior_pristine, displaced_atom) -> bool:
        if prior_pristine is None or prior_pristine.kmf is None:
            return False
        if self.settings.frozen_embedding:
            return True
        if not self.settings.frozen_embedding_outside_fragment:
            return False
        if displaced_atom is None:
            return False
        if self.fixed_fragment_atoms is None:
            return False
        return int(displaced_atom) not in self.fixed_fragment_atoms

    def energy(
        self,
        coords_bohr: np.ndarray,
        reason: str,
        *,
        accept_center: bool = False,
        displaced_atom: int | None = None,
        relax_active: bool = True,
    ) -> float:
        if accept_center and not relax_active:
            raise ValueError(
                "an accepted centre requires active-space relaxation"
            )
        if not relax_active and self.active_solution_reference is None:
            raise RuntimeError(
                "a semi-analytic displacement requires an accepted centre"
            )
        settings = self.settings
        coords_bohr = np.asarray(coords_bohr, dtype=float).reshape(-1, 3)
        strict_continuation = displaced_atom is not None

        key = self._key(coords_bohr)
        cached = self.cache.get(key) if relax_active else None
        if (cached is not None
                and (not accept_center or self.last_accepted_key == key)):
            return cached

        prior_pristine = self.pristine_reference
        transported = self._transport_pristine(prior_pristine, displaced_atom)
        if transported:
            pristine_kmf = prior_pristine.kmf
            next_pristine_reference = prior_pristine
        else:
            _, pristine_kmf = build_pristine_mean_field(
                restored_pristine_atoms(self.labels, coords_bohr),
                self.lattice_bohr,
                settings.basis,
                kmesh=settings.reference_kmesh,
                pseudo=None,
                charge=0,
                spin=0,
                exxdiv=None,
                verbose=settings.verbose,
                max_memory=settings.memory_mb,
                conv_tol=settings.scf_conv_tol,
                conv_tol_grad=settings.scf_conv_tol_grad,
                max_cycle=settings.scf_max_cycle,
                auxbasis=settings.auxbasis,
                dm0=None if prior_pristine is None else prior_pristine.density,
            )
            occupied_match = align_pristine_occupied_subspace(
                pristine_kmf, prior_pristine
            )
            self._guard_pristine_overlap(
                None if occupied_match is None else occupied_match.fidelity,
                strict_continuation=strict_continuation,
            )
            next_pristine_reference = make_pristine_reference(pristine_kmf)

        common = dict(
            vac_species=self.case.vacancy,
            n_frag=settings.fragment,
            charge=self.case.charge,
            bath_tol=settings.bath_tol,
            max_cycle=settings.scf_max_cycle,
            level_shift=settings.level_shift,
            verbose=settings.verbose,
            defect=True,
            eri_mode="cluster",
            seed_vacancy=self.case.seed_vacancy,
            vacancy_index=self.vacancy_index,
            fixed_n_bath=self.fixed_n_bath,
            fixed_fragment_atoms=self.fixed_fragment_atoms,
            auxbasis=settings.auxbasis,
            embedding_geometry=coords_bohr if transported else None,
        )
        if self.case.spin == 0:
            solver = SchmidtEmbeddedRHF(
                pristine_kmf, settings.reference_kmesh, **common
            )
        else:
            solver = SchmidtEmbeddedROHF(
                pristine_kmf, settings.reference_kmesh,
                spin=self.case.spin, **common
            )

        if solver.scell.natm != len(self.labels):
            raise RuntimeError(
                "the geometry (%d atoms) does not match the k-to-real-space "
                "embedding representation (%d atoms)"
                % (len(self.labels), solver.scell.natm)
            )
        if not np.allclose(
            solver.scell.atom_coords(), coords_bohr, atol=1e-9, rtol=0.0
        ):
            raise RuntimeError(
                "full-chain KRHF/embedding coordinates do not match the "
                "requested relaxation geometry"
            )

        start_rotation = None
        start_occ = None
        if self.active_solution_reference is not None:
            transport = transport_active_solution(
                self.active_solution_reference, solver
            )
            start_rotation = transport.rotation
            start_occ = transport.occupations
            transported_defect = transport_defect_orbital(
                self.active_solution_reference, solver, start_rotation
            )
            if transported_defect is not None:
                solver._c_def = transported_defect
            if transport.fidelity < settings.min_subspace_overlap:
                if strict_continuation:
                    raise SubspaceContinuityError(
                        "active-space transport fidelity %.6f is below %.6f; "
                        "the displaced active space no longer matches the "
                        "accepted centre"
                        % (transport.fidelity, settings.min_subspace_overlap),
                        continuity_space="active",
                    )
                if transport.fidelity < settings.min_center_subspace_overlap:
                    raise RuntimeError(
                        "active-space transport fidelity %.6f at a gradient "
                        "centre is below the centre threshold %.6f"
                        % (transport.fidelity,
                           settings.min_center_subspace_overlap)
                    )
                start_rotation = None
                start_occ = None

        if relax_active:
            result = solver.kernel(
                conv_tol=settings.scf_conv_tol,
                conv_tol_grad=settings.scf_conv_tol_grad,
                start_rotation=start_rotation,
                start_occ=start_occ,
            )

            if not result[0] and settings.rescue_unconverged_scf:
                if accept_center:
                    result = self._rescue_unconverged_scf(result)
                elif start_rotation is not None:
                    result = self._converge_on_center_branch(
                        solver, result, start_rotation, start_occ
                    )

            if (result[0]
                    and (accept_center or self.last_center_followed)
                    and settings.follow_negative_mode):
                result, followed = self._follow_negative_mode(result)
                if accept_center:
                    self.last_center_followed = bool(followed)
        else:
            if start_rotation is None or start_occ is None:
                raise RuntimeError(
                    "the accepted active state could not be transported"
                )
            coeff, occ, h1e, vhf, eri = solver._start_frame(
                solver.mo_coeff[:, solver.active_orb],
                start_rotation,
                start_occ,
            )
            result = [
                True,
                float(solver.energy_elec(h1e, vhf, occ)),
                np.zeros_like(occ),
                coeff,
                occ,
                h1e,
                eri,
            ]

        if not result[0]:
            raise SCFNotConverged(
                "frozen-bath SCF on H_act^V did not converge", reason=reason
            )

        if any(
            value is None
            for value in (solver.E1e_core, solver.E2e_core, solver.Enuc)
        ):
            raise RuntimeError("the full H_act^V objective requires core energies")
        energy = float(
            result[1] + solver.E1e_core + solver.E2e_core + solver.Enuc
        )

        if accept_center:
            self.pristine_reference = next_pristine_reference
            accepted_coeff = np.asarray(result[3]).real.copy()
            self.active_solution_reference = ActiveSolutionReference(
                cell=solver.scell,
                active_coeff=accepted_coeff,
                active_occ=np.asarray(result[4]).real.copy(),
                defect_index=defect_orbital_index(solver, accepted_coeff),
            )
            if self.fixed_n_bath is None:
                self.fixed_n_bath = int(solver.n_bath)
            if self.fixed_fragment_atoms is None:
                self.fixed_fragment_atoms = tuple(sorted(solver.frag_atoms))
            self.cache.clear()
            self.last_accepted_key = key

        if relax_active:
            self.cache[key] = energy
        print("# %-26s E=% .12f Ha" % (reason, energy), flush=True)
        gc.collect()
        return energy
