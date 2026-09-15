from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import time
from typing import Any, Sequence

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
from hact_relax.errors import (
    RootContinuityError,
    SCFNotConverged,
    SubspaceContinuityError,
)
from hact_relax.geometry import restored_pristine_atoms
from hact_relax.recording import NullRecorder
from hact_relax.tracking import (
    ActiveSolutionReference,
    CASCIReference,
    PristineReference,
    align_pristine_occupied_subspace,
    defect_orbital_index,
    casci_energy,
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
    auxbasis: str | None = None
    follow_negative_mode: bool = False
    follow_mode_tol: float = 1e-8
    rescue_unconverged_scf: bool = False
    rescue_tol_factor: float = 5.0
    checkpoint_eri: bool = True
    # --state excited: CASCI on H_act^V instead of the frozen-bath SCF energy.
    # Defaults reproduce the sibling repository's own defaults.
    state: str = "ground"
    root: int = 1
    nroots: int = 4
    ncas: int = 0
    ncas_elec: int = 0
    casci_two_s: int | None = None
    min_casci_root_overlap: float = 0.5


class ActiveHamiltonianSurface:

    def __init__(
        self,
        settings: SurfaceSettings,
        labels: Sequence[str],
        vacancy_index: int,
        lattice_bohr: np.ndarray,
        recorder=None,
    ):
        self.settings = settings
        self.recorder = (
            NullRecorder() if recorder is None else recorder
        )
        self.case = CASES[settings.case_name]
        self.labels = list(labels)
        self.vacancy_index = int(vacancy_index)
        self.lattice_bohr = np.asarray(lattice_bohr, dtype=float)
        self.cache: dict[str, float] = {}
        self.last_accepted_key: str | None = None
        self.pristine_reference: PristineReference | None = None
        self.active_solution_reference: ActiveSolutionReference | None = None
        self.casci_reference: CASCIReference | None = None
        self.fixed_n_bath: int | None = None
        self.fixed_fragment_atoms: tuple[int, ...] | None = None
        self.last_center_followed = False
        self.last_energy_terms: dict[str, float] | None = None
        self.measured_noise_floor: float | None = None

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
            return result, {"scf_rescued": False}
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
            return result, {
                "scf_rescued": False,
                "scf_rescue_gradient_norm": gradient,
            }
        updated = self._apply_orbital_rotation(
            result, kappa, float(energy_at(h1e, eri, occ, kappa))
        )
        updated[0] = True
        return updated, {
            "scf_rescued": True,
            "scf_rescue_gradient_norm": gradient,
        }

    def _converge_on_center_branch(self, solver, result, start_rotation, start_occ):
        if start_rotation is None or start_occ is None:
            return result, {"scf_rescued_on_center_branch": False}
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
            return result, {
                "scf_rescued_on_center_branch": False,
                "scf_rescue_gradient_norm": gradient,
            }
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
        return updated, {
            "scf_rescued_on_center_branch": True,
            "scf_rescue_gradient_norm": gradient,
        }

    def _follow_negative_mode(self, result):
        h1e = np.asarray(result[5]).real
        eri = np.asarray(result[6]).real
        occ = np.asarray(result[4]).real
        report = so.stability_analysis(h1e, eri, occ)
        info = {
            "negative_mode_eigenvalue": float(report["lowest"]),
            "negative_mode_followed": False,
            "negative_mode_energy_drop_hartree": None,
        }
        if report["lowest"] > -abs(self.settings.follow_mode_tol):
            return result, False, info

        before = energy_at(h1e, eri, occ)
        _, kappa = minimize_orbitals(
            h1e, eri, occ, 0.2 * np.asarray(report["lowest_mode"], dtype=float)
        )
        after = energy_at(h1e, eri, occ, kappa)
        info["negative_mode_energy_drop_hartree"] = float(after - before)
        if not (after < before - abs(self.settings.follow_mode_tol)):
            return result, False, info
        info["negative_mode_followed"] = True
        return self._apply_orbital_rotation(result, kappa, after), True, info

    def _embedding_record(self, solver) -> dict[str, Any]:
        singular = np.asarray(
            getattr(solver, "bath_singular_values", None)
            if getattr(solver, "bath_singular_values", None) is not None
            else [], dtype=float
        )
        n_bath = int(solver.n_bath)
        kept_min = float(singular[:n_bath].min()) if n_bath else None
        dropped_max = (
            float(singular[n_bath:].max()) if singular.size > n_bath else None
        )
        active = np.asarray(solver.active_orb, dtype=bool)
        return {
            "n_entangled_bath": n_bath,
            "threshold_n_entangled_bath": (
                None if self.fixed_n_bath is None else int(self.fixed_n_bath)
            ),
            "bath_rank_locked": self.fixed_n_bath is not None,
            "bath_tol": float(self.settings.bath_tol),
            "bath_singular_value_kept_min": kept_min,
            "bath_singular_value_dropped_max": dropped_max,
            "bath_cut_ratio": (
                None if not kept_min or dropped_max is None
                else dropped_max / kept_min
            ),
            "fragment_atoms": [int(a) for a in sorted(solver.frag_atoms)],
            "fragment_atoms_locked": self.fixed_fragment_atoms is not None,
            "n_active_orbitals": int(active.sum()),
            "n_active_electrons": int(round(float(
                np.asarray(solver.mo_occ)[active].sum()
            ))),
        }

    def _write_hact_checkpoint(
        self, path, solver, result, energy, coords_bohr, key, evaluation
    ) -> bool:
        import h5py

        singular = getattr(solver, "bath_singular_values", None)
        with h5py.File(path, "w") as handle:
            group = handle.create_group("hact")
            group["case"] = self.settings.case_name
            group["fragment"] = int(self.settings.fragment)
            group["geometry_key"] = key
            group["evaluation"] = int(evaluation)
            group["converged"] = int(bool(result[0]))
            group["kmesh"] = np.asarray(self.settings.reference_kmesh, dtype=int)
            group["coords_bohr"] = np.asarray(coords_bohr, dtype=float)
            group["labels"] = np.asarray(self.labels, dtype=h5py.special_dtype(vlen=str))
            group["e_tot"] = float(energy)
            group["e_elec_active"] = float(result[1])
            group["e1e_core"] = float(solver.E1e_core)
            group["e2e_core"] = float(solver.E2e_core)
            group["enuc"] = float(solver.Enuc)
            group["core_energy_computed"] = 1
            group["mo_coeff_active"] = np.asarray(result[3]).real
            group["mo_occ_active"] = np.asarray(result[4]).real
            group["mo_energy_active"] = np.asarray(result[2]).real
            group["h1e_active"] = np.asarray(result[5]).real
            if self.settings.checkpoint_eri and result[6] is not None:
                group["eri_active"] = np.asarray(result[6]).real
            group["mo_coeff_embedding"] = np.asarray(solver.mo_coeff).real
            group["mo_occ_embedding"] = np.asarray(solver.mo_occ).real
            group["active_orb"] = np.asarray(solver.active_orb, dtype=np.int8)
            group["n_bath"] = int(solver.n_bath)
            group["bath_rank_locked"] = int(self.fixed_n_bath is not None)
            group["fragment_atoms"] = np.asarray(
                sorted(solver.frag_atoms), dtype=int
            )
            if singular is not None:
                group["bath_singular_values"] = np.asarray(singular, dtype=float)
        return True

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
        started = time.time()
        evaluation = self.recorder.next_evaluation()

        key = self._key(coords_bohr)
        cached = self.cache.get(key) if relax_active else None
        if (cached is not None
                and (not accept_center or self.last_accepted_key == key)):
            self.recorder.evaluation({
                "event": "energy_cache_hit",
                "evaluation": evaluation,
                "reason": reason,
                "key": key,
                "energy_hartree": float(cached),
                "accept_center": bool(accept_center),
                "relax_active": bool(relax_active),
                "displaced_atom": (None if displaced_atom is None
                                   else int(displaced_atom)),
                "wall_seconds": round(time.time() - started, 6),
            })
            return cached

        record: dict[str, Any] = {
            "event": "energy",
            "evaluation": evaluation,
            "reason": reason,
            "key": key,
            "case_name": settings.case_name,
            "fragment": settings.fragment,
            "accept_center": bool(accept_center),
            "relax_active": bool(relax_active),
            "strict_continuation": bool(strict_continuation),
            "displaced_atom": (None if displaced_atom is None
                               else int(displaced_atom)),
            "pristine_kmesh": list(settings.reference_kmesh),
            "pristine_reference_policy": "exact",
            "checkpoint_saved": False,
        }

        prior_pristine = self.pristine_reference
        pristine_scf: dict[str, Any] = {
            "cycles": 0, "delta": None, "gradient": None
        }

        def pristine_callback(envs):
            pristine_scf["cycles"] = int(envs["cycle"]) + 1
            pristine_scf["delta"] = float(envs["e_tot"] - envs["last_hf_e"])
            pristine_scf["gradient"] = float(envs["norm_gorb"])
            self.recorder.scf_iteration({
                "scf_kind": "pristine_krhf",
                "evaluation": evaluation,
                "reason": reason,
                "cycle": pristine_scf["cycles"],
                "energy_hartree": float(envs["e_tot"]),
                "delta_energy_hartree": pristine_scf["delta"],
                "orbital_gradient_norm": pristine_scf["gradient"],
                "density_change_norm": float(envs["norm_ddm"]),
            })

        with self.recorder.stage(
            "pristine_scf", evaluation=evaluation, reason=reason
        ):
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
                callback=pristine_callback,
                chkfile=(self.recorder.pristine_chkfile
                         if accept_center else None),
            )
        record.update({
            "pristine_energy_hartree": float(pristine_kmf.e_tot),
            "pristine_scf_converged": bool(pristine_kmf.converged),
            "pristine_scf_cycles": pristine_scf["cycles"],
            "pristine_scf_final_delta_energy_hartree": pristine_scf["delta"],
            "pristine_scf_final_orbital_gradient_norm": pristine_scf["gradient"],
            "pristine_guess_from_previous_center": prior_pristine is not None,
        })
        if pristine_scf["delta"] is not None:
            noise = abs(float(pristine_scf["delta"]))
            self.measured_noise_floor = max(
                noise, self.measured_noise_floor or 0.0
            )
            record["energy_noise_floor_hartree"] = noise
            record["suggested_fd_step_bohr"] = (3.0 * noise) ** (1.0 / 3.0)

        occupied_match = align_pristine_occupied_subspace(
            pristine_kmf, prior_pristine
        )
        record["pristine_occupied_transport_fidelity"] = (
            None if occupied_match is None else float(occupied_match.fidelity)
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
        )
        with self.recorder.stage(
            "embedding_build", evaluation=evaluation, reason=reason
        ):
            if self.case.spin == 0:
                solver = SchmidtEmbeddedRHF(
                    pristine_kmf, settings.reference_kmesh, **common
                )
            else:
                solver = SchmidtEmbeddedROHF(
                    pristine_kmf, settings.reference_kmesh,
                    spin=self.case.spin, **common
                )
        record.update(self._embedding_record(solver))

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
        record["active_transport_fidelity"] = None
        record["active_transport_fallback"] = False
        if self.active_solution_reference is not None:
            transport = transport_active_solution(
                self.active_solution_reference, solver
            )
            start_rotation = transport.rotation
            start_occ = transport.occupations
            record["active_transport_fidelity"] = float(transport.fidelity)
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
                record["active_transport_fallback"] = True

        hact_scf: dict[str, Any] = {
            "cycles": 0, "delta": None, "gradient": None
        }

        def hact_callback(cycle, energy_elec, delta, gradient_norm):
            hact_scf["cycles"] = int(cycle)
            hact_scf["delta"] = float(delta)
            hact_scf["gradient"] = float(gradient_norm)
            self.recorder.scf_iteration({
                "scf_kind": "hact_frozen_bath",
                "evaluation": evaluation,
                "reason": reason,
                "cycle": int(cycle),
                "energy_hartree": float(energy_elec),
                "delta_energy_hartree": float(delta),
                "orbital_gradient_norm": float(gradient_norm),
            })

        solver.scf_callback = hact_callback
        record.update({
            "scf_rescued": False,
            "scf_rescued_on_center_branch": False,
            "scf_rescue_gradient_norm": None,
            "negative_mode_eigenvalue": None,
            "negative_mode_followed": False,
            "negative_mode_energy_drop_hartree": None,
            "hact_scf_continued_from_center": start_rotation is not None,
        })

        if relax_active:
            with self.recorder.stage(
                "hact_scf", evaluation=evaluation, reason=reason
            ):
                result = solver.kernel(
                    conv_tol=settings.scf_conv_tol,
                    conv_tol_grad=settings.scf_conv_tol_grad,
                    start_rotation=start_rotation,
                    start_occ=start_occ,
                )

                if not result[0] and settings.rescue_unconverged_scf:
                    if accept_center:
                        result, rescue = self._rescue_unconverged_scf(result)
                        record.update(rescue)
                    elif start_rotation is not None:
                        result, rescue = self._converge_on_center_branch(
                            solver, result, start_rotation, start_occ
                        )
                        record.update(rescue)

                if (result[0]
                        and (accept_center or self.last_center_followed)
                        and settings.follow_negative_mode):
                    result, followed, mode = self._follow_negative_mode(result)
                    record.update(mode)
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

        record.update({
            "hact_scf_converged": bool(result[0]),
            "hact_scf_iterations_profiled": hact_scf["cycles"],
            "hact_scf_final_delta_energy_hartree": hact_scf["delta"],
            "hact_scf_final_orbital_gradient_norm": hact_scf["gradient"],
        })

        if not result[0]:
            self.recorder.evaluation({
                **record,
                "event": "energy_failed",
                "error": "frozen-bath SCF on H_act^V did not converge",
                "wall_seconds": round(time.time() - started, 6),
            })
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
        record.update({
            "surface": settings.state,
            "energy_hartree": energy,
            "e_total_hact_v_hartree": energy,
            "e_active_hartree": float(result[1]),
            "e1e_core_hartree": float(solver.E1e_core),
            "e2e_core_hartree": float(solver.E2e_core),
            "enuc_hartree": float(solver.Enuc),
            "core_energy_computed": True,
            "n_active_orbitals": int(np.sum(solver.active_orb)),
            "n_active_electrons": (
                int(sum(solver.nelecas)) if np.ndim(solver.nelecas)
                else int(solver.nelecas)
            ),
        })

        casci = None
        if settings.state != "ground":
            if not relax_active and self.casci_reference is None:
                # Without a tracked reference casci_energy orders the CAS window
                # by SCF orbital energy -- and a frozen frame carries none
                # (result[2] is zeros), so it would silently pick an arbitrary
                # window and return a plausible number.
                raise RuntimeError(
                    "a semi-analytic CASCI displacement requires an accepted "
                    "centre with a tracked CI reference"
                )
            # CASCI replaces the SCF active energy; the core terms are folded in
            # by _fake_mf via energy_nuc, so casci.energy is already the total.
            with self.recorder.stage(
                "casci", evaluation=evaluation, reason=reason
            ):
                casci = casci_energy(
                    solver,
                    result,
                    ncas=settings.ncas,
                    ncas_elec=settings.ncas_elec,
                    two_s=(self.case.spin if settings.casci_two_s is None
                           else settings.casci_two_s),
                    root=settings.root,
                    nroots=settings.nroots,
                    reference=self.casci_reference,
                )
            selected_overlap = (
                None if self.casci_reference is None
                else float(casci.root_overlaps[casci.selected_root])
            )
            record.update({
                "energy_hartree": float(casci.energy),
                "casci_energy_hartree": float(casci.energy),
                "casci_s_squared": float(casci.s_squared),
                "casci_root_energies": casci.root_energies.tolist(),
                "casci_root_s_squared": casci.root_s_squared.tolist(),
                "casci_two_s": int(self.case.spin if settings.casci_two_s is None
                                   else settings.casci_two_s),
                "casci_ncas": int(settings.ncas),
                "casci_ncas_elec": int(settings.ncas_elec),
                "cas_ncore": int(casci.ncore),
                "requested_root": int(casci.requested_root),
                "selected_root": int(casci.selected_root),
                "root_overlaps": casci.root_overlaps.tolist(),
                "selected_root_overlap": selected_overlap,
                "cas_orbital_min_overlap": casci.orbital_min_overlap,
            })

            # Guard 1 -- the active orbitals must still be the same frame, or the
            # CI vectors being compared are not comparable.
            if (casci.orbital_min_overlap is not None
                    and casci.orbital_min_overlap
                    < settings.min_subspace_overlap):
                self.recorder.evaluation({
                    **record,
                    "event": "energy_failed",
                    "error": "CASCI active-orbital continuity lost",
                    "wall_seconds": round(time.time() - started, 6),
                })
                raise SubspaceContinuityError(
                    "minimum tracked active-orbital overlap %.6f is below %.6f"
                    % (casci.orbital_min_overlap, settings.min_subspace_overlap),
                    continuity_space="casci-active-orbitals",
                )
            # Guard 2 -- some root must still carry the tracked state.
            if (selected_overlap is not None
                    and selected_overlap < settings.min_casci_root_overlap):
                self.recorder.evaluation({
                    **record,
                    "event": "energy_failed",
                    "error": "CASCI root continuity lost",
                    "wall_seconds": round(time.time() - started, 6),
                })
                raise RootContinuityError(
                    "best CASCI root overlap %.6f is below %.6f: the tracked "
                    "state is no longer among the %d computed roots"
                    % (selected_overlap, settings.min_casci_root_overlap,
                       len(casci.root_energies)),
                    selected_root=int(casci.selected_root),
                    root_overlaps=casci.root_overlaps.tolist(),
                    min_casci_root_overlap=float(settings.min_casci_root_overlap),
                )
            energy = float(casci.energy)
        if accept_center:
            self.last_energy_terms = {
                "e_active_hartree": float(result[1]),
                "e1e_core_hartree": float(solver.E1e_core),
                "e2e_core_hartree": float(solver.E2e_core),
                "enuc_hartree": float(solver.Enuc),
                # The relaxed surface and the underlying SCF total differ once
                # --state excited is in play.
                "e_total_hact_v_hartree": float(
                    result[1] + solver.E1e_core + solver.E2e_core + solver.Enuc
                ),
                "e_casci_hartree": (None if casci is None
                                    else float(casci.energy)),
            }

        if accept_center:
            self.pristine_reference = next_pristine_reference
            accepted_coeff = np.asarray(result[3]).real.copy()
            self.active_solution_reference = ActiveSolutionReference(
                cell=solver.scell,
                active_coeff=accepted_coeff,
                active_occ=np.asarray(result[4]).real.copy(),
                defect_index=defect_orbital_index(solver, accepted_coeff),
            )
            if casci is not None:
                if casci.ordered_active_coeff is None:
                    raise RuntimeError(
                        "CASCI did not return active orbitals for root tracking"
                    )
                self.casci_reference = CASCIReference(
                    cell=solver.scell,
                    ordered_active_coeff=casci.ordered_active_coeff.copy(),
                    ci_vector=casci.ci_vectors[casci.selected_root].copy(),
                )
            if self.fixed_n_bath is None:
                self.fixed_n_bath = int(solver.n_bath)
            if self.fixed_fragment_atoms is None:
                self.fixed_fragment_atoms = tuple(sorted(solver.frag_atoms))
            self.cache.clear()
            self.last_accepted_key = key
            if self.recorder.hact_chkfile is not None:
                with self.recorder.stage(
                    "checkpoint_write", evaluation=evaluation, reason=reason
                ):
                    record["checkpoint_saved"] = self._write_hact_checkpoint(
                        self.recorder.hact_chkfile, solver, result, energy,
                        coords_bohr, key, evaluation,
                    )

        if relax_active:
            self.cache[key] = energy
        record["wall_seconds"] = round(time.time() - started, 6)
        self.recorder.evaluation(record)
        print("# %-26s E=% .12f Ha" % (reason, energy), flush=True)
        gc.collect()
        return energy
