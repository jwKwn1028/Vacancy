from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import time
from typing import Any, Sequence

import numpy as np

from frozen_bath_scf import second_order as so
from frozen_bath_scf.df_eri import as_eri
from frozen_bath_scf.orbital_optimizer import (
    canonical_block_rotation,
    energy_at,
    minimize_orbitals,
    newton_finish,
    rotate_hamiltonian,
)
from frozen_bath_scf.schmidt_embedding import (
    SchmidtEmbeddedRHF,
    SchmidtEmbeddedROHF,
)
from system.pristine import build_pristine_mean_field
from system.shells import coordination_shells, nearest_atoms
from system.fragment_fit import FragmentFit, KpointFit
from system.split_fit import KpointOutside

from hact_relax.cases import CASES
from hact_relax.checkpoint import write_hact_checkpoint, write_hact_input_checkpoint
from hact_relax.errors import (
    RootContinuityError,
    SCFNotConverged,
    SubspaceContinuityError,
)
from hact_relax.geometry import ghost_species, ideal_supercell, restored_pristine_atoms
from hact_relax.recording import NullRecorder
from hact_relax.symmetry import (
    active_representation,
    rotation_group,
    site_rotation,
    symmetric_frame,
)
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


# Open-shell weight spread over equivalent atoms above which a centre warns.
OPEN_SHELL_SPREAD_WARNING = 0.05
EQUIVALENT_ATOM_TOL_BOHR = 1e-3


def apply_orbital_rotation(result, kappa, energy, *, canonicalize=True, info=None,
                           rotation=None):
    """``result`` with its active orbitals rotated by exp(kappa), or by ``rotation``."""
    h1e = np.asarray(result[5]).real
    eri = as_eri(result[6])
    occ = np.asarray(result[4]).real
    u = so._expm_antisym(kappa) if rotation is None else np.asarray(rotation)
    dm_a, dm_b = so.densities(occ)
    v_a, v_b = so.general_veff(eri, u @ dm_a @ u.T, u @ dm_b @ u.T)
    fock = so._rotate(h1e + 0.5 * (v_a + v_b), u)
    if canonicalize:
        block, mo_energy, off_diagonal = canonical_block_rotation(fock, occ)
        u = u @ block
    else:
        mo_energy, off_diagonal = np.diag(fock).copy(), None
    if info is not None:
        info["orbitals_canonicalized"] = bool(canonicalize)
        info["canonical_offdiagonal_fock_hartree"] = off_diagonal
    h1e_new, eri_new = rotate_hamiltonian(h1e, eri, u)
    updated = list(result)
    updated[1] = float(energy)
    updated[2] = mo_energy
    updated[3] = np.asarray(result[3]).real @ u
    updated[5] = h1e_new
    updated[6] = eri_new
    return updated


def minimize_and_finish(h1e, eri, occ, x0, newton_tol, escapes=2, ops=None):
    """L-BFGS from ``x0``, then a Newton finish to ``newton_tol``; at a saddle the lowest
    mode is followed both ways, at most ``escapes`` times.  ``ops`` (a group in the frame
    of h1e) restricts all of it to invariant rotations.  Returns the rotation, energy,
    gradient norm and record."""
    project = so.symmetry_projector(ops, occ)
    keep = (lambda x: x) if project is None else project
    _, kappa = minimize_orbitals(h1e, eri, occ, x0, project=project)
    u = so._expm_antisym(kappa)
    h1e, eri = rotate_hamiltonian(h1e, eri, u)
    lbfgs = float(np.linalg.norm(keep(so.orbital_gradient(h1e, eri, occ))))
    v, h1e_n, eri_n, newton = newton_finish(h1e, eri, occ, newton_tol, project=project)
    mode = newton.pop("lowest_mode", None)
    newton["lbfgs_gradient_norm"] = lbfgs
    newton["saddle_escapes"] = 0
    if mode is None or escapes <= 0:
        return (u @ v, float(energy_at(h1e_n, eri_n, occ)),
                newton["newton_gradient_after"], newton)
    step = 0.2 * np.asarray(mode, dtype=float)
    ends = [minimize_and_finish(h1e, eri, occ, sign * step, newton_tol, escapes - 1, ops)
            for sign in (1, -1)]
    lower = int(ends[1][1] < ends[0][1] - 1e-10)
    w, energy, gradient, record = ends[lower]
    record = dict(record)
    record["lbfgs_gradient_norm"] = lbfgs
    record["saddle_escapes"] += 1
    record["saddle_lowest_hessian"] = newton["newton_lowest_hessian"]
    record["saddle_branch_gap_hartree"] = float(ends[1 - lower][1] - energy)
    return u @ w, energy, gradient, record


def rescue_unconverged(result, gradient_limit, newton_tol=None, ops=None):
    """Finish an unconverged frozen-bath SCF (minimize_and_finish); accepted, canonical, if
    its gradient is at most ``gradient_limit``."""
    if result[5] is None or result[6] is None:
        return result, {"scf_rescued": False}
    h1e = np.asarray(result[5]).real
    eri = as_eri(result[6])
    occ = np.asarray(result[4]).real
    u, energy, gradient, newton = minimize_and_finish(
        h1e, eri, occ, np.zeros(int(so.rotation_mask(occ).sum())),
        gradient_limit if newton_tol is None else newton_tol, ops=ops,
    )
    info = {"scf_rescued": gradient <= gradient_limit,
            "scf_rescue_gradient_norm": gradient,
            "scf_rescue_lbfgs_gradient_norm": newton["lbfgs_gradient_norm"],
            "scf_rescue_newton_cycles": newton["newton_cycles"],
            "scf_rescue_newton_stop": newton["newton_stop"],
            "scf_rescue_saddle_escapes": newton["saddle_escapes"]}
    if not info["scf_rescued"]:
        return result, info
    updated = apply_orbital_rotation(result, None, energy, info=info, rotation=u)
    updated[0] = True
    return updated, info


def follow_negative_mode(result, tol, newton_tol=1e-9, ops=None):
    """Follow the lowest orbital-Hessian mode below -``tol`` (among invariant rotations,
    with ``ops``) both ways and keep the lower minimum, canonical.  Returns the result,
    whether a mode was followed, and the record."""
    h1e = np.asarray(result[5]).real
    eri = as_eri(result[6])
    occ = np.asarray(result[4]).real
    report = so.stability_analysis(h1e, eri, occ,
                                   project=so.symmetry_projector(ops, occ))
    info = {
        "negative_mode_eigenvalue": float(report["lowest"]),
        "negative_mode_followed": False,
        "negative_mode_energy_drop_hartree": None,
    }
    if report["lowest"] > -abs(tol):
        return result, False, info

    before = energy_at(h1e, eri, occ)
    mode = 0.2 * np.asarray(report["lowest_mode"], dtype=float)
    ends = [minimize_and_finish(h1e, eri, occ, sign * mode, newton_tol, ops=ops)
            for sign in (1, -1)]
    lower = int(ends[1][1] < ends[0][1] - 1e-10)
    u, after, gradient, newton = ends[lower]
    info.update({
        "negative_mode_direction": (1, -1)[lower],
        "negative_mode_branch_gap_hartree": float(ends[1 - lower][1] - after),
        "negative_mode_energy_drop_hartree": float(after - before),
        "negative_mode_gradient_norm": gradient,
        "negative_mode_lbfgs_gradient_norm": newton["lbfgs_gradient_norm"],
        "negative_mode_newton_cycles": newton["newton_cycles"],
        "negative_mode_newton_stop": newton["newton_stop"],
        "negative_mode_saddle_escapes": newton["saddle_escapes"],
    })
    if not (after < before - abs(tol)):
        return result, False, info
    info["negative_mode_followed"] = True
    return (apply_orbital_rotation(result, None, after, info=info, rotation=u), True,
            info)


def symmetric_resolve(result, ops, newton_tol, tol=1e-10, constrained=False):
    """Re-minimize from a symmetric frame (``constrained``: within invariant rotations) and
    keep the lower minimum, canonical."""
    h1e = np.asarray(result[5]).real
    eri = as_eri(result[6])
    occ = np.asarray(result[4]).real
    before = float(energy_at(h1e, eri, occ))
    frame, report = symmetric_frame(h1e, eri, occ, ops)
    info = {"symmetric_start_order": len(ops),
            "symmetric_start_frontier_energies": report["frontier_energies"],
            "symmetric_start_kept": False}
    if frame is None:
        info["symmetric_start_skipped"] = "no symmetric occupation"
        return result, False, info
    h, e = rotate_hamiltonian(h1e, eri, frame)
    u, after, gradient, newton = minimize_and_finish(
        h, e, occ, np.zeros(int(so.rotation_mask(occ).sum())), newton_tol,
        ops=[frame.T @ w @ frame for w in ops] if constrained else None)
    info.update({
        "symmetric_start_energy_change_hartree": float(after - before),
        "symmetric_start_gradient_norm": gradient,
        "symmetric_start_saddle_escapes": newton["saddle_escapes"],
        # From a symmetric saddle the two ends are mirror images: their gap is the splitting.
        "symmetric_start_branch_gap_hartree": newton.get("saddle_branch_gap_hartree"),
        "symmetric_start_singly_occupied_character": report["singly_occupied_character"],
    })
    if not (after < before - abs(tol) and gradient <= newton_tol):
        return result, False, info
    info["symmetric_start_kept"] = True
    return (apply_orbital_rotation(result, None, after, info=info, rotation=frame @ u), True,
            info)


def occupation_breaking(ops, occ):
    """Largest change ``ops`` make to the frame's densities (0: invariant)."""
    dm_a, dm_b = so.densities(occ)
    return float(max(np.abs(w @ d @ w.T - d).max() for w in ops for d in (dm_a, dm_b)))


def symmetric_solution(result, ops, newton_tol, gradient_limit):
    """``result`` if it converged to an ``ops``-invariant determinant, else the minimum
    within invariant rotations from symmetric_frame, accepted, canonical, if its gradient
    is at most ``gradient_limit``.  Returns the result and a record."""
    occ = np.asarray(result[4]).real
    breaking = occupation_breaking(ops, occ)
    info = {"symmetry_occupation_breaking": breaking, "symmetry_resolved": False}
    if result[0] and breaking < 1e-8:
        return result, info
    h1e = np.asarray(result[5]).real
    eri = as_eri(result[6])
    frame, report = symmetric_frame(h1e, eri, occ, ops)
    if frame is None:
        info["symmetry_resolve_skipped"] = "no symmetric occupation"
        updated = list(result)
        updated[0] = False
        return updated, info
    h, e = rotate_hamiltonian(h1e, eri, frame)
    u, energy, gradient, newton = minimize_and_finish(
        h, e, occ, np.zeros(int(so.rotation_mask(occ).sum())), newton_tol,
        ops=[frame.T @ w @ frame for w in ops])
    info.update({"symmetry_resolved": gradient <= gradient_limit,
                 "symmetry_resolve_gradient_norm": gradient,
                 "symmetry_resolve_newton_stop": newton["newton_stop"],
                 "symmetry_resolve_saddle_escapes": newton["saddle_escapes"]})
    if not info["symmetry_resolved"]:
        updated = list(result)
        updated[0] = False
        return updated, info
    updated = apply_orbital_rotation(result, None, energy, info=info, rotation=frame @ u)
    updated[0] = True
    return updated, info


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
    follow_negative_mode: bool = True
    follow_mode_tol: float = 1e-8
    rescue_unconverged_scf: bool = True
    rescue_tol_factor: float = 5.0
    # First centre: also minimize from a symmetric start, keep the lower.
    symmetric_start: bool = True
    # Every centre: hold the SCF on states invariant under the rotation about the vacancy.
    constrain_symmetry: bool = False
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
    cas_weight_gap_tol: float = 1e-3
    cas_weight_floor: float = 0.1
    allow_degenerate_cas: bool = False
    cas_selection: str = "vacancy-local"
    cas_energy_gap_tol: float = 1e-4
    wrap_about_vacancy: bool = False
    primitive_atoms: list | None = None
    primitive_lattice_bohr: np.ndarray | None = None
    split_kmesh: tuple[int, int, int] | None = None
    split_primitive_atoms: list | None = None
    split_primitive_lattice_bohr: np.ndarray | None = None
    # "fragment": the exact Gamma fit from the k-point one; "gamma": Gamma inside only.
    split_fit: str = "fragment"
    # Thresholds of the ideal crystal's k-point SCF; None takes scf_conv_tol(_grad).
    split_kpoint_conv_tol: float | None = None
    split_kpoint_conv_tol_grad: float | None = None
    # Precision of the fragment fit's moving parts; None takes the cell's.
    split_fit_precision: float | None = 1e-12
    # Atoms the fragment fit treats as moved at every geometry.
    split_fit_explicit_atoms: tuple[int, ...] = ()
    # Worker threads of the fit's k-space exchange; 0 takes PySCF's thread count.
    split_exchange_workers: int = 0


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
        # energy(keep_point=True) keeps the accepted centre's mean field and solver.
        self.kept_centre: dict | None = None
        self.pristine_reference: PristineReference | None = None
        self.active_solution_reference: ActiveSolutionReference | None = None
        self.casci_reference: CASCIReference | None = None
        self.fixed_n_bath: int | None = None
        self.fixed_fragment_atoms: tuple[int, ...] | None = None
        self.last_center_followed = False
        self.last_energy_terms: dict[str, float] | None = None
        self.measured_noise_floor: float | None = None
        self.ideal_coords_bohr: np.ndarray | None = None
        self.population_reference_coords: np.ndarray | None = None
        if settings.primitive_atoms is not None:
            _, self.ideal_coords_bohr = ideal_supercell(
                settings.primitive_atoms, settings.primitive_lattice_bohr,
                settings.reference_kmesh,
            )
        self.split_outside: KpointOutside | None = None
        self.split_ideal_coords: np.ndarray | None = None
        self.split_atoms: tuple[int, ...] | None = None
        if settings.split_fit not in ("gamma", "fragment"):
            raise ValueError("split_fit must be 'gamma' or 'fragment', got %r"
                             % (settings.split_fit,))
        if settings.split_kmesh is not None:
            symbols, self.split_ideal_coords = ideal_supercell(
                settings.split_primitive_atoms, settings.split_primitive_lattice_bohr,
                settings.split_kmesh,
            )
            if [ghost_species(x).lower() for x in self.labels] != [
                    str(x).lower() for x in symbols]:
                raise ValueError("--split: the geometry does not list its atoms in "
                                 "the order of the ideal k-mesh supercell")
            self.split_atoms = tuple(int(i) for i in nearest_atoms(
                self.split_ideal_coords, self.lattice_bohr, self.vacancy_index,
                settings.fragment,
            ))

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

    _apply_orbital_rotation = staticmethod(apply_orbital_rotation)

    def _split_moved_atoms(self, coords_bohr: np.ndarray) -> list[int]:
        shift = np.asarray(coords_bohr) - self.split_ideal_coords
        frac = shift @ np.linalg.inv(self.lattice_bohr)
        shift = (frac - np.round(frac)) @ self.lattice_bohr
        return [int(i) for i in np.where(np.linalg.norm(shift, axis=1) > 1e-6)[0]]

    def _split_integrals(self, coords_bohr, reason, evaluation, record):
        moved = self._split_moved_atoms(coords_bohr)
        outside = sorted(set(moved) - set(self.split_atoms))
        if outside:
            raise ValueError(
                "%s: --split takes the integrals outside the fragment %s from the "
                "ideal crystal, but atoms %s outside it have moved"
                % (reason, list(self.split_atoms), outside)
            )
        if self.split_outside is None:
            settings = self.settings
            fragment_fit = settings.split_fit == "fragment"
            records = [] if fragment_fit else None
            with self.recorder.stage(
                "split_kpoint_scf", evaluation=evaluation, reason=reason
            ):
                _, kmf = build_pristine_mean_field(
                    settings.split_primitive_atoms,
                    settings.split_primitive_lattice_bohr,
                    settings.basis,
                    kmesh=settings.split_kmesh,
                    exxdiv=None,
                    verbose=settings.verbose,
                    max_memory=settings.memory_mb,
                    conv_tol=(settings.scf_conv_tol
                              if settings.split_kpoint_conv_tol is None
                              else settings.split_kpoint_conv_tol),
                    conv_tol_grad=(settings.scf_conv_tol_grad
                                   if settings.split_kpoint_conv_tol_grad is None
                                   else settings.split_kpoint_conv_tol_grad),
                    max_cycle=settings.scf_max_cycle,
                    auxbasis=settings.auxbasis,
                    fit_records=records,
                    floor_factor=self._floor_factor(),
                )
                if kmf.converged_at_floor is not None:
                    print("# k-point SCF of the ideal crystal accepted at its gradient floor, "
                          "|g| %.2e" % kmf.converged_at_floor, flush=True)
                self.split_outside = (
                    KpointFit(kmf, settings.split_kmesh, records) if fragment_fit
                    else KpointOutside(kmf, settings.split_kmesh)
                )
                if fragment_fit:
                    self.split_outside.workers = settings.split_exchange_workers or None
                del kmf
                gc.collect()
        record.update({
            "split_fit": self.settings.split_fit,
            "split_fit_precision": self.settings.split_fit_precision,
            "pristine_reference": "split",
            "split_kmesh": list(self.settings.split_kmesh),
            "split_fragment_atoms": list(self.split_atoms),
            "split_moved_atoms": moved,
        })
        return self.split_outside, self.split_atoms

    def _rescue_limit(self) -> float:
        return self.settings.scf_conv_tol_grad * self.settings.rescue_tol_factor

    def _rescue_unconverged_scf(self, result, ops=None):
        return rescue_unconverged(result, self._rescue_limit(),
                                  self.settings.scf_conv_tol_grad, ops)

    def _converge_on_center_branch(self, solver, result, start_rotation, start_occ):
        if start_rotation is None or start_occ is None:
            return result, {"scf_rescued_on_center_branch": False}
        u0 = np.asarray(start_rotation, dtype=float)
        occ = np.asarray(start_occ, dtype=float).copy()
        h1e = np.asarray(solver.get_hcore(solver.heff, u0)).real
        _, eri = solver.get_veff(occ, solver.eri_active, u0)
        eri = as_eri(eri)
        u, energy, gradient, newton = minimize_and_finish(
            h1e, eri, occ, np.zeros(int(so.rotation_mask(occ).sum())),
            self.settings.scf_conv_tol_grad,
        )
        record = {
            "scf_rescued_on_center_branch": gradient <= self._rescue_limit(),
            "scf_rescue_gradient_norm": gradient,
            "scf_rescue_lbfgs_gradient_norm": newton["lbfgs_gradient_norm"],
            "scf_rescue_newton_cycles": newton["newton_cycles"],
            "scf_rescue_newton_stop": newton["newton_stop"],
            "scf_rescue_saddle_escapes": newton["saddle_escapes"],
        }
        if not record["scf_rescued_on_center_branch"]:
            return result, record
        seed = [
            True,
            float(energy_at(h1e, eri, occ)),
            np.zeros_like(occ),
            np.asarray(solver.mo_coeff[:, solver.active_orb]).real @ u0,
            occ,
            h1e,
            eri,
        ]
        updated = apply_orbital_rotation(seed, None, energy, info=record, rotation=u)
        updated[0] = True
        return updated, record

    def _floor_factor(self):
        """Host SCFs accept a stalled gradient within the rescue factor, if rescuing."""
        settings = self.settings
        return settings.rescue_tol_factor if settings.rescue_unconverged_scf else None

    def _follow_negative_mode(self, result, ops=None):
        return follow_negative_mode(result, self.settings.follow_mode_tol,
                                    self.settings.scf_conv_tol_grad, ops)

    def site_symmetry(self, solver):
        """(n, AO matrix) of the rotation about the vacancy, (None, None) if none: an error
        under constrain_symmetry."""
        n, rotation = site_rotation(solver.scell, solver.S, self.vacancy_index)
        if n is None and self.settings.constrain_symmetry:
            raise ValueError("--constrain-symmetry: no rotation about the vacancy maps this "
                             "geometry onto itself")
        return n, rotation

    @staticmethod
    def active_symmetry_ops(solver, coeff, symmetry):
        """The rotation's group in the frame of ``coeff``, and the active space's invariance
        (an error below 1 - 1e-6)."""
        n, rotation = symmetry
        w, invariance = active_representation(np.asarray(coeff).real, solver.S, rotation)
        if invariance < 1.0 - 1e-6:
            raise ValueError("--constrain-symmetry: C%d does not map the active space onto "
                             "itself (invariance %.6f)" % (n, invariance))
        return rotation_group(w, n), invariance

    @staticmethod
    def _symmetry_record(result):
        """The full orbital gradient and lowest orbital-Hessian eigenvalue of the held state
        (negative: a saddle)."""
        occ = np.asarray(result[4]).real
        report = so.stability_analysis(np.asarray(result[5]).real, as_eri(result[6]), occ)
        print("# symmetry held: full orbital gradient %.1e, lowest full orbital Hessian "
              "%+.3e%s" % (report["gradient_norm"], report["lowest"],
                           " (the symmetric state is a saddle)" if report["lowest"] < 0
                           else ""), flush=True)
        return {"symmetry_full_gradient_norm": report["gradient_norm"],
                "symmetry_full_lowest_hessian": report["lowest"]}

    def _symmetric_start(self, solver, result):
        """symmetric_resolve under the rotation about the vacancy, if one exists."""
        n, rotation = self.site_symmetry(solver)
        if n is None:
            return result, {"symmetric_start_order": None}
        w, invariance = active_representation(np.asarray(result[3]).real, solver.S,
                                              rotation)
        if invariance < 1.0 - 1e-6:
            return result, {"symmetric_start_order": n, "symmetric_start_skipped":
                            "active space not invariant (%.6f)" % invariance}
        result, kept, info = symmetric_resolve(result, rotation_group(w, n),
                                               self.settings.scf_conv_tol_grad,
                                               constrained=self.settings.constrain_symmetry)
        info["symmetric_start_active_invariance"] = invariance
        change = info.get("symmetric_start_energy_change_hartree")
        print("# symmetric start (C%d about the vacancy): %s" % (
            n, info.get("symmetric_start_skipped") or "%s, %+.3e Ha from the SCF's solution"
            % ("kept" if kept else "not lower", change)), flush=True)
        return result, info

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

    def _equivalent_fragment_atoms(self, atoms) -> list[list[int]]:
        """Fragment atoms grouped by element and distance from the vacancy (ideal crystal,
        else the first geometry); groups of two or more."""
        ref = self.split_ideal_coords
        if ref is None:
            ref = self.population_reference_coords
        atoms = set(atoms) - {self.vacancy_index}
        groups = []
        for shell in coordination_shells(ref, self.lattice_bohr, self.vacancy_index,
                                         EQUIVALENT_ATOM_TOL_BOHR):
            by_element: dict[str, list[int]] = {}
            for a in shell:
                if a in atoms:
                    by_element.setdefault(ghost_species(self.labels[a]), []).append(a)
            groups.extend(g for g in by_element.values() if len(g) > 1)
        return groups

    def open_shell_record(self, solver, coeff, occ) -> dict[str, Any]:
        """Open-shell Mulliken weight per fragment atom, and its largest spread over
        equivalent atoms."""
        occ = np.asarray(occ).real
        single = np.flatnonzero(np.isclose(occ, 1.0))
        if not single.size:
            return {}
        c = np.asarray(coeff).real[:, single]
        sc = solver.S @ c
        aoslice = solver.scell.aoslice_by_atom()
        weights = {}
        for a in sorted(int(a) for a in solver.frag_atoms):
            s0, s1 = aoslice[a][2:4]
            weights[a] = float(np.einsum("ui,ui->", c[s0:s1], sc[s0:s1]))
        spread, worst = None, []
        for group in self._equivalent_fragment_atoms(weights):
            values = [weights[a] for a in group]
            if spread is None or max(values) - min(values) > spread:
                spread, worst = max(values) - min(values), group
        return {"open_shell_population": {str(a): w for a, w in weights.items()},
                "open_shell_population_spread": spread,
                "open_shell_population_worst_group": worst}

    def _checkpoint_meta(self, coords_bohr, key, evaluation) -> dict[str, Any]:
        return {
            "case": self.settings.case_name,
            "fragment": int(self.settings.fragment),
            "geometry_key": key,
            "evaluation": int(evaluation),
            "kmesh": np.asarray(self.settings.reference_kmesh, dtype=int),
            "coords_bohr": np.asarray(coords_bohr, dtype=float),
            "labels": list(self.labels),
            "bath_rank_locked": int(self.fixed_n_bath is not None),
        }

    def _write_hact_checkpoint(
        self, path, solver, result, energy, coords_bohr, key, evaluation
    ) -> bool:
        write_hact_checkpoint(
            path,
            meta=self._checkpoint_meta(coords_bohr, key, evaluation),
            solver=solver,
            result=result,
            energy=energy,
            checkpoint_eri=self.settings.checkpoint_eri,
        )
        return True

    def _write_hact_input_checkpoint(
        self, path, solver, coords_bohr, key, evaluation, start_rotation, start_occ
    ) -> bool:
        settings = self.settings
        meta = self._checkpoint_meta(coords_bohr, key, evaluation)
        meta.update({
            "basis": settings.basis,
            "lattice_bohr": np.asarray(solver.scell.lattice_vectors(), dtype=float),
            "spin": int(self.case.spin),
            "charge": int(self.case.charge),
            "vacancy_index": int(self.vacancy_index),
        })
        write_hact_input_checkpoint(
            path,
            meta=meta,
            solver=solver,
            checkpoint_eri=settings.checkpoint_eri,
            scf={
                "scf_conv_tol": float(settings.scf_conv_tol),
                "scf_conv_tol_grad": float(settings.scf_conv_tol_grad),
                "rescue_unconverged_scf": int(bool(settings.rescue_unconverged_scf)),
                "rescue_tol_factor": float(settings.rescue_tol_factor),
                "follow_negative_mode": int(bool(settings.follow_negative_mode)),
                "follow_mode_tol": float(settings.follow_mode_tol),
            },
            start_rotation=start_rotation,
            start_occ=start_occ,
        )
        return True

    def energy(
        self,
        coords_bohr: np.ndarray,
        reason: str,
        *,
        accept_center: bool = False,
        displaced_atom: int | None = None,
        relax_active: bool = True,
        keep_point: bool = False,
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
        if self.population_reference_coords is None:
            self.population_reference_coords = coords_bohr.copy()
        if self.ideal_coords_bohr is not None and not np.allclose(
            coords_bohr, self.ideal_coords_bohr, atol=1e-9, rtol=0.0
        ):
            raise ValueError(
                "%s: the k-point reference describes only the ideal lattice, "
                "and this geometry is displaced from it by up to %.3e Bohr; "
                "use the Gamma reference for displaced geometries"
                % (reason, float(np.abs(coords_bohr - self.ideal_coords_bohr).max()))
            )
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
            "pristine_reference": (
                "gamma" if self.ideal_coords_bohr is None else "kmesh"
            ),
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

        if self.ideal_coords_bohr is None:
            pristine_atoms = restored_pristine_atoms(self.labels, coords_bohr)
            pristine_lattice = self.lattice_bohr
        else:
            pristine_atoms = settings.primitive_atoms
            pristine_lattice = settings.primitive_lattice_bohr
        split = None
        if self.split_atoms is not None:
            split = self._split_integrals(coords_bohr, reason, evaluation, record)
        if prior_pristine is not None:
            dm0 = prior_pristine.density
        else:
            dm0 = None if split is None else split[0].density
        # On a k-mesh the KRHF energy is per primitive cell.
        ncells = int(np.prod(settings.reference_kmesh))
        with self.recorder.stage(
            "pristine_scf", evaluation=evaluation, reason=reason
        ):
            _, pristine_kmf = build_pristine_mean_field(
                pristine_atoms,
                pristine_lattice,
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
                dm0=dm0,
                callback=pristine_callback,
                chkfile=(self.recorder.pristine_chkfile
                         if accept_center else None),
                split=split,
                fit_precision=settings.split_fit_precision,
                fit_explicit=settings.split_fit_explicit_atoms,
                floor_factor=self._floor_factor(),
            )
        floor = getattr(pristine_kmf, "converged_at_floor", None)
        if floor is not None:
            print("# %s: host SCF accepted at its gradient floor, |g| %.2e (threshold %.0e)"
                  % (reason, floor, settings.scf_conv_tol_grad), flush=True)
        record.update({
            "pristine_energy_hartree": float(pristine_kmf.e_tot) * ncells,
            "pristine_energy_per_cell_hartree": float(pristine_kmf.e_tot),
            "pristine_scf_converged": bool(pristine_kmf.converged),
            "pristine_scf_cycles": pristine_scf["cycles"],
            "pristine_scf_final_delta_energy_hartree": pristine_scf["delta"],
            "pristine_scf_final_orbital_gradient_norm": pristine_scf["gradient"],
            "pristine_scf_floor_accepted": floor,
            "pristine_guess_from_previous_center": prior_pristine is not None,
        })
        fit = pristine_kmf.with_df
        if isinstance(fit, FragmentFit):
            record.update({
                "fragment_fit_moved_atoms": [int(a) for a in fit.moved],
                "fragment_fit_timings": {k: float(v) for k, v in fit.timings.items()},
                "fragment_fit_seam_max": fit.seam_max,
                "fragment_fit_aux_residual_min": fit.aux_residual_min,
            })
        if pristine_scf["delta"] is not None:
            noise = abs(float(pristine_scf["delta"])) * ncells
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
        record["pristine_occupied_transport_spatial_fidelity"] = (
            None if occupied_match is None else occupied_match.spatial_fidelity
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
            wrap_about_vacancy=settings.wrap_about_vacancy,
            keep_cluster_df=keep_point and accept_center,
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
                "supercell KRHF/embedding coordinates do not match the "
                "requested relaxation geometry"
            )

        start_rotation = None
        start_occ = None
        record["active_transport_fidelity"] = None
        record["active_transport_spatial_fidelity"] = None
        record["active_transport_fallback"] = False
        if self.active_solution_reference is not None:
            transport = transport_active_solution(
                self.active_solution_reference, solver
            )
            start_rotation = transport.rotation
            start_occ = transport.occupations
            record["active_transport_fidelity"] = float(transport.fidelity)
            record["active_transport_spatial_fidelity"] = (
                transport.match.spatial_fidelity)
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
            "orbitals_canonicalized": False,
            "canonical_offdiagonal_fock_hartree": None,
            "hact_scf_continued_from_center": start_rotation is not None,
            "input_checkpoint_saved": False,
        })

        # Written before the SCF, so a failed one can be re-solved by resolve_hact.py.
        input_chkfile = getattr(self.recorder, "hact_input_chkfile", None)
        if accept_center and relax_active and input_chkfile is not None:
            try:
                with self.recorder.stage(
                    "input_checkpoint_write", evaluation=evaluation, reason=reason
                ):
                    record["input_checkpoint_saved"] = (
                        self._write_hact_input_checkpoint(
                            input_chkfile, solver, coords_bohr, key, evaluation,
                            start_rotation, start_occ,
                        )
                    )
            except Exception as exc:  # noqa: BLE001 -- recorded and reported
                record["input_checkpoint_error"] = "%s: %s" % (
                    type(exc).__name__, exc)
                print("# warning: hact_input.chk was not written (%s)"
                      % record["input_checkpoint_error"], flush=True)

        if relax_active:
            symmetry = None
            if settings.constrain_symmetry:
                symmetry = self.site_symmetry(solver)
                start = np.asarray(solver.mo_coeff[:, solver.active_orb]).real
                if start_rotation is not None:
                    start = start @ np.asarray(start_rotation, dtype=float)
                solver.symmetry_ops, invariance = self.active_symmetry_ops(
                    solver, start, symmetry)
                record.update({"symmetry_constrained": True, "symmetry_order": symmetry[0],
                               "symmetry_active_invariance": invariance})

            def ops_of(result):
                return (None if symmetry is None
                        else self.active_symmetry_ops(solver, result[3], symmetry)[0])

            with self.recorder.stage(
                "hact_scf", evaluation=evaluation, reason=reason
            ):
                result = solver.kernel(
                    conv_tol=settings.scf_conv_tol,
                    conv_tol_grad=settings.scf_conv_tol_grad,
                    start_rotation=start_rotation,
                    start_occ=start_occ,
                )

                if symmetry is not None:
                    # Aufbau can split a degenerate set at the occupation boundary.
                    result, hold = symmetric_solution(
                        result, ops_of(result), settings.scf_conv_tol_grad,
                        self._rescue_limit())
                    record.update(hold)
                elif not result[0] and settings.rescue_unconverged_scf:
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
                    result, followed, mode = self._follow_negative_mode(result,
                                                                        ops_of(result))
                    record.update(mode)
                    if accept_center:
                        self.last_center_followed = bool(followed)

                # First centre only; later ones follow its solution.
                if (result[0] and accept_center and settings.symmetric_start
                        and self.active_solution_reference is None):
                    result, symmetric = self._symmetric_start(solver, result)
                    record.update(symmetric)

                if result[0] and symmetry is not None:
                    record.update(self._symmetry_record(result))
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
        record.update(self.open_shell_record(solver, result[3], result[4]))
        spread = record.get("open_shell_population_spread")
        if spread is not None and spread > OPEN_SHELL_SPREAD_WARNING:
            print("# warning: %s: the open shell differs between equivalent fragment atoms "
                  "(%s; spread %.3f): the frozen-bath state breaks the defect's symmetry"
                  % (reason, ", ".join(
                      "%s%d %.3f" % (ghost_species(self.labels[a]), a,
                                     record["open_shell_population"][str(a)])
                      for a in record["open_shell_population_worst_group"]), spread),
                  flush=True)

        casci = None
        if settings.state != "ground":
            if not relax_active and self.casci_reference is None:
                # A frozen frame has no orbital energies to order the CAS window by.
                raise RuntimeError(
                    "a semi-analytic CASCI displacement requires an accepted "
                    "centre with a tracked CI reference"
                )
            # casci.energy is already the total: _fake_mf folds the core terms in.
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
                    cas_weight_gap_tol=settings.cas_weight_gap_tol,
                    cas_weight_floor=settings.cas_weight_floor,
                    allow_degenerate_cas=settings.allow_degenerate_cas,
                    cas_selection=settings.cas_selection,
                    cas_energy_gap_tol=settings.cas_energy_gap_tol,
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
                "cas_window_min_overlap": casci.cas_window_min_overlap,
                "cas_orbital_subspace_singular_values": (
                    None if casci.orbital_subspace_singular_values is None
                    else casci.orbital_subspace_singular_values.tolist()
                ),
                **(casci.selection or {}),
            })

            # Guard 1: the CAS window (not the whole active space) keeps its frame.
            if (casci.cas_window_min_overlap is not None
                    and casci.cas_window_min_overlap
                    < settings.min_subspace_overlap):
                self.recorder.evaluation({
                    **record,
                    "event": "energy_failed",
                    "error": "CASCI active-orbital continuity lost",
                    "wall_seconds": round(time.time() - started, 6),
                })
                raise SubspaceContinuityError(
                    "minimum tracked CAS-window orbital overlap %.6f is below "
                    "%.6f (whole active space: %.6f)"
                    % (casci.cas_window_min_overlap,
                       settings.min_subspace_overlap,
                       casci.orbital_min_overlap),
                    continuity_space="casci-active-orbitals",
                )
            # Guard 2: some root still carries the tracked state.
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
                # Differs from the relaxed surface with --state excited.
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
                defect_index=defect_orbital_index(
                    solver, accepted_coeff, np.asarray(result[4]).real
                ),
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
                    ncore=int(casci.ncore),
                    ncas=int(settings.ncas),
                    ncas_elec=int(settings.ncas_elec),
                    two_s=int(self.case.spin if settings.casci_two_s is None
                              else settings.casci_two_s),
                    cas_indices=(
                        self.casci_reference.cas_indices
                        if self.casci_reference is not None
                        else (casci.selection or {}).get("cas_indices")
                    ),
                )
            if self.fixed_n_bath is None:
                self.fixed_n_bath = int(solver.n_bath)
            if self.fixed_fragment_atoms is None:
                self.fixed_fragment_atoms = tuple(sorted(solver.frag_atoms))
            self.cache.clear()
            self.last_accepted_key = key
            self.kept_centre = ({"kmf": pristine_kmf, "solver": solver}
                                if keep_point else None)
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
