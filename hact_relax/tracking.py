from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from pyscf.pbc.gto import cell as pbc_cell

from frozen_bath_scf.casci import _fake_mf, _run_casci


@dataclass
class PristineReference:
    cell: object
    density: np.ndarray
    occupied_coeff: np.ndarray


@dataclass
class ActiveSolutionReference:
    cell: object
    active_coeff: np.ndarray
    active_occ: np.ndarray
    defect_index: int | None = None


@dataclass
class SubspaceMatch:
    overlap: np.ndarray
    left: np.ndarray
    singular_values: np.ndarray
    right_h: np.ndarray

    @property
    def fidelity(self) -> float:
        return (float(self.singular_values.min())
                if self.singular_values.size else 1.0)


@dataclass
class ActiveTransportResult:
    rotation: np.ndarray
    occupations: np.ndarray
    match: SubspaceMatch

    @property
    def fidelity(self) -> float:
        return self.match.fidelity


def _cell_lattice(cell) -> np.ndarray | None:
    lattice = getattr(cell, "a", None)
    if lattice is None:
        method = getattr(cell, "lattice_vectors", None)
        if method is None:
            return None
        lattice = method()
    array = np.asarray(lattice, dtype=float)
    return array if array.shape == (3, 3) else None


def _ao_layout(cell) -> tuple[str, ...] | None:
    labels = getattr(cell, "ao_labels", None)
    if labels is None:
        return None
    return tuple(str(label) for label in labels())


def _fractional_coordinates(cell, lattice: np.ndarray) -> np.ndarray | None:
    atom_coords = getattr(cell, "atom_coords", None)
    if atom_coords is None:
        return None
    coords = np.asarray(atom_coords(), dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        return None
    return np.linalg.solve(lattice.T, coords.T).T


def match_orbital_subspaces(
    reference_cell,
    reference_coeff: np.ndarray,
    target_cell,
    target_coeff: np.ndarray,
    *,
    target_overlap: np.ndarray | None = None,
    fractional_tolerance: float = 1e-7,
) -> SubspaceMatch:
    previous = np.asarray(reference_coeff)
    current = np.asarray(target_coeff)
    if previous.ndim != 2 or current.ndim != 2:
        raise ValueError("orbital coefficient arrays must both be matrices")
    if previous.shape[0] != current.shape[0]:
        raise RuntimeError(
            "AO dimension changed from %d to %d between continuation points"
            % (previous.shape[0], current.shape[0])
        )

    lattice0 = _cell_lattice(reference_cell)
    lattice1 = _cell_lattice(target_cell)
    lattice_changed = bool(
        lattice0 is not None
        and lattice1 is not None
        and not np.allclose(lattice0, lattice1, atol=1e-10, rtol=1e-10)
    )

    if not lattice_changed:
        cross = np.asarray(pbc_cell.intor_cross(
            "int1e_ovlp", reference_cell, target_cell, kpt=np.zeros(3)
        ))
        overlap = previous.conj().T @ cross @ current
    else:
        layout0 = _ao_layout(reference_cell)
        layout1 = _ao_layout(target_cell)
        if layout0 is None or layout1 is None or layout0 != layout1:
            raise RuntimeError(
                "a changed lattice requires identical atom/AO ordering for "
                "co-moving orbital transport"
            )

        fractional0 = _fractional_coordinates(reference_cell, lattice0)
        fractional1 = _fractional_coordinates(target_cell, lattice1)
        if (fractional0 is None or fractional1 is None
                or fractional0.shape != fractional1.shape):
            raise RuntimeError(
                "a changed lattice requires matching atomic fractional coordinates"
            )
        delta = fractional1 - fractional0
        delta -= np.rint(delta)
        drift = float(np.abs(delta).max()) if delta.size else 0.0
        if drift > float(fractional_tolerance):
            raise RuntimeError(
                "cell continuation is not at fixed fractional coordinates: "
                "maximum drift %.3e exceeds %.3e" % (drift, fractional_tolerance)
            )

        if target_overlap is None:
            target_overlap = target_cell.pbc_intor("int1e_ovlp", hermi=1)
        metric1 = np.asarray(target_overlap)
        if metric1.shape != (current.shape[0], current.shape[0]):
            raise RuntimeError(
                "target AO overlap has shape %s, expected (%d, %d)"
                % (metric1.shape, current.shape[0], current.shape[0])
            )
        reference_metric = previous.conj().T @ metric1 @ previous
        reference_metric = 0.5 * (reference_metric + reference_metric.conj().T)
        eigenvalues, vectors = np.linalg.eigh(reference_metric)
        if eigenvalues.size and eigenvalues.min() < 1e-10:
            raise RuntimeError(
                "co-moving reference space became linearly dependent in the "
                "target-cell metric: minimum eigenvalue %.3e" % eigenvalues.min()
            )
        inverse_sqrt = (vectors / np.sqrt(eigenvalues)) @ vectors.conj().T
        overlap = (previous @ inverse_sqrt).conj().T @ metric1 @ current

    left, singular, right_h = np.linalg.svd(overlap, full_matrices=False)
    return SubspaceMatch(
        overlap=overlap,
        left=left,
        singular_values=np.asarray(singular, dtype=float),
        right_h=right_h,
    )


def transport_active_solution(
    reference: ActiveSolutionReference, solver
) -> ActiveTransportResult:
    active_new = np.asarray(solver.mo_coeff[:, solver.active_orb]).real
    previous = np.asarray(reference.active_coeff).real
    if previous.shape[1] != active_new.shape[1]:
        raise RuntimeError(
            "active-space dimension changed from %d to %d between geometries; "
            "the converged solution cannot be continued"
            % (previous.shape[1], active_new.shape[1])
        )
    match = match_orbital_subspaces(
        reference.cell, previous, solver.scell, active_new,
        target_overlap=getattr(solver, "S", None),
    )
    return ActiveTransportResult(
        rotation=(match.left @ match.right_h).conj().T.real,
        occupations=np.asarray(reference.active_occ, dtype=float).copy(),
        match=match,
    )


def defect_orbital_index(solver, active_coeff):
    c_def = getattr(solver, "_c_def", None)
    if c_def is None:
        return None
    active = np.asarray(active_coeff).real
    overlap = np.abs(np.asarray(c_def).real @ (solver.S @ active))
    return int(np.argmax(overlap))


def transport_defect_orbital(reference, solver, rotation):
    if reference.defect_index is None:
        return None
    active_new = np.asarray(solver.mo_coeff[:, solver.active_orb]).real
    return np.asarray(
        (active_new @ rotation)[:, reference.defect_index]
    ).real.copy()


def _gamma_matrix(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    return array[0] if array.ndim == 3 else array


def align_pristine_occupied_subspace(kmf, reference: PristineReference | None):
    if reference is None:
        return None
    coeff_all = np.asarray(kmf.mo_coeff)
    occ = np.asarray(kmf.mo_occ)
    coeff = _gamma_matrix(coeff_all)
    occ_gamma = occ[0] if occ.ndim == 2 else occ
    occupied = np.where(occ_gamma > 0)[0]
    current = coeff[:, occupied]
    if current.shape != reference.occupied_coeff.shape:
        raise RuntimeError(
            "occupied KRHF dimension changed from %s to %s"
            % (reference.occupied_coeff.shape, current.shape)
        )
    match = match_orbital_subspaces(
        reference.cell, reference.occupied_coeff, kmf.cell, current
    )
    aligned = current @ (match.right_h.conj().T @ match.left.conj().T)
    updated = coeff_all.copy()
    if updated.ndim == 3:
        updated[0][:, occupied] = aligned
    else:
        updated[:, occupied] = aligned
    kmf.mo_coeff = updated
    return match


def make_pristine_reference(kmf) -> PristineReference:
    coeff = _gamma_matrix(np.asarray(kmf.mo_coeff))
    occ = np.asarray(kmf.mo_occ)
    occ_gamma = occ[0] if occ.ndim == 2 else occ
    return PristineReference(
        cell=kmf.cell,
        density=np.asarray(kmf.make_rdm1()).copy(),
        occupied_coeff=coeff[:, occ_gamma > 0].copy(),
    )


# ---------------------------------------------------------------------------
# CASCI on H_act^V, with maximum-overlap orbital and root tracking.
#
# The relaxation needs the SAME excited state at every finite-difference point,
# which is two separate tracking problems:
#
#   1. the active orbitals must be put in a common frame across geometries,
#      otherwise CI vectors from two geometries are not comparable at all;
#   2. within that frame, the root carrying the tracked state must be identified,
#      since CASCI returns roots in energy order and roots cross.
#
# (1) is a Hungarian assignment on the cross-geometry AO overlap plus a sign fix;
# (2) is a modulus CI overlap against the last accepted centre.
# ---------------------------------------------------------------------------


@dataclass
class CASCIReference:
    """The tracked state at the last accepted centre."""

    cell: object
    ordered_active_coeff: np.ndarray
    ci_vector: np.ndarray


@dataclass
class CASCIResult:
    energy: float
    s_squared: float
    ncore: int
    orbital_order: np.ndarray
    root_energies: np.ndarray
    root_s_squared: np.ndarray
    ci_vectors: list[np.ndarray]
    nalpha: int
    nbeta: int
    requested_root: int
    selected_root: int
    root_overlaps: np.ndarray
    ordered_active_coeff: np.ndarray | None
    orbital_min_overlap: float | None
    orbital_subspace_singular_values: np.ndarray | None = None


def _normalized_ci_overlaps(
    reference: np.ndarray, vectors: Sequence[np.ndarray]
) -> np.ndarray:
    ref = np.asarray(reference)
    values = []
    for vector in vectors:
        vec = np.asarray(vector)
        if vec.shape != ref.shape:
            # CI vector shape is (n_alpha_strings, n_beta_strings) and therefore
            # spin-sector dependent.  Ravelling mismatched shapes would compare
            # unrelated determinants and silently return a plausible number.
            raise ValueError(
                "CI vector shape %s does not match the tracked reference %s; "
                "the CAS or the spin sector changed between geometries"
                % (vec.shape, ref.shape)
            )
        flat_ref = ref.ravel()
        flat_vec = vec.ravel()
        denom = np.linalg.norm(flat_ref) * np.linalg.norm(flat_vec)
        values.append(0.0 if denom == 0 else abs(np.vdot(flat_ref, flat_vec)) / denom)
    return np.asarray(values, dtype=float)


def casci_energy(
    solver,
    result,
    *,
    ncas: int,
    ncas_elec: int,
    two_s: int,
    root: int,
    nroots: int,
    reference: CASCIReference | None = None,
) -> CASCIResult:
    """CASCI on ``H_act^V`` with maximum-overlap orbital/root tracking."""
    h1e = np.asarray(result[5]).real
    eri = np.asarray(result[6]).real
    orbital_energy = np.asarray(result[2]).real
    active_coeff = None if result[3] is None else np.asarray(result[3]).real

    orbital_min_overlap = None
    orbital_subspace_singular_values = None
    if reference is None:
        # Without a reference the CAS window is placed by Fock energy.  The
        # relax_active=False path returns zeros for result[2], so ordering would
        # silently degenerate into "by index" -- refuse instead.  A semi-analytic
        # displaced point always has an accepted centre, so this cannot fire
        # there; it fires only if a frozen evaluation is ever reached first.
        if orbital_energy.size and np.ptp(orbital_energy) == 0.0:
            raise RuntimeError(
                "CASCI cannot order the active space: the orbital energies are "
                "all equal, so no frontier window can be identified.  This "
                "happens on a frozen (relax_active=False) evaluation reached "
                "without an accepted centre."
            )
        order = np.argsort(orbital_energy)
        signs = np.ones(order.size)
    else:
        if (active_coeff is None
                or active_coeff.shape != reference.ordered_active_coeff.shape):
            raise RuntimeError(
                "CASCI active-orbital dimension changed during root tracking"
            )
        match = match_orbital_subspaces(
            reference.cell,
            reference.ordered_active_coeff,
            solver.scell,
            active_coeff,
            target_overlap=getattr(solver, "S", None),
        )
        overlap = np.asarray(match.overlap).real
        rows, columns = linear_sum_assignment(-np.abs(overlap))
        order = columns[np.argsort(rows)]
        matched = overlap[np.arange(order.size), order]
        signs = np.where(matched < 0.0, -1.0, 1.0)
        orbital_min_overlap = float(np.min(np.abs(matched))) if matched.size else 1.0
        orbital_subspace_singular_values = match.singular_values.copy()

    h1e = h1e[np.ix_(order, order)]
    eri = eri[np.ix_(order, order, order, order)]
    h1e = h1e * (signs[:, None] * signs[None, :])
    eri = eri * (
        signs[:, None, None, None]
        * signs[None, :, None, None]
        * signs[None, None, :, None]
        * signs[None, None, None, :]
    )
    ordered_active_coeff = (
        None if active_coeff is None else active_coeff[:, order] * signs[None, :]
    )

    active_electrons = (
        int(sum(solver.nelecas)) if np.ndim(solver.nelecas) else int(solver.nelecas)
    )
    if (active_electrons - ncas_elec) % 2:
        raise ValueError(
            "CAS electron count leaves a non-integer doubly occupied core: "
            "active=%d, CAS=%d" % (active_electrons, ncas_elec)
        )
    ncore = (active_electrons - ncas_elec) // 2
    nact = h1e.shape[0]
    if ncore < 0 or ncore + ncas > nact:
        raise ValueError(
            "CAS(%de,%do) does not fit H_act^V: active_electrons=%d, nact=%d, "
            "ncore=%d" % (ncas_elec, ncas, active_electrons, nact, ncore)
        )

    ecore = float(solver.E1e_core + solver.E2e_core + solver.Enuc)
    fake_mf = _fake_mf(h1e, eri, ecore, active_electrons)
    max_memory = getattr(solver, "max_memory", None)
    if max_memory is not None:
        fake_mf.max_memory = int(max_memory)
        fake_mf.mol.max_memory = int(max_memory)
    states = _run_casci(
        fake_mf,
        ncas,
        ncas_elec,
        two_s=two_s,
        nroots=max(nroots, root + 1),
        ncore=ncore,
    )
    states = sorted(states, key=lambda state: state[0])
    if root >= len(states):
        raise ValueError(
            "CASCI returned %d roots, so zero-based root %d does not exist; "
            "enlarge the CAS or select a lower root" % (len(states), root)
        )

    ci_vectors = [np.asarray(state[2]) for state in states]
    if reference is None:
        root_overlaps = np.full(len(states), np.nan)
        selected_root = int(root)
    else:
        root_overlaps = _normalized_ci_overlaps(reference.ci_vector, ci_vectors)
        selected_root = int(np.argmax(root_overlaps))

    return CASCIResult(
        energy=float(states[selected_root][0]),
        s_squared=float(states[selected_root][1]),
        ncore=ncore,
        orbital_order=np.asarray(order, dtype=int),
        root_energies=np.asarray([state[0] for state in states], dtype=float),
        root_s_squared=np.asarray([state[1] for state in states], dtype=float),
        ci_vectors=ci_vectors,
        nalpha=int(states[0][3]),
        nbeta=int(states[0][4]),
        requested_root=int(root),
        selected_root=selected_root,
        root_overlaps=root_overlaps,
        ordered_active_coeff=ordered_active_coeff,
        orbital_min_overlap=orbital_min_overlap,
        orbital_subspace_singular_values=orbital_subspace_singular_values,
    )
