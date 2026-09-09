from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyscf.pbc.gto import cell as pbc_cell


@dataclass
class PristineReference:
    cell: object
    density: np.ndarray
    occupied_coeff: np.ndarray
    kmf: object = None


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
        kmf=kmf,
        cell=kmf.cell,
        density=np.asarray(kmf.make_rdm1()).copy(),
        occupied_coeff=coeff[:, occ_gamma > 0].copy(),
    )
