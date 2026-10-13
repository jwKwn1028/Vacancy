from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

from pyscf import gto
from pyscf.geomopt import geometric_solver

from hact_relax.cases import (
    BOHR,
    CASES,
    CHAIN_CELLS,
    R_ANG,
    VacancyCase,
    fragment_relax_radius,
)
from system.hbn import (
    A_HBN,
    HBN_CELLS,
    VACUUM_HBN,
    central_boron_index,
    hbn_primitive_atoms,
    hbn_primitive_lattice,
    hbn_supercell,
    hbn_supercell_lattice,
)
from system.shells import minimum_image_distances, nearest_atoms
from system.vacancy import get_vacancy_xyz


def ghost_species(label: str) -> str:
    if not label.upper().startswith("GHOST"):
        return label
    tail = label[len("GHOST"):].lstrip("-:_")
    if not tail:
        raise ValueError("ghost label has no parent element: %r" % label)
    return tail


def pristine_chain_body(ncells: int = CHAIN_CELLS, r_ang: float = R_ANG) -> str:
    if int(ncells) < 1:
        raise ValueError("ncells must be positive")
    rows = []
    for cell_index in range(int(ncells)):
        x0 = 2.0 * float(r_ang) * cell_index
        rows.append("Li %.10f 0.0 0.0" % x0)
        rows.append("H %.10f 0.0 0.0" % (x0 + float(r_ang)))
    return "\n".join(rows) + "\n"


def generated_starting_xyz(
    case_name: str, ncells: int = CHAIN_CELLS, r_ang: float = R_ANG
) -> str:
    case = CASES[case_name]
    ncell = int(ncells)
    vacancy_index = 2 * (ncell // 2) + (1 if case.vacancy == "H" else 0)
    body = get_vacancy_xyz(pristine_chain_body(ncell, r_ang), [vacancy_index])
    body = body.replace("Ghost:", "GHOST-")
    comment = (
        "LiH-1D R=%.3f A, cells=%d, full-chain kmesh=1x1x1, one %s vacancy"
        % (r_ang, ncell, case.vacancy)
    )
    return "%d\n%s\n%s\n" % (2 * ncell, comment, body)


def parse_xyz(
    text: str, source: str = "<generated>"
) -> tuple[list[str], np.ndarray, int]:
    lines = text.splitlines()
    if len(lines) < 2:
        raise ValueError("invalid XYZ: %s" % source)
    natm = int(lines[0].split()[0])
    rows = lines[2: 2 + natm]
    if len(rows) != natm:
        raise ValueError("%s declares %d atoms but has %d"
                         % (source, natm, len(rows)))
    labels: list[str] = []
    coords: list[list[float]] = []
    ghost = []
    for i, row in enumerate(rows):
        fields = row.split()
        labels.append(fields[0])
        coords.append([float(x) for x in fields[1:4]])
        if fields[0].upper().startswith("GHOST"):
            ghost.append(i)
    if len(ghost) != 1:
        raise ValueError("%s must contain exactly one GHOST centre" % source)
    return labels, np.asarray(coords) / BOHR, ghost[0]


def read_xyz(path: Path) -> tuple[list[str], np.ndarray, int]:
    return parse_xyz(path.read_text(), os.fspath(path))


def build_starting_geometry(
    case_name: str, ncells: int = CHAIN_CELLS, r_ang: float = R_ANG
) -> tuple[list[str], np.ndarray, int]:
    return parse_xyz(
        generated_starting_xyz(case_name, ncells, r_ang),
        "generated %s" % case_name,
    )


def generated_hbn_xyz(
    case_name: str,
    ncells: int = HBN_CELLS,
    a_ang: float = A_HBN,
    vacuum_ang: float = VACUUM_HBN,
) -> str:
    case = CASES[case_name]
    symbols, coords, _ = hbn_supercell(ncells, a_ang, vacuum_ang)
    vacancy_index = central_boron_index(ncells)
    if symbols[vacancy_index] != case.vacancy:
        raise ValueError(
            "hBN site %d is %s, not the %s vacancy of case %s"
            % (vacancy_index, symbols[vacancy_index], case.vacancy, case_name)
        )
    rows = []
    for index, (symbol, xyz) in enumerate(zip(symbols, coords)):
        label = "GHOST-" + symbol if index == vacancy_index else symbol
        rows.append("%-8s %18.10f %18.10f %18.10f" % (label, *xyz))
    comment = (
        "hBN-2D a=%.4f A, %dx%d supercell (%d atoms), vacuum %.1f A, one %s vacancy"
        % (a_ang, ncells, ncells, len(symbols), vacuum_ang, case.vacancy)
    )
    return "%d\n%s\n%s\n" % (len(symbols), comment, "\n".join(rows))


def build_hbn_starting_geometry(
    case_name: str,
    ncells: int = HBN_CELLS,
    a_ang: float = A_HBN,
    vacuum_ang: float = VACUUM_HBN,
) -> tuple[list[str], np.ndarray, int]:
    return parse_xyz(
        generated_hbn_xyz(case_name, ncells, a_ang, vacuum_ang),
        "generated %s" % case_name,
    )


def hbn_lattice(
    ncells: int = HBN_CELLS, a_ang: float = A_HBN, vacuum_ang: float = VACUUM_HBN
) -> np.ndarray:
    """Supercell lattice vectors as rows, Bohr."""
    return hbn_supercell_lattice(ncells, a_ang, vacuum_ang) / BOHR


def check_geometry_fits_supercell(
    coords_bohr: np.ndarray, lattice_bohr: np.ndarray, source: str
) -> float:
    """Shortest nearest-image distance between two atoms, Angstrom.

    The 2D analogue of the chain's wrap-bond check: an atom moved onto another
    one's periodic image shows up as a vanishing distance.
    """
    coords = np.asarray(coords_bohr, dtype=float)
    shortest = min(
        float(np.delete(minimum_image_distances(coords, lattice_bohr, i), i).min())
        for i in range(len(coords))
    ) * BOHR
    if shortest < 0.5:
        raise ValueError(
            "%s has two atoms %.4f A apart under the nearest-image rule; the "
            "geometry does not fit its supercell" % (source, shortest)
        )
    return shortest


def fragment_movable_atoms(
    labels: Sequence[str],
    coords_bohr: np.ndarray,
    lattice_bohr: np.ndarray,
    vacancy_index: int,
    n_frag: int,
) -> list[int]:
    """The real atoms of the fragment: its complete shells, minus the ghost."""
    fragment = nearest_atoms(coords_bohr, lattice_bohr, vacancy_index, n_frag)
    return sorted(
        i for i in fragment if not labels[i].upper().startswith("GHOST")
    )


def ideal_supercell(primitive_atoms, primitive_lattice, kmesh):
    """(symbols, coords) of the primitive cell repeated over ``kmesh``.

    Image by image, the order pyscf.pbc.tools.super_cell and k2gamma use, so it
    is the supercell the k-point embedding unfolds to.  Units are those of the
    inputs.
    """
    lattice = np.asarray(primitive_lattice, dtype=float)
    images = np.array([
        [i, j, k]
        for i in range(int(kmesh[0]))
        for j in range(int(kmesh[1]))
        for k in range(int(kmesh[2]))
    ])
    shifts = images @ lattice
    symbols = [str(symbol) for _ in shifts for symbol, _ in primitive_atoms]
    coords = np.array([
        np.asarray(xyz, dtype=float) + shift
        for shift in shifts for _, xyz in primitive_atoms
    ])
    return symbols, coords


def lih_primitive_cell(
    r_ang: float = R_ANG, vacuum_ang: float = 30.0
) -> tuple[list, np.ndarray]:
    """The 2-atom LiH-1D cell as ([symbol, xyz], lattice), Bohr."""
    atoms = [["Li", np.zeros(3)], ["H", np.array([float(r_ang) / BOHR, 0.0, 0.0])]]
    return atoms, full_chain_lattice(1, r_ang, vacuum_ang)


def hbn_primitive_cell(
    a_ang: float = A_HBN, vacuum_ang: float = VACUUM_HBN
) -> tuple[list, np.ndarray]:
    """The 2-atom hBN-2D cell as ([symbol, xyz], lattice), Bohr."""
    atoms = [[symbol, np.asarray(xyz) / BOHR]
             for symbol, xyz in hbn_primitive_atoms(a_ang, vacuum_ang)]
    return atoms, hbn_primitive_lattice(a_ang, vacuum_ang) / BOHR


def snap_to_ideal_lattice(
    labels: Sequence[str],
    coords_bohr: np.ndarray,
    ideal_symbols: Sequence[str],
    ideal_coords_bohr: np.ndarray,
    source: str,
    tol_bohr: float = 1e-6,
) -> np.ndarray:
    """The ideal coordinates, after checking the geometry is that lattice.

    A k-point reference is the perfect crystal, so it can only stand for a
    geometry that is the ideal lattice up to the rounding of an xyz file.
    """
    coords = np.asarray(coords_bohr, dtype=float)
    ideal = np.asarray(ideal_coords_bohr, dtype=float)
    if len(labels) != len(ideal_symbols) or coords.shape != ideal.shape:
        raise ValueError(
            "%s has %d atoms; the k-point supercell has %d"
            % (source, len(labels), len(ideal_symbols))
        )
    wrong = [i for i, (label, symbol) in enumerate(zip(labels, ideal_symbols))
             if ghost_species(label).lower() != symbol.lower()]
    if wrong:
        raise ValueError(
            "%s does not list its atoms in the k-point supercell order: atom %d "
            "is %s, expected %s"
            % (source, wrong[0], labels[wrong[0]], ideal_symbols[wrong[0]])
        )
    deviation = float(np.abs(coords - ideal).max()) if coords.size else 0.0
    if deviation > tol_bohr:
        raise ValueError(
            "%s is displaced from the ideal lattice by up to %.3e Bohr; the "
            "k-point reference describes only the perfect crystal, so use the "
            "Gamma reference for this geometry" % (source, deviation)
        )
    return ideal.copy()


def write_xyz(
    path: Path,
    labels: Sequence[str],
    coords_bohr: np.ndarray,
    comment: str,
    append: bool = False,
) -> None:
    lines = [str(len(labels)), comment]
    for label, xyz in zip(labels, np.asarray(coords_bohr) * BOHR):
        lines.append("%-10s %18.10f %18.10f %18.10f"
                     % (label, xyz[0], xyz[1], xyz[2]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w") as handle:
        handle.write("\n".join(lines) + "\n")


def check_geometry_fits_lattice(
    coords_bohr: np.ndarray, ncells: int, r_ang: float, source: str, axis: int = 0
) -> float:
    x = np.asarray(coords_bohr)[:, int(axis)]
    span = float(x.max() - x.min()) * BOHR
    length = 2.0 * float(r_ang) * int(ncells)
    wrap = length - span
    if wrap <= 0.0:
        raise ValueError(
            "%s spans %.4f A in a %.4f A cell, so the wrap bond is %.4f A: the "
            "chain overruns its own periodic image"
            % (source, span, length, wrap)
        )
    return wrap


def full_chain_lattice(
    ncells: int = CHAIN_CELLS, r_ang: float = R_ANG, vacuum_ang: float = 30.0
) -> np.ndarray:
    if int(ncells) < 1:
        raise ValueError("ncells must be positive")
    return np.diag([
        2.0 * float(r_ang) * int(ncells) / BOHR,
        float(vacuum_ang) / BOHR,
        float(vacuum_ang) / BOHR,
    ])


def restored_pristine_atoms(
    labels: Sequence[str], coords_bohr: np.ndarray
) -> list[list[object]]:
    return [
        [ghost_species(label), np.asarray(xyz, dtype=float).copy()]
        for label, xyz in zip(labels, np.asarray(coords_bohr))
    ]


def movable_atoms(
    labels: Sequence[str],
    coords_bohr: np.ndarray,
    vacancy_index: int,
    n_frag: int,
    r_ang: float = R_ANG,
) -> list[int]:
    candidates = [
        i for i, label in enumerate(labels)
        if not label.upper().startswith("GHOST")
    ]
    radius_ang = fragment_relax_radius(n_frag, r_ang)
    distance = np.linalg.norm(
        np.asarray(coords_bohr) - np.asarray(coords_bohr)[vacancy_index], axis=1
    )
    return [i for i in candidates if distance[i] * BOHR <= radius_ang + 1e-12]


def seed_displace_from_vacancy(
    coords_bohr: np.ndarray, vacancy_index: int, displacement_ang: float
) -> np.ndarray:
    coords = np.asarray(coords_bohr, dtype=float).copy()
    if displacement_ang == 0.0:
        return coords
    displacement_bohr = displacement_ang / BOHR
    vacancy_x = coords[vacancy_index, 0]
    coords[coords[:, 0] < vacancy_x, 0] -= displacement_bohr
    coords[coords[:, 0] > vacancy_x, 0] += displacement_bohr
    coords[vacancy_index] = np.asarray(coords_bohr)[vacancy_index]
    return coords


def patch_geometric_for_ghost_atoms() -> None:
    base = geometric_solver.PySCFEngine
    if getattr(base, "_hact_ghost_safe", False):
        return

    class GhostSafePySCFEngine(base):
        _hact_ghost_safe = True

        def __init__(self, scanner):
            super().__init__(scanner)
            self.M.elem = [
                symbol[len("GHOST-"):] if symbol.startswith("GHOST-") else symbol
                for symbol in self.M.elem
            ]

    geometric_solver.PySCFEngine = GhostSafePySCFEngine


def build_driver_molecule(
    labels: Sequence[str],
    coords_bohr: np.ndarray,
    case: VacancyCase,
    basis: str,
    memory_mb: int,
    verbose: int,
):
    species = {label for label in labels if not label.upper().startswith("GHOST")}
    basis_by_label = {symbol: basis for symbol in sorted(species)}
    basis_by_label["GHOST-" + case.vacancy] = gto.basis.load(basis, case.vacancy)
    return gto.M(
        atom=[[label, xyz] for label, xyz in zip(labels, coords_bohr)],
        unit="Bohr",
        basis=basis_by_label,
        charge=case.charge,
        spin=case.spin,
        verbose=verbose,
        max_memory=memory_mb,
    )
