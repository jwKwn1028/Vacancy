from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

from pyscf import gto
from pyscf.geomopt import geometric_solver

from hact_relax.cases import BOHR, CASES, CHAIN_CELLS, R_ANG, VacancyCase
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


def write_xyz(
    path: Path, labels: Sequence[str], coords_bohr: np.ndarray, comment: str
) -> None:
    lines = [str(len(labels)), comment]
    for label, xyz in zip(labels, np.asarray(coords_bohr) * BOHR):
        lines.append("%-10s %18.10f %18.10f %18.10f"
                     % (label, xyz[0], xyz[1], xyz[2]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def scaled_geometry(coords_bohr, epsilon, axis=0):
    out = np.array(coords_bohr, dtype=float, copy=True)
    out[:, int(axis)] *= 1.0 + float(epsilon)
    return out


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
    radius_ang: float | None,
) -> list[int]:
    candidates = [
        i for i, label in enumerate(labels)
        if not label.upper().startswith("GHOST")
    ]
    if radius_ang is None:
        return candidates
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
    return gto.M(
        atom=[[label, xyz] for label, xyz in zip(labels, coords_bohr)],
        unit="Bohr",
        basis={
            "Li": basis,
            "H": basis,
            "GHOST-" + case.vacancy: gto.basis.load(basis, case.vacancy),
        },
        charge=case.charge,
        spin=case.spin,
        verbose=verbose,
        max_memory=memory_mb,
    )
