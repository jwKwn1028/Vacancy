from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from hact_relax.errors import BranchFlip
from hact_relax.surface import ActiveHamiltonianSurface


class FiniteDifferenceScanner:

    def __init__(
        self,
        surface: ActiveHamiltonianSurface,
        movable: Sequence[int],
        step_bohr: float,
        max_gradient: float = 1.0,
        axes: Sequence[int] = (0, 1, 2),
        mode: str = "central",
    ):
        self.surface = surface
        self.movable = tuple(int(i) for i in movable)
        self.step_bohr = float(step_bohr)
        self.max_gradient = float(max_gradient)
        self.axes = tuple(sorted({int(a) for a in axes}))
        if not self.axes or not all(0 <= a <= 2 for a in self.axes):
            raise ValueError("axes must be a non-empty subset of (0, 1, 2)")
        if mode not in ("central", "forward"):
            raise ValueError("mode must be 'central' or 'forward'")
        self.mode = mode
        self.calls = 0
        self.last_energy: float | None = None
        self.last_gradient: np.ndarray | None = None
        self.last_coords: np.ndarray | None = None

    @property
    def point_count_per_gradient(self) -> int:
        per_axis = 2 if self.mode == "central" else 1
        return 1 + per_axis * len(self.axes) * len(self.movable)

    def _check_branch(self, displaced: float, center: float, label: str) -> None:
        implied = abs(float(displaced) - float(center)) / self.step_bohr
        if implied > self.max_gradient:
            raise BranchFlip(
                "%s implies |g| = %.6e Ha/Bohr, above the %.3e limit: the "
                "displaced point converged onto a different SCF solution "
                "(E_displaced = %.12f, E_center = %.12f, step = %.3e Bohr)"
                % (label, implied, self.max_gradient, displaced, center,
                   self.step_bohr),
                implied_gradient_hartree_per_bohr=implied,
                max_gradient_hartree_per_bohr=self.max_gradient,
            )

    def __call__(self, mol):
        self.calls += 1
        coords = np.asarray(mol.atom_coords(), dtype=float)
        energy = self.surface.energy(
            coords, "gradient-%d-center" % self.calls, accept_center=True
        )
        gradient = np.zeros_like(coords)
        for atom_index in self.movable:
            for axis in self.axes:
                plus = coords.copy()
                plus[atom_index, axis] += self.step_bohr
                label = "gradient-%d-a%d%s" % (
                    self.calls, atom_index, "xyz"[axis]
                )
                ep = self.surface.energy(
                    plus, label + "+", displaced_atom=atom_index
                )
                self._check_branch(ep, energy, label + "+")
                if self.mode == "central":
                    minus = coords.copy()
                    minus[atom_index, axis] -= self.step_bohr
                    em = self.surface.energy(
                        minus, label + "-", displaced_atom=atom_index
                    )
                    self._check_branch(em, energy, label + "-")
                    gradient[atom_index, axis] = (
                        (ep - em) / (2.0 * self.step_bohr)
                    )
                else:
                    gradient[atom_index, axis] = (
                        (ep - energy) / self.step_bohr
                    )

        self.last_energy = energy
        self.last_gradient = gradient
        self.last_coords = coords.copy()
        return energy, gradient


def write_constraints(
    path: Path,
    natm: int,
    movable: Sequence[int],
    axes: Sequence[int] = (0, 1, 2),
) -> Path | None:
    frozen = sorted(set(range(natm)) - set(movable))
    held = "".join("xyz"[a] for a in range(3) if a not in set(int(x) for x in axes))
    lines = ["$freeze"]
    if frozen:
        lines.append("xyz %s" % ",".join(str(i + 1) for i in frozen))
    if held and movable:
        lines.append("%s %s" % (held, ",".join(str(i + 1) for i in sorted(movable))))
    if len(lines) == 1:
        return None
    path.write_text("\n".join(lines) + "\n")
    return path


def save_final_gradient(
    output_dir: Path, labels: Sequence[str], gradient: np.ndarray
) -> tuple[Path, Path]:
    gradient = np.asarray(gradient, dtype=float).reshape(len(labels), 3)
    npy_path = output_dir / "final_gradient.npy"
    txt_path = output_dir / "final_gradient.txt"
    np.save(npy_path, gradient)
    lines = ["# atom label gx gy gz  [Hartree/Bohr]"]
    for index, (label, row) in enumerate(zip(labels, gradient), start=1):
        lines.append("%5d %-10s %20.12e %20.12e %20.12e"
                     % (index, label, row[0], row[1], row[2]))
    txt_path.write_text("\n".join(lines) + "\n")
    return npy_path, txt_path
