"""Geometry optimization on the analytic gradient: one centre per geomeTRIC step.

Guards before a step reaches geomeTRIC: a clean bath cut, a stationary minimum of the active
problem, and the trapezoid rule between consecutive points,

    |E1 - E0 - (g0 + g1).dx/2| <= branch_tol + |(g1 - g0).dx|/4.

Each call writes restart.npz, which Restart continues from.  Each Z-vector starts from the
previous point's response.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

from hact_relax import analytic_gradient as ag
from hact_relax.errors import BranchFlip, SubspaceContinuityError

STATIONARITY_LIMIT = 1e-6
RESTART_FILE = "restart.npz"


class AnalyticGradientScanner:
    """geomeTRIC's energy and gradient; the fit must treat ``movable`` as moved."""

    branch_tol = 1e-5
    zvector_tol = 1e-10
    zvector_warm_start = True
    bath_kept_min = 1e-5
    bath_cut_ratio_max = 1e-3
    restart = None

    def __init__(self, surface, movable, axes=(0, 1, 2), trajectory=None, **options):
        for name, value in options.items():
            if name not in ("branch_tol", "zvector_tol", "zvector_warm_start",
                            "bath_kept_min", "bath_cut_ratio_max", "restart"):
                raise TypeError("unknown option %r" % name)
            setattr(self, name, value)
        self.surface = surface
        self.trajectory = trajectory
        self.movable = tuple(sorted(int(i) for i in movable))
        self.axes = tuple(sorted({int(a) for a in axes}))
        self.calls = 0
        self.last_energy = self.last_gradient = self.last_coords = None
        self.hessian = None
        self.zvector_density = None
        self.run_dir = (None if trajectory is None
                        else Path(trajectory.xyz_path).parent.resolve())
        self.log_path = (None if self.run_dir is None
                         else self.run_dir / "analytic_gradient.jsonl")
        if self.restart is not None:
            self.restart.apply(self)

    def __call__(self, mol):
        self.calls += 1
        coords = np.asarray(mol.atom_coords(), dtype=float)
        row = {"call": self.calls}
        if (self.calls == 1 and self.restart is not None
                and np.allclose(coords, self.last_coords, atol=1e-8, rtol=0.0)):
            row.update(restored_from=os.fspath(self.restart.path),
                       energy_hartree=self.last_energy, guard=None,
                       max_abs_gradient=float(abs(self.last_gradient).max()))
            self._log(row)
            self.last_coords = coords.copy()
            if self.trajectory is not None:
                self.trajectory.record(self.calls, coords, self.last_energy,
                                       self.last_gradient)
            self._save_restart()
            return self.last_energy, self.last_gradient.copy()
        t = time.perf_counter()
        point = ag.capture_point(self.surface, coords,
                                 "analytic-gradient-%d-center" % self.calls)
        row["energy_seconds"] = time.perf_counter() - t
        row["production_energy_hartree"] = point.energy
        energy, gradient = point.energy, np.zeros_like(coords)
        bath = self._bath_record(point.solver)
        row.update(bath)
        open_shell = getattr(self.surface, "open_shell_record", None)
        if open_shell is not None:
            row.update(open_shell(point.solver, point.active_coeff, point.active_occ))
        problem = self._bath_problem(bath)
        if problem is not None:
            row.update(energy_hartree=energy, guard=problem)
            self._log(row)
            raise SubspaceContinuityError(problem, continuity_space="bath", call=self.calls,
                                          **bath)
        if self.movable:
            t = time.perf_counter()
            grad = ag.analytic_gradient(
                point, self.movable, axes=self.axes, zvector_tol=self.zvector_tol,
                zvector_guess=self.zvector_density if self.zvector_warm_start else None)
            row["gradient_seconds"] = time.perf_counter() - t
            self.zvector_density = grad.zvector_density
            gradient[list(self.movable)] = grad.total
            energy = point.energy + grad.stationarity["energy_change"]
            row.update(stationarity=grad.stationarity, zvector=grad.zvector,
                       timings=grad.timings)
            problem = self._stationarity_problem(grad.stationarity)
            if problem is None and self.last_coords is not None:
                problem = self._branch_problem(coords, energy, gradient, row)
        row.update(energy_hartree=energy, max_abs_gradient=float(abs(gradient).max()),
                   guard=problem)
        self._log(row)
        if problem is not None:
            raise BranchFlip(problem, call=self.calls, energy_hartree=energy,
                             previous_energy_hartree=self.last_energy)
        self._update_hessian(coords, gradient)
        self.last_energy, self.last_gradient = energy, gradient
        self.last_coords = coords.copy()
        if self.trajectory is not None:
            self.trajectory.record(self.calls, coords, energy, gradient)
        self._save_restart()
        return energy, gradient

    def _update_hessian(self, coords, gradient):
        """BFGS from geomeTRIC's Cartesian guess, 0.5; skips pairs of negative curvature."""
        if self.hessian is None:
            self.hessian = 0.5 * np.eye(coords.size)
        if self.last_coords is None:
            return
        s = (coords - self.last_coords).ravel()
        y = (gradient - self.last_gradient).ravel()
        sy = float(s @ y)
        if sy <= 1e-8 * np.linalg.norm(s) * np.linalg.norm(y):
            return
        hs = self.hessian @ s
        self.hessian += np.outer(y, y) / sy - np.outer(hs, hs) / float(s @ hs)

    def _save_restart(self):
        surface = self.surface
        active = getattr(surface, "active_solution_reference", None)
        pristine = getattr(surface, "pristine_reference", None)
        if self.run_dir is None or active is None or pristine is None:
            return
        data = {
            "case": surface.settings.case_name, "fragment": int(surface.settings.fragment),
            "call": self.calls, "labels": np.asarray(surface.labels),
            "coords_bohr": self.last_coords, "energy_hartree": self.last_energy,
            "gradient": self.last_gradient, "hessian": self.hessian,
            "movable": np.asarray(self.movable), "axes": np.asarray(self.axes),
            "fixed_n_bath": int(surface.fixed_n_bath),
            "fixed_fragment_atoms": np.asarray(surface.fixed_fragment_atoms, dtype=int),
            "active_cell": active.cell.dumps(),
            "active_coeff": np.asarray(active.active_coeff),
            "active_occ": np.asarray(active.active_occ),
            "defect_index": -1 if active.defect_index is None else int(active.defect_index),
            "pristine_cell": pristine.cell.dumps(),
            "pristine_density": np.asarray(pristine.density),
            "pristine_occupied_coeff": np.asarray(pristine.occupied_coeff),
        }
        if self.zvector_density is not None:
            data["zvector_density"] = self.zvector_density
        path = self.run_dir / RESTART_FILE
        partial = path.with_name(path.name + ".partial")
        with open(partial, "wb") as f:
            np.savez(f, **data)
        os.replace(partial, path)

    @staticmethod
    def _bath_record(solver):
        values = getattr(solver, "bath_singular_values", None)
        s = np.asarray([] if values is None else values, dtype=float)
        n = int(solver.n_bath)
        kept = float(s[:n].min()) if n else None
        dropped = float(s[n:].max()) if s.size > n else None
        return {"n_bath": n, "bath_singular_value_kept_min": kept,
                "bath_singular_value_dropped_max": dropped,
                "bath_cut_ratio": None if not kept or dropped is None else dropped / kept}

    def _bath_problem(self, bath):
        kept, ratio = bath["bath_singular_value_kept_min"], bath["bath_cut_ratio"]
        if kept is not None and kept < self.bath_kept_min:
            return ("the weakest kept bath singular value %.2e is below %.0e: the host SCF's "
                    "convergence noise sets that orbital's direction, and the gradient's "
                    "bath term grows as its inverse; start again with a --bath-tol above it"
                    % (kept, self.bath_kept_min))
        if ratio is not None and ratio > self.bath_cut_ratio_max:
            return ("the bath cut is not clean: the largest dropped singular value is %.1e "
                    "of the weakest kept (limit %.0e), so a step can swap orbitals across "
                    "it at the locked rank, a jump the gradient does not see; start again "
                    "with a --bath-tol in a clean gap" % (ratio, self.bath_cut_ratio_max))
        return None

    @staticmethod
    def _stationarity_problem(st):
        if st["gradient_after"] > STATIONARITY_LIMIT:
            return ("the polished centre's active orbital gradient is %.1e (limit %.0e): the "
                    "analytic gradient holds the active solution fixed, which needs it "
                    "stationary" % (st["gradient_after"], STATIONARITY_LIMIT))
        if st["lowest_hessian"] <= 0.0:
            return ("the centre's active solution is a saddle (lowest orbital Hessian %.3e); "
                    "run with --follow-negative-mode" % st["lowest_hessian"])
        return None

    def _branch_problem(self, coords, energy, gradient, row):
        dx = coords - self.last_coords
        change = float(energy - self.last_energy)
        mismatch = change - 0.5 * float(np.sum((gradient + self.last_gradient) * dx))
        curvature = float(np.sum((gradient - self.last_gradient) * dx))
        limit = self.branch_tol + 0.25 * abs(curvature)
        row.update(step_bohr=float(np.linalg.norm(dx)), trapezoid_mismatch_hartree=mismatch,
                   step_curvature_hartree=curvature, trapezoid_limit_hartree=limit)
        if abs(mismatch) <= limit:
            return None
        return ("E - E_prev = %.3e Ha but (g + g_prev).dx/2 = %.3e Ha: mismatch %.1e above "
                "%.1e (%.0e + |(g - g_prev).dx|/4) over a %.3e Bohr step; the point is on "
                "another SCF solution"
                % (change, change - mismatch, mismatch, limit, self.branch_tol,
                   float(np.linalg.norm(dx))))

    def _log(self, row):
        if self.log_path is None:
            return
        with open(self.log_path, "a") as f:
            f.write(json.dumps(row, default=float, sort_keys=True) + "\n")


class Restart:
    """A run's restart.npz."""

    def __init__(self, run_dir):
        self.path = (Path(run_dir) / RESTART_FILE).resolve()
        if not self.path.exists():
            raise FileNotFoundError("%s does not exist: the run completed no analytic "
                                    "call, or predates the restart file" % self.path)
        with np.load(self.path) as f:
            self.data = {k: f[k] for k in f.files}
        self.case = str(self.data["case"])
        self.fragment = int(self.data["fragment"])
        self.call = int(self.data["call"])
        self.hessian_file = None

    def write_xyz(self):
        """The saved geometry as an xyz file beside restart.npz."""
        from hact_relax.geometry import write_xyz

        d = self.data
        path = self.path.with_name("restart-call%d.xyz" % self.call)
        write_xyz(path, [str(x) for x in d["labels"]], d["coords_bohr"],
                  "%s call %d E=%.12f Ha" % (self.path, self.call, float(d["energy_hartree"])))
        return path

    def apply(self, scanner):
        """Restore the surface state, the last point and the Hessian (saved for geomeTRIC)."""
        from pyscf.pbc import gto as pbc_gto

        from hact_relax.tracking import ActiveSolutionReference, PristineReference

        d, surface = self.data, scanner.surface
        if [str(x) for x in d["labels"]] != list(surface.labels):
            raise RuntimeError("%s: the atoms differ from this run's" % self.path)
        if (tuple(int(a) for a in d["movable"]) != scanner.movable
                or tuple(int(a) for a in d["axes"]) != scanner.axes):
            raise RuntimeError("%s: movable atoms %s on axes %s, this run %s on %s"
                               % (self.path, d["movable"].tolist(), d["axes"].tolist(),
                                  list(scanner.movable), list(scanner.axes)))
        surface.fixed_n_bath = int(d["fixed_n_bath"])
        surface.fixed_fragment_atoms = tuple(int(a) for a in d["fixed_fragment_atoms"])
        index = int(d["defect_index"])
        surface.active_solution_reference = ActiveSolutionReference(
            cell=pbc_gto.loads(str(d["active_cell"])),
            active_coeff=d["active_coeff"].copy(), active_occ=d["active_occ"].copy(),
            defect_index=None if index < 0 else index)
        surface.pristine_reference = PristineReference(
            cell=pbc_gto.loads(str(d["pristine_cell"])), density=d["pristine_density"].copy(),
            occupied_coeff=d["pristine_occupied_coeff"].copy())
        scanner.last_coords = d["coords_bohr"].copy()
        scanner.last_energy = float(d["energy_hartree"])
        scanner.last_gradient = d["gradient"].copy()
        scanner.hessian = d["hessian"].copy()
        if "zvector_density" in d:
            scanner.zvector_density = d["zvector_density"].copy()
        if scanner.run_dir is not None:
            self.hessian_file = scanner.run_dir / "restart_hessian.txt"
            np.savetxt(self.hessian_file, scanner.hessian)
