#!/usr/bin/env python
"""Re-solve the frozen-bath SCF of a saved centre, without rebuilding anything.

Usage:

    python hact_relax/resolve_hact.py RUN_DIR --rescue-unconverged-scf \\
        --rescue-tol-factor 20 --scf-max-cycle 400

Options left out take the run's own settings, which the file records.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if os.fspath(_REPO) not in sys.path:
    sys.path.insert(0, os.fspath(_REPO))

from hact_relax._bootstrap import THREADS  # sets the thread env before NumPy

import numpy as np
from pyscf import lib

from hact_relax.checkpoint import (
    load_hact_input,
    solver_from_hact_input,
    write_hact_checkpoint,
)
from hact_relax.surface import follow_negative_mode, rescue_unconverged


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_dir", type=Path,
                   help="the case directory a run wrote (hact_input.chk)")
    p.add_argument("--rescue-unconverged-scf", default=None,
                   action=argparse.BooleanOptionalAction)
    p.add_argument("--rescue-tol-factor", type=float)
    p.add_argument("--follow-negative-mode", default=None,
                   action=argparse.BooleanOptionalAction)
    p.add_argument("--level-shift", type=float)
    p.add_argument("--scf-max-cycle", type=int)
    p.add_argument("--scf-conv-tol", type=float)
    p.add_argument("--scf-conv-tol-grad", type=float)
    p.add_argument("--no-continuation", action="store_true",
                   help="start from the pristine seed even when the centre was "
                        "continued from the previous one")
    p.add_argument("--out", type=Path,
                   help="the hact.chk to write (default RUN_DIR/hact.chk)")
    p.add_argument("--threads", type=int, default=THREADS)
    p.add_argument("--memory-mb", type=int, default=4000)
    return p.parse_args(argv)


def _pick(value, stored, cast):
    return cast(stored) if value is None else cast(value)


def main(argv=None) -> int:
    args = parse_args(argv)
    lib.num_threads(args.threads)
    started = time.time()
    run_dir = args.run_dir.resolve()
    try:
        data = load_hact_input(run_dir / "hact_input.chk")
    except (FileNotFoundError, RuntimeError) as exc:
        print("# error: %s" % exc, file=sys.stderr, flush=True)
        return 2

    settings = {
        "scf_conv_tol": _pick(args.scf_conv_tol, data["scf_conv_tol"], float),
        "scf_conv_tol_grad": _pick(args.scf_conv_tol_grad,
                                   data["scf_conv_tol_grad"], float),
        "scf_max_cycle": _pick(args.scf_max_cycle, data["max_cycle"], int),
        "level_shift": _pick(args.level_shift, data["level_shift"], float),
        "rescue_unconverged_scf": _pick(args.rescue_unconverged_scf,
                                        data["rescue_unconverged_scf"], bool),
        "rescue_tol_factor": _pick(args.rescue_tol_factor,
                                   data["rescue_tol_factor"], float),
        "follow_negative_mode": _pick(args.follow_negative_mode,
                                      data["follow_negative_mode"], bool),
        "follow_mode_tol": float(data["follow_mode_tol"]),
        "continuation": (not args.no_continuation
                         and data["start_rotation"] is not None),
    }

    solver = solver_from_hact_input(data, memory_mb=args.memory_mb)
    solver.level_shift = settings["level_shift"]
    solver.max_cycle = settings["scf_max_cycle"]
    trace = []
    solver.scf_callback = lambda cycle, e, delta, g: trace.append((cycle, e, delta, g))
    print("# %s frag%s, evaluation %s: %d active orbitals, %s electrons; "
          "re-solving with %s"
          % (data["case"], data["fragment"], data["evaluation"],
             int(solver.active_orb.sum()), solver.nelecas,
             ", ".join("%s=%s" % kv for kv in settings.items())), flush=True)

    start = {}
    if settings["continuation"]:
        start = {"start_rotation": np.asarray(data["start_rotation"]),
                 "start_occ": np.asarray(data["start_occ"])}
    result = solver.kernel(conv_tol=settings["scf_conv_tol"],
                           conv_tol_grad=settings["scf_conv_tol_grad"], **start)
    report = {
        "run_dir": os.fspath(run_dir),
        "input_checkpoint": os.fspath(run_dir / "hact_input.chk"),
        "settings": settings,
        "scf_cycles": len(trace),
        "scf_converged_by_diis": bool(result[0]),
        "scf_final_orbital_gradient_norm": trace[-1][3] if trace else None,
        "scf_rescued": False,
        "orbitals_canonicalized": False,
    }
    if not result[0] and settings["rescue_unconverged_scf"]:
        limit = settings["scf_conv_tol_grad"] * settings["rescue_tol_factor"]
        result, rescue = rescue_unconverged(result, limit, settings["scf_conv_tol_grad"])
        report.update(rescue)
    if result[0] and settings["follow_negative_mode"]:
        result, _, mode = follow_negative_mode(result, settings["follow_mode_tol"],
                                               settings["scf_conv_tol_grad"])
        report.update(mode)

    report["converged"] = bool(result[0])
    out = args.out or (run_dir / "hact.chk")
    if result[0]:
        energy = float(result[1] + solver.E1e_core + solver.E2e_core + solver.Enuc)
        report.update({
            "e_total_hact_v_hartree": energy,
            "e_active_hartree": float(result[1]),
            "e1e_core_hartree": float(solver.E1e_core),
            "e2e_core_hartree": float(solver.E2e_core),
            "enuc_hartree": float(solver.Enuc),
            "n_active_orbitals": int(solver.active_orb.sum()),
            "n_entangled_bath": int(solver.n_bath),
            "hact_checkpoint": os.fspath(out),
        })
        meta = {name: data[name] for name in (
            "case", "fragment", "geometry_key", "evaluation", "kmesh",
            "coords_bohr", "labels")}
        # An accepted centre locks the bath rank before the run writes hact.chk.
        meta["bath_rank_locked"] = 1
        write_hact_checkpoint(
            out,
            meta=meta,
            solver=solver,
            result=result,
            energy=energy,
            checkpoint_eri=True,
            extra={"resolved_offline": 1},
        )
    report["wall_seconds"] = round(time.time() - started, 3)
    (run_dir / "resolve_hact.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float) + "\n"
    )

    if not result[0]:
        print("# NOT converged after %d cycles (final |g| %s)%s; nothing written but "
              "%s" % (len(trace), report["scf_final_orbital_gradient_norm"],
                      "" if not settings["rescue_unconverged_scf"] else
                      ", and the rescue ended at |g| %.2e"
                      % report.get("scf_rescue_gradient_norm", float("nan")),
                      run_dir / "resolve_hact.json"), flush=True)
        return 1
    print("# converged%s%s after %d cycles: E(H_act^V) = %.12f Ha  (E_active %.12f)"
          % (" by the rescue" if report["scf_rescued"] else "",
             ", negative mode followed" if report.get("negative_mode_followed")
             else "", len(trace), report["e_total_hact_v_hartree"],
             report["e_active_hartree"]), flush=True)
    print("# wrote %s (%.1f s)" % (out, report["wall_seconds"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
