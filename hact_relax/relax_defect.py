#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

_REPO = Path(__file__).resolve().parents[1]
if os.fspath(_REPO) not in sys.path:
    sys.path.insert(0, os.fspath(_REPO))

from hact_relax._bootstrap import REPO, THREADS

from pyscf import lib

from hact_relax.cases import (
    CASES,
    CHAIN_CELLS,
    R_ANG,
    parse_cases,
    parse_fragments,
)
from hact_relax.driver import run_one


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Relax LiH vacancies on the embedded H_act^V surface"
    )
    parser.add_argument("cases", nargs="?", default="all")
    parser.add_argument("fragments", nargs="?", default="all")

    parser.add_argument("--xyz", type=Path)
    parser.add_argument("--output-dir", type=Path, default=REPO / "result")
    parser.add_argument("--basis", default="pob-tzvp")
    parser.add_argument("--chain-cells", type=int, default=CHAIN_CELLS)
    parser.add_argument("--lattice-r", type=float, default=R_ANG)
    parser.add_argument("--vacuum", type=float, default=30.0)
    parser.add_argument("--seed-displace", type=float, default=0.0)
    parser.add_argument("--coordsys", default="cart",
                        choices=("tric", "cart", "prim", "dlc", "hdlc", "tric-p"))

    parser.add_argument("--bath-tol", type=float, default=1e-6)
    parser.add_argument("--scf-max-cycle", type=int, default=200)
    parser.add_argument("--scf-conv-tol", type=float, default=1e-9)
    parser.add_argument("--scf-conv-tol-grad", type=float, default=1e-6)
    parser.add_argument("--rescue-unconverged-scf", action="store_true")
    parser.add_argument("--rescue-tol-factor", type=float, default=5.0)
    parser.add_argument("--follow-negative-mode", action="store_true")
    parser.add_argument("--min-subspace-overlap", type=float, default=0.5)
    parser.add_argument("--min-center-subspace-overlap", type=float, default=0.0)

    parser.add_argument("--state", choices=("ground", "excited"),
                        default="ground",
                        help="ground: relax on the frozen-bath SCF energy. "
                             "excited: relax on a tracked CASCI root of the "
                             "same H_act^V")
    parser.add_argument("--root", type=int, default=1,
                        help="zero-based CASCI root, selected by energy at the "
                             "first accepted centre and by maximum CI overlap "
                             "thereafter")
    parser.add_argument("--nroots", type=int, default=4)
    parser.add_argument("--ncas", type=int,
                        help="CAS orbitals; defaults to the case value")
    parser.add_argument("--ncas-elec", type=int,
                        help="CAS electrons; defaults to the case value")
    parser.add_argument("--casci-two-s", type=int, default=None,
                        help="CASCI spin sector as 2S (0 = singlet, 2 = "
                             "triplet).  Defaults to the case's own spin. "
                             "H_act^V is spin-free, so the same Hamiltonian can "
                             "be diagonalized in another sector")
    parser.add_argument("--min-casci-root-overlap", type=float, default=0.5,
                        help="abort if no CASCI root overlaps the tracked state "
                             "by this much")

    parser.add_argument("--fd-step", type=float, default=1e-3)
    parser.add_argument("--fd-axes", default="x")
    parser.add_argument("--fd-mode", default="central",
                        choices=("central", "forward"))
    parser.add_argument("--max-gradient", type=float, default=1.0)
    parser.add_argument("--semi-analytic-gradient",
                        "--semi-analytical-gradient", action="store_true")

    parser.add_argument("--maxsteps", type=int, default=50)
    parser.add_argument("--convergence-energy", type=float, default=1e-6)
    parser.add_argument("--convergence-grms", type=float, default=3e-4)
    parser.add_argument("--convergence-gmax", type=float, default=4.5e-4)
    parser.add_argument("--convergence-drms", type=float, default=1.2e-3)
    parser.add_argument("--convergence-dmax", type=float, default=1.8e-3)
    parser.add_argument("--tmax", type=float, default=0.15)
    parser.add_argument("--allow-unconverged-geometry", action="store_true")
    parser.add_argument("--single-point", action="store_true")

    parser.add_argument("--resource-interval", type=float, default=30.0)
    parser.add_argument("--no-eri-checkpoint", action="store_true")

    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--memory-mb", type=int,
                        default=int(os.environ.get("SLURM_MEM_PER_NODE", "4000")))
    parser.add_argument("--verbose", type=int, default=4)
    return parser


def validate_args(parser: argparse.ArgumentParser, args):
    try:
        cases = parse_cases(args.cases)
        fragments = parse_fragments(args.fragments)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.threads != THREADS:
        parser.error("--threads must be parsed before NumPy; pass it once")
    if args.memory_mb < 1 or args.fd_step <= 0 or args.maxsteps < 1:
        parser.error("--memory-mb, --fd-step and --maxsteps must be positive")
    if args.max_gradient <= 0.0 or args.bath_tol <= 0.0:
        parser.error("--max-gradient and --bath-tol must be positive")
    if not 0.0 <= args.min_subspace_overlap <= 1.0:
        parser.error("--min-subspace-overlap must be between zero and one")
    if not 0.0 <= args.min_center_subspace_overlap <= 1.0:
        parser.error("--min-center-subspace-overlap must be between zero and one")
    if args.root < 0:
        parser.error("--root is zero-based and must not be negative")
    if args.nroots < 1:
        parser.error("--nroots must be positive")
    if not 0.0 <= args.min_casci_root_overlap <= 1.0:
        parser.error("--min-casci-root-overlap must be between zero and one")
    if (args.state != "ground"
            and args.min_center_subspace_overlap < args.min_subspace_overlap):
        print("# warning: --min-center-subspace-overlap (%g) is below "
              "--min-subspace-overlap (%g).  On --state excited the centre "
              "fallback re-seeds the SCF in an unrelated orbital frame, which "
              "the CASCI orbital guard then rejects anyway -- the run aborts "
              "later and with a less obvious message"
              % (args.min_center_subspace_overlap, args.min_subspace_overlap),
              flush=True)
    if args.state == "ground" and args.casci_two_s is not None:
        print("# warning: --casci-two-s is ignored with --state ground",
              flush=True)
    if args.state != "ground":
        for name in cases:
            case = CASES[name]
            ncas = args.ncas if args.ncas is not None else case.ncas
            ncas_elec = (args.ncas_elec if args.ncas_elec is not None
                         else case.ncas_elec)
            if ncas < 1 or ncas_elec < 0:
                parser.error("--ncas must be positive and --ncas-elec "
                             "non-negative (case %s)" % name)
            if ncas_elec > 2 * ncas:
                parser.error(
                    "CAS(%de,%do) holds at most %d electrons (case %s)"
                    % (ncas_elec, ncas, 2 * ncas, name)
                )
            two_s = case.spin if args.casci_two_s is None else args.casci_two_s
            if two_s < 0:
                parser.error("--casci-two-s is 2S and must not be negative")
            if (two_s - ncas_elec) % 2:
                parser.error(
                    "2S=%d has the wrong parity for %d CAS electrons (case %s)"
                    % (two_s, ncas_elec, name)
                )
            if two_s > min(ncas_elec, 2 * ncas - ncas_elec):
                parser.error(
                    "2S=%d exceeds the maximum spin of CAS(%de,%do) (case %s)"
                    % (two_s, ncas_elec, ncas, name)
                )
    axes = "".join(dict.fromkeys(str(args.fd_axes).strip().lower()))
    if not axes or set(axes) - set("xyz"):
        parser.error("--fd-axes must be a non-empty subset of 'xyz'")
    args.fd_axis_indices = tuple(sorted("xyz".index(c) for c in axes))
    if args.chain_cells < 1:
        parser.error("--chain-cells must be positive")
    if args.lattice_r <= 0.0:
        parser.error("--lattice-r must be positive")
    if args.xyz is not None and len(cases) * len(fragments) != 1:
        parser.error("--xyz can only describe one case/fragment combination")
    largest = max(fragments)
    if largest > 2 * args.chain_cells:
        parser.error(
            "N_Frag=%d exceeds the %d-atom %d-cell chain"
            % (largest, 2 * args.chain_cells, args.chain_cells)
        )
    args.output_dir = args.output_dir.expanduser().resolve()
    return cases, fragments


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cases, fragments = validate_args(parser, args)
    lib.num_threads(THREADS)

    outputs = []
    for fragment in fragments:
        for case_name in cases:
            input_xyz = args.xyz.resolve() if args.xyz is not None else None
            outputs.append(run_one(args, case_name, fragment, input_xyz))
    print("# completed outputs:")
    for path in outputs:
        print("#   %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
