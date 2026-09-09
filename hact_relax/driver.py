from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass
import os
from pathlib import Path
import time

import numpy as np

from pyscf.geomopt import geometric_solver
from pyscf.geomopt.addons import as_pyscf_method

from hact_relax.cases import CASES, FULL_CHAIN_KMESH
from hact_relax.errors import (
    GeometryOptimizationNotConverged,
    SubspaceContinuityError,
)
from hact_relax.geometry import (
    build_driver_molecule,
    build_starting_geometry,
    check_geometry_fits_lattice,
    full_chain_lattice,
    ghost_species,
    movable_atoms,
    patch_geometric_for_ghost_atoms,
    read_xyz,
    scaled_geometry,
    seed_displace_from_vacancy,
    write_xyz,
)
from hact_relax.gradient import (
    FiniteDifferenceScanner,
    save_final_gradient,
    write_constraints,
)
from hact_relax.surface import ActiveHamiltonianSurface, SurfaceSettings


@dataclass
class RunContext:
    output_dir: Path
    surface: ActiveHamiltonianSurface
    final_coords: np.ndarray
    final_energy: float


@contextlib.contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def unique_output_dir(base: Path, stem: str) -> Path:
    candidate = base / stem
    if not candidate.exists():
        return candidate
    suffix = time.strftime("%Y%m%d-%H%M%S")
    numbered = base / (stem + "-" + suffix)
    serial = 1
    while numbered.exists():
        serial += 1
        numbered = base / (stem + "-" + suffix + "-%d" % serial)
    return numbered


def _sub_args(args, **overrides):
    clone = copy.copy(args)
    for key, value in overrides.items():
        setattr(clone, key, value)
    return clone


def run_one(
    args,
    case_name: str,
    fragment: int,
    input_xyz: Path | None = None,
    *,
    continuation: ActiveHamiltonianSurface | None = None,
    include_pristine_continuation: bool = True,
) -> RunContext:
    case = CASES[case_name]
    chain_cells = int(args.chain_cells)
    if input_xyz is None:
        labels, coords_bohr, vacancy_index = build_starting_geometry(
            case_name, chain_cells, args.lattice_r
        )
        input_description = "generated LiH-1D + central %s vacancy" % case.vacancy
    else:
        labels, coords_bohr, vacancy_index = read_xyz(input_xyz)
        input_description = os.fspath(input_xyz)

    parent_species = ghost_species(labels[vacancy_index])
    if parent_species.lower() != case.vacancy.lower():
        raise ValueError(
            "%s contains %s at the ghost site, expected %s"
            % (input_description, parent_species, case.vacancy)
        )
    expected_natm = 2 * chain_cells
    if len(labels) != expected_natm:
        raise ValueError(
            "%s has %d atoms; a %d-cell full LiH chain requires %d atoms"
            % (input_description, len(labels), chain_cells, expected_natm)
        )
    if fragment > expected_natm:
        raise ValueError(
            "fragment size %d exceeds the %d-atom full chain"
            % (fragment, expected_natm)
        )
    check_geometry_fits_lattice(
        coords_bohr, chain_cells, args.lattice_r, input_description
    )

    coords_bohr = seed_displace_from_vacancy(
        coords_bohr, vacancy_index, args.seed_displace
    )
    movable = movable_atoms(
        labels, coords_bohr, vacancy_index, args.relax_radius
    )
    if not movable:
        raise ValueError("no real atoms selected for relaxation")

    output_dir = unique_output_dir(
        args.output_dir, "%s-frag%d" % (case_name, fragment)
    )
    output_dir.mkdir(parents=True)

    settings = SurfaceSettings(
        case_name=case_name,
        fragment=fragment,
        reference_kmesh=FULL_CHAIN_KMESH,
        basis=args.basis,
        memory_mb=args.memory_mb,
        bath_tol=args.bath_tol,
        level_shift=case.level_shift,
        scf_max_cycle=args.scf_max_cycle,
        scf_conv_tol=args.scf_conv_tol,
        scf_conv_tol_grad=args.scf_conv_tol_grad,
        min_subspace_overlap=args.min_subspace_overlap,
        min_center_subspace_overlap=args.min_center_subspace_overlap,
        verbose=args.verbose,
        frozen_embedding=bool(args.frozen_embedding),
        frozen_embedding_outside_fragment=bool(
            args.frozen_embedding_outside_fragment
        ),
        follow_negative_mode=bool(args.follow_negative_mode),
        rescue_unconverged_scf=bool(args.rescue_unconverged_scf),
        rescue_tol_factor=float(args.rescue_tol_factor),
    )
    driver_mol = build_driver_molecule(
        labels, coords_bohr, case, args.basis, args.memory_mb, args.verbose
    )
    constraints = write_constraints(
        output_dir / "constraints.txt",
        driver_mol.natm,
        movable,
        args.fd_axis_indices,
    )
    write_xyz(output_dir / "initial.xyz", labels, coords_bohr, input_description)

    surface = ActiveHamiltonianSurface(
        settings,
        labels,
        vacancy_index,
        full_chain_lattice(chain_cells, args.lattice_r, args.vacuum),
    )
    if continuation is not None:
        surface.inherit_continuation(
            continuation, include_pristine=include_pristine_continuation
        )
    scanner = FiniteDifferenceScanner(
        surface,
        movable,
        args.fd_step,
        args.max_gradient,
        axes=args.fd_axis_indices,
        mode=args.fd_mode,
    )

    constraints_path = None if args.coordsys == "cart" else constraints
    patch_geometric_for_ghost_atoms()
    method = as_pyscf_method(driver_mol, scanner)
    geometry_converged: bool | None = None

    with working_directory(output_dir):
        if args.energy_only:
            energy = surface.energy(
                np.asarray(driver_mol.atom_coords()),
                "single-point",
                accept_center=True,
                compute_core_energy=args.compute_core_energy,
                strict_continuation=args.strict_continuation,
            )
            gradient = None
            relaxed = driver_mol
        elif args.single_point:
            energy, gradient = scanner(driver_mol)
            relaxed = driver_mol
        else:
            geometry_converged, relaxed = geometric_solver.kernel(
                method,
                include_ghost=True,
                maxsteps=args.maxsteps,
                constraints=(os.fspath(constraints_path)
                             if constraints_path else None),
                assert_convergence=not args.allow_unconverged_geometry,
                convergence_energy=args.convergence_energy,
                convergence_grms=args.convergence_grms,
                convergence_gmax=args.convergence_gmax,
                convergence_drms=args.convergence_drms,
                convergence_dmax=args.convergence_dmax,
                tmax=args.tmax,
                coordsys=args.coordsys,
            )
            geometry_converged = bool(geometry_converged)
            energy = scanner.last_energy
            gradient = scanner.last_gradient

    final_coords = np.asarray(relaxed.atom_coords())
    if not args.energy_only:
        if scanner.last_coords is None or not np.allclose(
            scanner.last_coords, final_coords, atol=1e-11, rtol=0.0
        ):
            energy, gradient = scanner(relaxed)
        if energy is None or gradient is None:
            raise RuntimeError(
                "optimizer returned without an H_act^V energy/gradient"
            )

    write_xyz(
        output_dir / "final.xyz",
        labels,
        final_coords,
        "%s frag%d E=%s Ha" % (case_name, fragment, energy),
    )
    if gradient is not None:
        save_final_gradient(output_dir, labels, gradient)

    if geometry_converged is False and not args.allow_unconverged_geometry:
        raise GeometryOptimizationNotConverged(
            "geomeTRIC did not converge in %d steps; final artifacts were "
            "saved under %s, but this geometry was not accepted"
            % (args.maxsteps, output_dir),
            output_dir=os.fspath(output_dir),
        )

    return RunContext(
        output_dir=output_dir,
        surface=surface,
        final_coords=np.asarray(final_coords, dtype=float).copy(),
        final_energy=float(energy),
    )


def run_chain_length_relaxation(
    args, case_name: str, fragment: int, input_xyz: Path | None = None
) -> Path:
    base = args.output_dir
    base.mkdir(parents=True, exist_ok=True)
    lattice_r = float(args.lattice_r)
    xyz = input_xyz
    final_context: RunContext | None = None
    continuation: ActiveHamiltonianSurface | None = None
    max_passes = int(args.max_chain_length_steps)
    max_halvings = int(args.max_cell_continuation_halvings)
    chain_cells = int(args.chain_cells)

    for step in range(max_passes):
        length = 2.0 * lattice_r * chain_cells
        print("\n# chain-length pass %d/%d: r = %.6f A, L = %.5f A"
              % (step + 1, max_passes, lattice_r, length), flush=True)
        pass_dir = base / ("pass-%02d" % step)
        outcome = run_one(
            _sub_args(
                args,
                lattice_r=lattice_r,
                output_dir=pass_dir,
                seed_displace=(args.seed_displace if step == 0 else 0.0),
            ),
            case_name,
            fragment,
            xyz,
            continuation=continuation,
            include_pristine_continuation=not bool(args.frozen_embedding),
        )
        final_context = outcome
        relaxed_xyz = outcome.output_dir / "final.xyz"
        labels, coords, _ = read_xyz(relaxed_xyz)
        centre = outcome.final_energy
        cell_continuation = outcome.surface

        if args.frozen_embedding:
            exact_centre = run_one(
                _sub_args(
                    args,
                    lattice_r=lattice_r,
                    output_dir=pass_dir / "centre-exact",
                    single_point=True,
                    energy_only=True,
                    relax_radius=None,
                    seed_displace=0.0,
                    frozen_embedding=False,
                    frozen_embedding_outside_fragment=False,
                    compute_core_energy=True,
                    strict_continuation=False,
                ),
                case_name,
                fragment,
                relaxed_xyz,
                continuation=outcome.surface,
            )
            centre = exact_centre.final_energy
            cell_continuation = exact_centre.surface

        eps = float(args.chain_length_epsilon)
        wings: dict[str, float] = {}

        def evaluate_strain_path(tag: str, sign: float) -> RunContext:
            probe_serial = 0

            def advance(start_strain, start_surface, target_strain, depth,
                        *, direct):
                nonlocal probe_serial
                probe_serial += 1
                r_target = lattice_r * (1.0 + target_strain)
                is_endpoint = bool(np.isclose(
                    target_strain, sign * eps, atol=0.0, rtol=1e-14
                ))
                if direct:
                    xyz_target = pass_dir / ("chain-%s.xyz" % tag)
                    output_base = pass_dir / ("wing-" + tag)
                else:
                    retry_root = pass_dir / "cell-continuation" / ("wing-" + tag)
                    xyz_target = retry_root / ("probe-%03d.xyz" % probe_serial)
                    output_base = retry_root / ("probe-%03d" % probe_serial)
                write_xyz(
                    xyz_target,
                    labels,
                    scaled_geometry(coords, target_strain),
                    "%s at fractional cell strain %+0.8f; lattice-r %.8f"
                    % (relaxed_xyz.name, target_strain, r_target),
                )

                wing_args = _sub_args(
                    args,
                    lattice_r=r_target,
                    output_dir=output_base,
                    single_point=True,
                    energy_only=True,
                    relax_radius=None,
                    seed_displace=0.0,
                    frozen_embedding=False,
                    frozen_embedding_outside_fragment=False,
                    compute_core_energy=is_endpoint,
                    strict_continuation=True,
                )
                try:
                    return run_one(
                        wing_args,
                        case_name,
                        fragment,
                        xyz_target,
                        continuation=start_surface,
                    )
                except SubspaceContinuityError:
                    if depth >= max_halvings:
                        raise
                    midpoint = 0.5 * (start_strain + target_strain)
                    middle = advance(
                        start_strain, start_surface, midpoint, depth + 1,
                        direct=False,
                    )
                    return advance(
                        midpoint, middle.surface, target_strain, depth + 1,
                        direct=False,
                    )

            return advance(0.0, cell_continuation, sign * eps, 0, direct=True)

        for tag, sign in (("minus", -1.0), ("plus", +1.0)):
            wings[tag] = evaluate_strain_path(tag, sign).final_energy

        h = 2.0 * lattice_r * eps * chain_cells
        d1 = (wings["plus"] - wings["minus"]) / (2.0 * h)
        d2 = (wings["plus"] - 2.0 * centre + wings["minus"]) / (h * h)
        if not np.isfinite(d1) or not np.isfinite(d2):
            raise RuntimeError(
                "non-finite chain-length finite difference: dE/dL=%r, "
                "d2E/dL2=%r" % (d1, d2)
            )
        print("# chain length: dE/dL = %+.6e Ha/A, d2E/dL2 = %+.6e Ha/A^2"
              % (d1, d2), flush=True)
        if d2 <= 0.0:
            break

        cap = float(args.max_chain_length_shift) * length
        shift = float(np.clip(-d1 / d2, -cap, cap))
        print("# chain length: dL = %+.6f A (%+.4f%% of L)"
              % (shift, 100.0 * shift / length), flush=True)
        if abs(shift) < float(args.chain_length_tol):
            break
        if step + 1 >= max_passes:
            break

        scale = (length + shift) / length
        next_lattice_r = lattice_r * scale
        next_xyz = pass_dir / "chain-next.xyz"
        write_xyz(
            next_xyz,
            labels,
            scaled_geometry(coords, scale - 1.0),
            "scaled to lattice-r %.6f for the next pass" % next_lattice_r,
        )

        continuation = cell_continuation
        lattice_r = next_lattice_r
        xyz = next_xyz

    if final_context is None:
        raise RuntimeError("chain-length relaxation completed no atomic pass")
    return final_context.output_dir
