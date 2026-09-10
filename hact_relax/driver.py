from __future__ import annotations

import contextlib
import os
from pathlib import Path
import time

import numpy as np

from pyscf.geomopt import geometric_solver
from pyscf.geomopt.addons import as_pyscf_method

from hact_relax.cases import CASES, FULL_CHAIN_KMESH, fragment_relax_radius
from hact_relax.errors import GeometryOptimizationNotConverged
from hact_relax.geometry import (
    build_driver_molecule,
    build_starting_geometry,
    check_geometry_fits_lattice,
    full_chain_lattice,
    ghost_species,
    movable_atoms,
    patch_geometric_for_ghost_atoms,
    read_xyz,
    seed_displace_from_vacancy,
    write_xyz,
)
from hact_relax.gradient import (
    FiniteDifferenceScanner,
    save_final_gradient,
    write_constraints,
)
from hact_relax.surface import ActiveHamiltonianSurface, SurfaceSettings


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


def run_one(
    args,
    case_name: str,
    fragment: int,
    input_xyz: Path | None = None,
) -> Path:
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
    relax_radius = fragment_relax_radius(fragment, args.lattice_r)
    movable = movable_atoms(labels, coords_bohr, vacancy_index, relax_radius)
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
        if args.single_point:
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

    return output_dir
