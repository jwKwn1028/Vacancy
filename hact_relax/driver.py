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
    SemiAnalyticGradientScanner,
    TrajectoryRecorder,
    point_count_per_gradient,
    save_final_gradient,
    write_constraints,
)
from hact_relax.recording import RunRecorder
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


def run_configuration(
    args,
    case_name: str,
    fragment: int,
    labels,
    vacancy_index: int,
    movable,
    output_dir: Path,
    input_description: str,
    wrap_bond_ang: float,
    points_per_gradient: int,
) -> dict:
    case = CASES[case_name]
    chain_cells = int(args.chain_cells)
    chain_length = 2.0 * float(args.lattice_r) * chain_cells
    semi_analytic = bool(getattr(args, "semi_analytic_gradient", False))
    return {
        "case_name": case_name,
        "fragment": fragment,
        "state": "ground",
        "relaxed_on": "E[H_act^V]",
        "basis": args.basis,
        "auxbasis": None,
        "chain_cells": chain_cells,
        "primitive_cell_atoms": 2,
        "full_chain_atoms": len(labels),
        "full_chain_kmesh": list(FULL_CHAIN_KMESH),
        "reference_kmesh": list(FULL_CHAIN_KMESH),
        "lattice_r_ang": float(args.lattice_r),
        "vacuum_ang": float(args.vacuum),
        "chain_length_ang": chain_length,
        "vacancy_image_separation_ang": chain_length,
        "wrap_bond_ang": float(wrap_bond_ang),
        "seed_displace_ang": float(args.seed_displace),
        "input_xyz": None if args.xyz is None else os.fspath(args.xyz),
        "input_geometry": input_description,
        "output_dir": os.fspath(output_dir),
        "vacancy_index_zero_based": int(vacancy_index),
        "movable_atoms_zero_based": [int(i) for i in movable],
        "relax_radius_ang": fragment_relax_radius(fragment, args.lattice_r),
        "relax_radius_policy": "fragment extent from N_Frag",
        "relax_radius_request": "auto",
        "relax_on_active_energy": False,
        "pinned_chain_end_atoms_zero_based": [],
        "defective_supercell_scf": False,
        "full_chain_is_large_cell_scf": True,
        "initial_uniform_geometry_primitive_kmesh": [chain_cells, 1, 1],
        "periodic_scf": "pristine full-chain Gamma KRHF with the vacancy restored",
        "pristine_reference_policy": "exact",
        "fragment_policy": "nearest atoms at first accepted centre, fixed thereafter",
        "bath_rank_policy": "threshold at first accepted centre, fixed thereafter",
        "hact_scf_continuation":
            "converged active solution from the last accepted centre",
        "bath_tol": float(args.bath_tol),
        "level_shift": float(case.level_shift),
        "level_shift_source": "case default",
        "scf_max_cycle": int(args.scf_max_cycle),
        "scf_conv_tol": float(args.scf_conv_tol),
        "scf_conv_tol_grad": float(args.scf_conv_tol_grad),
        "rescue_unconverged_scf": bool(args.rescue_unconverged_scf),
        "rescue_tol_factor": float(args.rescue_tol_factor),
        "follow_negative_mode": bool(args.follow_negative_mode),
        "follow_mode_tol": 1e-8,
        "min_subspace_overlap": float(args.min_subspace_overlap),
        "min_center_subspace_overlap": float(args.min_center_subspace_overlap),
        "gradient_policy": (
            "semi-analytic: active space frozen at the displaced points"
            if semi_analytic else
            "finite difference: active space relaxed at every point"
        ),
        "semi_analytic_gradient": semi_analytic,
        "fd_step_bohr": float(args.fd_step),
        "fd_axes": str(args.fd_axes),
        "fd_mode": str(args.fd_mode),
        "points_per_gradient": int(points_per_gradient),
        "max_gradient_hartree_per_bohr": float(args.max_gradient),
        "coordsys": str(args.coordsys),
        "maxsteps": int(args.maxsteps),
        "convergence_energy": float(args.convergence_energy),
        "convergence_grms": float(args.convergence_grms),
        "convergence_gmax": float(args.convergence_gmax),
        "convergence_drms": float(args.convergence_drms),
        "convergence_dmax": float(args.convergence_dmax),
        "tmax": float(args.tmax),
        "single_point": bool(args.single_point),
        "allow_unconverged_geometry": bool(args.allow_unconverged_geometry),
        "checkpoint_eri": not bool(getattr(args, "no_eri_checkpoint", False)),
        "memory_mb": int(args.memory_mb),
        "threads": int(args.threads),
        "verbose": int(args.verbose),
    }


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
    wrap_bond_ang = check_geometry_fits_lattice(
        coords_bohr, chain_cells, args.lattice_r, input_description
    )

    coords_bohr = seed_displace_from_vacancy(
        coords_bohr, vacancy_index, args.seed_displace
    )
    movable = movable_atoms(
        labels, coords_bohr, vacancy_index, fragment, args.lattice_r
    )
    if not movable:
        raise ValueError("no real atoms selected for relaxation")

    output_stem = "%s-frag%d" % (case_name, fragment)
    if getattr(args, "semi_analytic_gradient", False):
        output_stem += "-semi-analytic"
    output_dir = unique_output_dir(args.output_dir, output_stem)
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
        follow_negative_mode=bool(args.follow_negative_mode),
        rescue_unconverged_scf=bool(args.rescue_unconverged_scf),
        rescue_tol_factor=float(args.rescue_tol_factor),
        checkpoint_eri=not bool(getattr(args, "no_eri_checkpoint", False)),
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
    initial_xyz = output_dir / "initial.xyz"
    write_xyz(initial_xyz, labels, coords_bohr, input_description)

    trajectory = TrajectoryRecorder(output_dir, labels)
    scanner_type = (
        SemiAnalyticGradientScanner
        if getattr(args, "semi_analytic_gradient", False)
        else FiniteDifferenceScanner
    )
    recorder = RunRecorder(
        output_dir,
        run_configuration(
            args, case_name, fragment, labels, vacancy_index, movable,
            output_dir, input_description, wrap_bond_ang,
            point_count_per_gradient(movable, args.fd_axis_indices, args.fd_mode),
        ),
        resource_interval_seconds=float(
            getattr(args, "resource_interval", 30.0)
        ),
    )
    surface = ActiveHamiltonianSurface(
        settings,
        labels,
        vacancy_index,
        full_chain_lattice(chain_cells, args.lattice_r, args.vacuum),
        recorder=recorder,
    )
    scanner = scanner_type(
        surface,
        movable,
        args.fd_step,
        args.max_gradient,
        axes=args.fd_axis_indices,
        mode=args.fd_mode,
        trajectory=trajectory,
    )

    constraints_path = None if args.coordsys == "cart" else constraints
    patch_geometric_for_ghost_atoms()
    method = as_pyscf_method(driver_mol, scanner)
    geometry_converged: bool | None = None
    outcome: dict = {"run_status": "failed", "error": None}
    started = time.time()

    try:
        recorder.phase("optimizing")
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

        recorder.phase("finalizing")
        final_coords = np.asarray(relaxed.atom_coords())
        if scanner.last_coords is None or not np.allclose(
            scanner.last_coords, final_coords, atol=1e-11, rtol=0.0
        ):
            energy, gradient = scanner(relaxed)
        if energy is None or gradient is None:
            raise RuntimeError(
                "optimizer returned without an H_act^V energy/gradient"
            )

        final_xyz = output_dir / "final.xyz"
        write_xyz(
            final_xyz,
            labels,
            final_coords,
            "%s frag%d E=%s Ha" % (case_name, fragment, energy),
        )
        npy_path, txt_path = save_final_gradient(output_dir, labels, gradient)

        frozen = sorted(set(range(len(labels))) - set(movable))
        outcome.update({
            "run_status": "complete",
            "final_energy_hartree": float(energy),
            "final_total_hact_v_hartree": float(energy),
            "final_energy_terms_hartree": surface.last_energy_terms,
            "final_gradient_norm_hartree_per_bohr": float(
                np.linalg.norm(np.asarray(gradient))
            ),
            "frozen_coordinate_drift_bohr": (
                float(np.abs(final_coords[frozen] - coords_bohr[frozen]).max())
                if frozen else 0.0
            ),
            "final_xyz": os.fspath(final_xyz),
            "final_gradient_npy": os.fspath(npy_path),
            "final_gradient_txt": os.fspath(txt_path),
        })
        if geometry_converged is False:
            outcome["run_status"] = "geometry-not-converged"
            if not args.allow_unconverged_geometry:
                raise GeometryOptimizationNotConverged(
                    "geomeTRIC did not converge in %d steps; final artifacts "
                    "were saved under %s, but this geometry was not accepted"
                    % (args.maxsteps, output_dir),
                    output_dir=os.fspath(output_dir),
                )
        return output_dir
    except BaseException as exc:
        outcome["error"] = "%s: %s" % (type(exc).__name__, exc)
        raise
    finally:
        outcome.update({
            "geometry_optimizer_converged": geometry_converged,
            "geometry_accepted_by_override": bool(
                args.allow_unconverged_geometry
            ),
            "optimization_wall_seconds": round(time.time() - started, 6),
            "gradient_calls": scanner.calls,
            "measured_energy_noise_floor_hartree": surface.measured_noise_floor,
            "measured_suggested_fd_step_bohr": (
                None if surface.measured_noise_floor is None
                else (3.0 * surface.measured_noise_floor) ** (1.0 / 3.0)
            ),
            "fixed_n_entangled_bath": surface.fixed_n_bath,
            "fixed_fragment_atoms_zero_based": (
                None if surface.fixed_fragment_atoms is None
                else [int(a) for a in surface.fixed_fragment_atoms]
            ),
            "initial_xyz": os.fspath(initial_xyz),
            "constraints_file": (None if constraints is None
                                 else os.fspath(constraints)),
            "trajectory_xyz": os.fspath(trajectory.xyz_path),
            "trajectory_jsonl": os.fspath(trajectory.jsonl_path),
            "trajectory_npz": os.fspath(trajectory.npz_path),
            "gradient_trajectory_npy": os.fspath(trajectory.gradient_path),
            "pristine_chkfile": (os.fspath(recorder.pristine_chkfile)
                                 if recorder.pristine_chkfile.exists() else None),
            "hact_chkfile": (os.fspath(recorder.hact_chkfile)
                             if recorder.hact_chkfile.exists() else None),
        })
        recorder.summary(outcome)
        recorder.close()
