"""The two checkpoints an accepted centre writes, and the solver rebuilt from one.

"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

import numpy as np

HACT_GROUP = "hact"
INPUT_GROUP = "hact_input"


@contextlib.contextmanager
def _atomic_h5(path):
    """An HDF5 file written beside ``path`` and moved over it only when complete."""
    import h5py

    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    try:
        with h5py.File(partial, "w") as handle:
            yield handle
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def _string_array(values):
    import h5py

    return np.asarray(list(values), dtype=h5py.special_dtype(vlen=str))


def _write_meta(group, meta: dict[str, Any]) -> None:
    for name, value in meta.items():
        if name == "labels":
            group[name] = _string_array(value)
        elif value is not None:
            group[name] = value


def _write_embedding(group, solver, checkpoint_eri) -> None:
    singular = getattr(solver, "bath_singular_values", None)
    group["mo_coeff_embedding"] = np.asarray(solver.mo_coeff).real
    group["mo_occ_embedding"] = np.asarray(solver.mo_occ).real
    group["active_orb"] = np.asarray(solver.active_orb, dtype=np.int8)
    group["n_bath"] = int(solver.n_bath)
    group["fragment_atoms"] = np.asarray(sorted(solver.frag_atoms), dtype=int)
    if singular is not None:
        group["bath_singular_values"] = np.asarray(singular, dtype=float)
    group["e1e_core"] = float(solver.E1e_core)
    group["e2e_core"] = float(solver.E2e_core)
    group["enuc"] = float(solver.Enuc)
    group["core_energy_computed"] = 1


def _write_eri(group, eri) -> None:
    """The active ERIs: density-fitting factors (eri_active_df, eri_active_df_sign), or
    a dense eri_active."""
    from frozen_bath_scf.df_eri import DFEri, as_eri

    eri = as_eri(eri)
    if isinstance(eri, DFEri):
        group["eri_active_df"] = eri.factors
        group["eri_active_df_sign"] = eri.sign
    else:
        group["eri_active"] = eri


def eri_from_data(data):
    """The active ERIs of a loaded group (either form), or None."""
    from frozen_bath_scf.df_eri import DFEri

    if data.get("eri_active_df") is not None:
        return DFEri(data["eri_active_df"], data.get("eri_active_df_sign"))
    if data.get("eri_active") is not None:
        return np.asarray(data["eri_active"], dtype=float)
    return None


def write_hact_checkpoint(path, *, meta, solver, result, energy, checkpoint_eri,
                          extra=None) -> None:
    """Write group ``hact``: a converged frozen-bath SCF.  Replaces the file."""
    with _atomic_h5(path) as handle:
        group = handle.create_group(HACT_GROUP)
        _write_meta(group, meta)
        group["converged"] = int(bool(result[0]))
        group["e_tot"] = float(energy)
        group["e_elec_active"] = float(result[1])
        group["mo_coeff_active"] = np.asarray(result[3]).real
        group["mo_occ_active"] = np.asarray(result[4]).real
        group["mo_energy_active"] = np.asarray(result[2]).real
        group["h1e_active"] = np.asarray(result[5]).real
        if checkpoint_eri and result[6] is not None:
            _write_eri(group, result[6])
        _write_embedding(group, solver, checkpoint_eri)
        _write_meta(group, extra or {})


def write_hact_input_checkpoint(path, *, meta, solver, checkpoint_eri, scf,
                                start_rotation=None, start_occ=None) -> None:
    """Write group ``hact_input``: what the frozen-bath SCF starts from.  Replaces the file."""
    from frozen_bath_scf.frozen_rohf import Frozen_ROHF

    is_rohf = isinstance(solver, Frozen_ROHF)
    nelecas = np.atleast_1d(np.asarray(solver.nelecas, dtype=int))
    seed = solver.mo_occ_D if is_rohf else solver.mo_occ
    with _atomic_h5(path) as handle:
        group = handle.create_group(INPUT_GROUP)
        group["status"] = "written before the frozen-bath SCF"
        _write_meta(group, meta)
        group["solver_kind"] = "rohf" if is_rohf else "rhf"
        group["nelecas"] = nelecas
        group["level_shift"] = float(solver.level_shift)
        group["max_cycle"] = int(solver.max_cycle)
        group["seed_vacancy"] = int(bool(solver.seed_vacancy))
        group["vac_idx_sc"] = np.asarray(solver.vac_idx_sc, dtype=int)
        group["fragment_center_idx"] = np.asarray(solver.fragment_center_idx, dtype=int)
        for name, value in scf.items():
            group[name] = value
        group["heff"] = np.asarray(solver.heff).real
        if checkpoint_eri:
            _write_eri(group, solver.eri_active)
        group["mo_occ_seed"] = np.asarray(seed).real
        _write_embedding(group, solver, checkpoint_eri)
        c_def = getattr(solver, "_c_def", None)
        if c_def is not None:
            group["c_def"] = np.asarray(c_def).real
        if start_rotation is not None and start_occ is not None:
            group["start_rotation"] = np.asarray(start_rotation, dtype=float)
            group["start_occ"] = np.asarray(start_occ, dtype=float)


def _decode(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.dtype.kind in "OS":
        return [v.decode() if isinstance(v, bytes) else str(v) for v in value]
    return value


def load_hact_input(path) -> dict[str, Any]:
    """The ``hact_input`` group as a dict; missing optional items are None."""
    import h5py

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            "%s does not exist: the run predates the input checkpoint, or it "
            "never reached an accepted centre's SCF" % path
        )
    with h5py.File(path, "r") as handle:
        if INPUT_GROUP not in handle:
            raise RuntimeError("%s has no %s group" % (path, INPUT_GROUP))
        group = handle[INPUT_GROUP]
        data = {name: _decode(group[name][()]) for name in group}
    for name in ("eri_active", "eri_active_df", "eri_active_df_sign", "c_def",
                 "start_rotation", "start_occ", "bath_singular_values"):
        data.setdefault(name, None)
    return data


def solver_from_hact_input(data: dict[str, Any], memory_mb: int = 4000):
    """The frozen-bath solver of a saved centre, ready for ``kernel``.

    The class is the production one (``SchmidtEmbeddedRHF``/``ROHF``), so the SCF
    loop, the occupation rule and the defect-orbital tracking are the same code;
    only ``build`` is skipped, its products being restored from the file.  The
    supercell is rebuilt for its AO overlap, which the defect-orbital tracking uses.
    """
    from pyscf.pbc import gto as pbc_gto
    from pyscf.scf import rohf

    from frozen_bath_scf.schmidt_embedding import (
        SchmidtEmbeddedRHF,
        SchmidtEmbeddedROHF,
    )
    from hact_relax.geometry import ghost_species

    if eri_from_data(data) is None:
        raise RuntimeError(
            "the input checkpoint has no active ERIs (the run used "
            "--no-eri-checkpoint), so its SCF cannot be re-solved"
        )
    cell = pbc_gto.Cell()
    cell.atom = [[ghost_species(label), xyz]
                 for label, xyz in zip(data["labels"], np.asarray(data["coords_bohr"]))]
    cell.a = np.asarray(data["lattice_bohr"], dtype=float)
    cell.unit = "Bohr"
    cell.basis = data["basis"]
    cell.verbose = 0
    cell.max_memory = int(memory_mb)
    cell.build()

    is_rohf = data["solver_kind"] == "rohf"
    cls = SchmidtEmbeddedROHF if is_rohf else SchmidtEmbeddedRHF
    solver = cls.__new__(cls)
    rohf.ROHF.__init__(solver, cell)
    solver.verbose = 0
    solver.scell = cell
    solver.S = np.asarray(cell.pbc_intor("int1e_ovlp", hermi=1)).real
    solver.mo_coeff = np.asarray(data["mo_coeff_embedding"], dtype=float)
    solver.mo_occ = np.asarray(data["mo_occ_embedding"], dtype=float).copy()
    solver.active_orb = np.asarray(data["active_orb"]).astype(bool)
    if solver.mo_coeff.shape[0] != cell.nao_nr():
        raise RuntimeError(
            "the rebuilt supercell has %d AOs, the saved orbitals %d"
            % (cell.nao_nr(), solver.mo_coeff.shape[0])
        )
    seed = np.asarray(data["mo_occ_seed"], dtype=float).copy()
    nelecas = [int(n) for n in np.atleast_1d(data["nelecas"])]
    if is_rohf:
        solver.mo_occ_D = seed
        solver.spin = int(data["spin"])
        solver.nelecas = nelecas
    else:
        solver.mo_occ = seed
        solver.nelecas = nelecas[0]
    solver.heff = np.asarray(data["heff"], dtype=float)
    solver.eri_active = eri_from_data(data)
    solver.E1e_core = float(data["e1e_core"])
    solver.E2e_core = float(data["e2e_core"])
    solver.Enuc = float(data["enuc"])
    solver.compute_core_energy = True
    solver.level_shift = float(data["level_shift"])
    solver.max_cycle = int(data["max_cycle"])
    solver.seed_vacancy = bool(int(data["seed_vacancy"]))
    solver.vac_idx_sc = [int(i) for i in np.atleast_1d(data["vac_idx_sc"])]
    solver.fragment_center_idx = [
        int(i) for i in np.atleast_1d(data["fragment_center_idx"])
    ]
    solver.defect = True
    solver.charge = int(data["charge"])
    solver._c_def = (None if data["c_def"] is None
                     else np.asarray(data["c_def"], dtype=float))
    solver.n_bath = int(data["n_bath"])
    solver.frag_atoms = set(int(a) for a in np.atleast_1d(data["fragment_atoms"]))
    solver.bath_singular_values = (
        None if data["bath_singular_values"] is None
        else np.asarray(data["bath_singular_values"], dtype=float)
    )
    solver.eri_mode = "cluster"
    return solver
