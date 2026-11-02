#!/usr/bin/env python
"""Schmidt bath spectra of LiH chain fragments from a stored pristine density, with no SCF.

The bath is the SVD of the pristine density's environment-fragment block
(schmidt_embedding.py), so a pristine.chk gives the spectrum of any fragment.  For another
cell count the stored Fock, cut into real-space blocks F(R), is placed on that chain, each
block that fits once, and diagonalized there: an interpolation, exact only as far as F(R)
has decayed and the Fock does not change with the chain.

Usage:

    python hact_relax/bath_from_pristine.py RUN_DIR/pristine.chk --cells 21 40
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if os.fspath(_REPO) not in sys.path:
    sys.path.insert(0, os.fspath(_REPO))

from hact_relax._bootstrap import THREADS  # sets the thread env before NumPy

import h5py
import numpy as np
from pyscf import lib
from pyscf.pbc import gto
from scipy.linalg import eigh

from frozen_bath_scf.schmidt_embedding import iao_pao_supercell
from hact_relax.geometry import (
    BOHR,
    build_starting_geometry,
    nearest_atoms,
    pristine_chain_body,
)

KEPT_MIN = 1e-5
RATIO_LOOSE = 0.1


def load(path):
    with h5py.File(path, "r") as h:
        raw = h["mol"][()]
        coeff = np.asarray(h["scf/mo_coeff"][0]).real
        energy = np.asarray(h["scf/mo_energy"][0])
        occ = np.asarray(h["scf/mo_occ"][0])
    return gto.loads(raw.decode() if isinstance(raw, bytes) else raw), coeff, energy, occ


def real_space_fock(cell, coeff, energy):
    """F(R) for R = -N//2 ... N//2 of the stored N-cell chain, N odd."""
    ncell, nao = cell.natm // 2, cell.nao_nr() // (cell.natm // 2)
    if ncell % 2 == 0:
        raise ValueError("the stored chain must have an odd cell count")
    sc = np.asarray(cell.pbc_intor("int1e_ovlp", hermi=1)).real @ coeff
    fock = (sc * energy) @ sc.T
    return nao, {r: fock[:nao, (r % ncell) * nao:(r % ncell + 1) * nao]
                 for r in range(-(ncell // 2), ncell // 2 + 1)}


def chain_of(cell, ncell):
    lattice = cell.lattice_vectors() * BOHR
    bond = lattice[0, 0] / cell.natm
    out = cell.copy()
    out.atom = pristine_chain_body(ncell, bond)
    out.a = np.diag([2.0 * bond * ncell, lattice[1, 1], lattice[2, 2]])
    out.unit = "A"
    out.build(verbose=0)
    return out


def occupied(cell, nao, blocks):
    ncell = cell.natm // 2
    overlap = np.asarray(cell.pbc_intor("int1e_ovlp", hermi=1)).real
    fock = np.zeros_like(overlap)
    for i in range(ncell):
        for r, block in blocks.items():
            if 2 * abs(r) < ncell:
                j = (i + r) % ncell
                fock[i * nao:(i + 1) * nao, j * nao:(j + 1) * nao] = block
    _, vectors = eigh(0.5 * (fock + fock.T), overlap)
    return vectors[:, :2 * ncell], overlap


def spectrum(cell, gamma, atom_ids, centre, fragment):
    atoms = sorted(nearest_atoms(cell.atom_coords(), cell.lattice_vectors(), centre, fragment))
    inside = np.isin(atom_ids, atoms)
    block = gamma[np.ix_(np.flatnonzero(~inside), np.flatnonzero(inside))] / 2.0
    return np.sort(np.linalg.svd(block, compute_uv=False))[::-1]


def best_cut(s):
    """The cut keeping the most orbitals with a kept value >= 1e-5 and a ratio <= 0.1."""
    cuts = [(k, s[k - 1], s[k]) for k in range(1, len(s)) if s[k - 1] >= KEPT_MIN]
    cuts = [c for c in cuts if c[2] / c[1] <= RATIO_LOOSE]
    return max(cuts, key=lambda c: c[0]) if cuts else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pristine", type=Path)
    parser.add_argument("--cells", type=int, nargs="+", default=[21])
    parser.add_argument("--fragments", type=int, nargs="+", default=[3, 5, 7, 9, 11])
    parser.add_argument("--cases", nargs="+", default=["vlc", "vhc"], choices=("vlc", "vhc"))
    parser.add_argument("--threads", type=int, default=THREADS)
    args = parser.parse_args()
    lib.num_threads(args.threads)
    largest = max(args.fragments)
    for ncell in args.cells:
        if largest > 2 * ncell:
            parser.error("fragment %d exceeds the %d-atom %d-cell chain"
                         % (largest, 2 * ncell, ncell))
    stored, coeff, energy, occ = load(args.pristine)
    nao, blocks = real_space_fock(stored, coeff, energy)
    for ncell in args.cells:
        if ncell == stored.natm // 2:
            cell, orbocc = stored, coeff[:, occ > 0]
            overlap = np.asarray(cell.pbc_intor("int1e_ovlp", hermi=1)).real
        else:
            cell = chain_of(stored, ncell)
            orbocc, overlap = occupied(cell, nao, blocks)
        c_lo, _, atom_ids = iao_pao_supercell(cell, orbocc, ovlp=overlap, minao="minao",
                                              return_atom_ids=True)
        sc = overlap @ c_lo
        gamma = sc.T @ (2.0 * orbocc @ orbocc.T) @ sc
        for case in args.cases:
            centre = build_starting_geometry(case, ncell)[2]
            for fragment in args.fragments:
                s = spectrum(cell, gamma, atom_ids, centre, fragment)
                cut = best_cut(s)
                row = "%2d cells %s frag%-2d nF %3d" % (ncell, case, fragment, len(s))
                if cut is not None:
                    k, kept, nxt = cut
                    row += " | nB %2d kept %.2e next %.2e ratio %.3g bath_tol %.0e" % (
                        k, kept, nxt, nxt / kept, np.sqrt(kept * nxt))
                print(row)
                print("    below 1e-2: " + " ".join("%.2e" % x for x in s[s < 1e-2][:16]))


if __name__ == "__main__":
    main()
