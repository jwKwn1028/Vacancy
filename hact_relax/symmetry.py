"""Rotations about the vacancy as AO matrices, and a frozen-bath start they leave
invariant."""
from __future__ import annotations

import numpy as np

from frozen_bath_scf import second_order as so

# Largest first: C3 at an hBN B site, C2 on a chain.
ROTATION_ORDERS = (6, 4, 3, 2)


def rotation_operator(cell, coords, lattice, center, theta, tol=1e-6):
    """AO matrix and atom permutation of the rotation by theta about z through ``center``;
    (None, None) if it does not map the crystal onto itself."""
    from pyscf.symm.Dmatrix import Dmatrix

    c, s = np.cos(theta), np.sin(theta)
    q = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    rotated = coords[center] + (coords - coords[center]) @ q.T
    inv = np.linalg.inv(lattice.T)
    target = np.empty(cell.natm, dtype=int)
    for a in range(cell.natm):
        d = (coords - rotated[a]) @ inv.T
        d -= np.round(d)
        dist = np.linalg.norm(d @ lattice, axis=1)
        b = int(np.argmin(dist))
        if dist[b] > tol or cell.atom_symbol(b) != cell.atom_symbol(a):
            return None, None
        target[a] = b
    ao_loc = cell.ao_loc_nr()
    shells = {a: [i for i in range(cell.nbas) if cell.bas_atom(i) == a]
              for a in range(cell.natm)}
    blocks = {}
    nao = cell.nao_nr()
    rot = np.zeros((nao, nao))
    for a in range(cell.natm):
        for ia, ib in zip(shells[a], shells[target[a]]):
            ang = cell.bas_angular(ia)
            if ang not in blocks:
                blocks[ang] = Dmatrix(ang, theta, 0.0, 0.0, reorder_p=True)
            n = 2 * ang + 1
            for k in range(cell.bas_nctr(ia)):
                p0, q0 = ao_loc[ia] + k * n, ao_loc[ib] + k * n
                rot[q0:q0 + n, p0:p0 + n] = blocks[ang]
    return rot, target


def site_rotation(cell, overlap, center, orders=ROTATION_ORDERS, tol=1e-8):
    """(n, R) of the highest-order rotation about z through ``center`` with R^T S R = S."""
    coords = np.asarray(cell.atom_coords(), dtype=float)
    lattice = np.asarray(cell.lattice_vectors(), dtype=float)
    for n in orders:
        for sign in (1.0, -1.0):
            rot, _ = rotation_operator(cell, coords, lattice, center, sign * 2 * np.pi / n)
            if rot is None:
                break
            if np.abs(rot.T @ overlap @ rot - overlap).max() < tol:
                return n, rot
    return None, None


def active_representation(coeff, overlap, rotation):
    """Polar factor of C^T S R C and its smallest singular value (1: invariant)."""
    u, s, vt = np.linalg.svd(coeff.T @ overlap @ rotation @ coeff)
    return u @ vt, float(s.min())


def rotation_group(w, n):
    """I, W, ..., W^(n-1)."""
    ops = [np.eye(len(w))]
    for _ in range(n - 1):
        ops.append(ops[-1] @ w)
    return ops


def _fill(sizes, scores, count, taken):
    """Whole sets, highest score first, adding up to ``count``; None if none do."""
    chosen, left = [], count
    for k in sorted(set(range(len(sizes))) - taken, key=lambda k: -scores[k]):
        if left == 0:
            break
        if sizes[k] <= left:
            chosen.append(k)
            left -= sizes[k]
    return chosen if left == 0 else None


def symmetric_frame(h1e, eri, occ, ops, tol=1e-6):
    """A frame of ``occ`` with ``ops``-invariant occupied spaces: eigenvectors of the
    averaged Fock, in whole degenerate sets, filled in energy order or, if a set straddles a
    boundary, by their occupation in the averaged densities.  None if neither works."""
    def average(m):
        return sum(w @ m @ w.T for w in ops) / len(ops)

    dm_a, dm_b = so.densities(occ)
    dm_a, dm_b = average(dm_a), average(dm_b)
    v_a, v_b = so.general_veff(eri, dm_a, dm_b)
    energies, vectors = np.linalg.eigh(average(h1e + 0.5 * (v_a + v_b)))
    doubly, single = np.isclose(occ, 2.0), np.isclose(occ, 1.0)
    nd, ns = int(doubly.sum()), int(single.sum())
    starts = [0] + [i for i in range(1, energies.size)
                    if energies[i] - energies[i - 1] > tol] + [energies.size]
    sets = [np.arange(a, b) for a, b in zip(starts[:-1], starts[1:])]
    report = {"frontier_energies": energies[max(0, nd - 3):nd + ns + 3].tolist(),
              "fill": "aufbau"}
    if nd in starts and nd + ns in starts:
        order_d, order_s = np.arange(nd), np.arange(nd, nd + ns)
    else:
        report["fill"] = "densities"
        sizes = [len(s) for s in sets]

        def score(dm):
            return [float(np.trace(vectors[:, s].T @ dm @ vectors[:, s])) / len(s)
                    for s in sets]

        d = _fill(sizes, score(dm_b), nd, set())
        o = None if d is None else _fill(sizes, score(dm_a - dm_b), ns, set(d))
        if o is None:
            return None, report
        order_d = np.concatenate([sets[k] for k in sorted(d)]).astype(int)
        order_s = (np.concatenate([sets[k] for k in sorted(o)]).astype(int) if o
                   else np.zeros(0, dtype=int))
    rest = np.setdiff1d(np.arange(energies.size), np.concatenate([order_d, order_s]))
    frame = np.empty_like(vectors)
    frame[:, doubly] = vectors[:, order_d]
    frame[:, single] = vectors[:, order_s]
    frame[:, ~(doubly | single)] = vectors[:, rest]
    vs = vectors[:, order_s]
    report["singly_occupied_character"] = (float(np.trace(vs.T @ ops[1] @ vs))
                                           if len(ops) > 1 and ns else None)
    return frame, report
