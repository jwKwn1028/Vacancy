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
    """Whole sets adding up to ``count`` with the most occupation (knapsack); None if none
    do."""
    best = {0: (0.0, [])}
    for k in sorted(set(range(len(sizes))) - taken, key=lambda k: -scores[k]):
        for c, (value, chosen) in sorted(best.items(), reverse=True):
            n = c + sizes[k]
            gain = value + scores[k] * sizes[k]
            if n <= count and (n not in best or gain > best[n][0] + 1e-12):
                best[n] = (gain, chosen + [k])
    return best[count][1] if count in best else None


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


# Candidate point operations of a hexagonal sheet or an x chain: z rotations by 30 degrees,
# with or without a vertical and the horizontal mirror.
def _candidate_matrices():
    out = []
    for k in range(12):
        c, s = np.cos(k * np.pi / 6), np.sin(k * np.pi / 6)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        for vertical in (np.eye(3), np.diag([1.0, -1.0, 1.0])):
            for horizontal in (np.eye(3), np.diag([1.0, 1.0, -1.0])):
                out.append(rot @ vertical @ horizontal)
    return out


class SiteGroup:
    """Point operations x -> c + Q (x - c) about atom ``center`` mapping ``coords`` onto
    itself, as (Q, perm), perm[a] the image of atom a."""

    def __init__(self, coords, lattice, labels, center, tol=1e-6):
        self.coords = np.asarray(coords, dtype=float)
        self.lattice = np.asarray(lattice, dtype=float)
        self.center = int(center)
        self.tol = float(tol)
        labels = [str(x) for x in labels]
        self.ops = []
        for q in _candidate_matrices():
            perm = self._permutation(self.coords, q, labels)
            if perm is not None and not any(np.allclose(q, p) for p, _ in self.ops):
                self.ops.append((q, perm))
        self.inverse = [np.argsort(perm) for _, perm in self.ops]

    def _wrap(self, d):
        frac = d @ np.linalg.inv(self.lattice)
        return (frac - np.round(frac)) @ self.lattice

    def _permutation(self, coords, q, labels):
        c = coords[self.center]
        images = c + (coords - c) @ q.T
        perm = np.empty(len(coords), dtype=int)
        for a, r in enumerate(images):
            dist = np.linalg.norm(self._wrap(coords - r), axis=1)
            b = int(np.argmin(dist))
            if dist[b] > self.tol or labels[b] != labels[a]:
                return None
            perm[a] = b
        return perm if len(set(perm.tolist())) == len(perm) else None

    @property
    def order(self):
        return len(self.ops)

    def field(self, u):
        """The group average of a displacement or gradient field (natm, 3)."""
        u = np.asarray(u, dtype=float)
        out = np.zeros_like(u)
        for q, perm in self.ops:
            out[perm] += u @ q.T
        return out / self.order

    def geometry(self, coords):
        """``coords`` averaged over the group."""
        x = np.asarray(coords, dtype=float)
        c = x[self.center]
        shift = np.zeros_like(x)
        for (q, _), inv in zip(self.ops, self.inverse):
            shift += self._wrap(c + (x[inv] - c) @ q.T - x)
        return x + shift / self.order

    def check_subspace(self, movable, axes):
        """Refuse movable atoms or axes the group does not map onto themselves."""
        allowed = np.zeros(self.coords.shape, dtype=bool)
        allowed[np.ix_(list(movable), list(axes))] = True
        for q, perm in self.ops:
            moved = np.zeros_like(allowed)
            moved[perm] = (allowed.astype(float) @ np.abs(q).T) > 1e-12
            if (moved & ~allowed).any():
                raise ValueError("the site group does not map the movable atoms %s on axes "
                                 "%s onto themselves" % (list(movable), list(axes)))
