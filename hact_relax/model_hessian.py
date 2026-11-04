"""Lindh's model Hessian (Chem. Phys. Lett. 241, 423 (1995)) as a Cartesian starting
Hessian: stretches and bends over periodic images, no torsions."""
import itertools

import numpy as np

K_STRETCH = 0.45
K_BEND = 0.15
# Rows of the periodic table (1: H, He; 2: Li-Ne; 3: the rest).
ALPHA = np.array([[1.0000, 0.3949, 0.3949],
                  [0.3949, 0.2800, 0.2800],
                  [0.3949, 0.2800, 0.2800]])
R_REF = np.array([[1.35, 2.10, 2.53],
                  [2.10, 2.87, 3.40],
                  [2.53, 3.40, 3.40]])
RHO_MIN = 1e-8          # dropped weights
LINEAR_SIN = 1e-6       # skipped (linear) bends
GUESS = 0.5             # geomeTRIC's Cartesian guess, kept on fixed coordinates
FLOOR = 0.005           # smallest curvature on the moving coordinates


def _period_row(z):
    return 0 if z <= 2 else (1 if z <= 10 else 2)


def _neighbours(coords, rows, lattice):
    """[(i, j, r_ij, rho_ij)] over real atoms and the images of j."""
    n = len(coords)
    shifts = [np.zeros(3)] if lattice is None else [
        np.asarray(t) @ lattice for t in itertools.product((-1, 0, 1), repeat=3)]
    out = []
    for i in range(n):
        if rows[i] < 0:
            continue
        for j in range(n):
            if rows[j] < 0:
                continue
            for t in shifts:
                if i == j and not t.any():
                    continue
                r = coords[j] + t - coords[i]
                d2 = float(r @ r)
                a, b = rows[i], rows[j]
                rho = np.exp(ALPHA[a, b] * (R_REF[a, b] ** 2 - d2))
                if rho > RHO_MIN:
                    out.append((i, j, r, rho))
    return out


def lindh_hessian(coords, charges, lattice=None, free=None):
    """Model Hessian (3n, 3n), Hartree/Bohr^2, at ``coords`` (Bohr); zero charges are ghosts.
    With ``free`` (3n bools), other coordinates get GUESS and no coupling, and the free
    block's eigenvalues are raised to FLOOR."""
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    lattice = None if lattice is None else np.asarray(lattice, dtype=float)
    rows = [(-1 if z == 0 else _period_row(int(round(z)))) for z in np.asarray(charges)]
    h = np.zeros((n, 3, n, 3))

    def add(k, parts):
        # k g g^T, g given per atom.
        for p, gp in parts.items():
            for q, gq in parts.items():
                h[p, :, q, :] += k * np.outer(gp, gq)

    pairs = _neighbours(coords, rows, lattice)
    arms = {}
    for i, j, r, rho in pairs:
        d = np.linalg.norm(r)
        if j != i:
            arms.setdefault(i, []).append((j, r / d, d, rho))
        if i < j:
            u = r / d
            add(K_STRETCH * rho, {i: -u, j: u})
    for c, near in arms.items():
        for (a, ea, da, wa), (b, eb, db, wb) in itertools.combinations(near, 2):
            # ea, eb: unit vectors from the centre c to the neighbours.
            cos = float(np.clip(ea @ eb, -1.0, 1.0))
            sin = np.sqrt(max(0.0, 1.0 - cos * cos))
            if sin < LINEAR_SIN:
                continue
            ga = (cos * ea - eb) / (da * sin)
            gb = (cos * eb - ea) / (db * sin)
            parts = {c: -(ga + gb)}
            parts[a] = parts.get(a, 0.0) + ga
            parts[b] = parts.get(b, 0.0) + gb
            add(K_BEND * wa * wb, parts)
    h = h.reshape(3 * n, 3 * n)
    if free is None:
        return h
    free = np.asarray(free, dtype=bool)
    out = GUESS * np.eye(3 * n)
    idx = np.flatnonzero(free)
    if len(idx):
        w, v = np.linalg.eigh(h[np.ix_(idx, idx)])
        out[np.ix_(idx, idx)] = (v * np.maximum(w, FLOOR)) @ v.T
    return out


def free_coordinates(natm, movable, axes):
    """3 natm bools: ``movable`` atoms along ``axes``."""
    free = np.zeros((natm, 3), dtype=bool)
    for a in movable:
        free[int(a), list(axes)] = True
    return free.ravel()


def write_model_hessian(path, mol, lattice, movable, axes):
    """Write lindh_hessian of ``mol`` for geomeTRIC's hessian="file:" option."""
    h = lindh_hessian(mol.atom_coords(), mol.atom_charges(), lattice,
                      free_coordinates(mol.natm, movable, axes))
    np.savetxt(path, h)
    return path
