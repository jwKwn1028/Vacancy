import numpy as np
from scipy.linalg import expm, expm_frechet

from .df_eri import eri_j, eri_k


def general_veff(eri, dm_a, dm_b):
    j = eri_j(eri, dm_a + dm_b)
    return [j - eri_k(eri, dm_a), j - eri_k(eri, dm_b)]


def energy_from_dm(h1e, dm_a, dm_b, eri):
    v_a, v_b = general_veff(eri, dm_a, dm_b)
    e_1e = np.einsum("ij,ji->", h1e, dm_a) + np.einsum("ij,ji->", h1e, dm_b)
    e_2e = 0.5 * (np.einsum("ij,ji->", v_a, dm_a) + np.einsum("ij,ji->", v_b, dm_b))
    return float(e_1e + e_2e)


def rotation_mask(mo_occ):
    occ = np.asarray(mo_occ)
    occ_a = occ > 0
    occ_b = occ == 2
    var_a = (~occ_a).reshape(-1, 1) & occ_a
    var_b = (~occ_b).reshape(-1, 1) & occ_b
    return var_a | var_b


def densities(mo_occ):
    occ = np.asarray(mo_occ, dtype=float)
    return np.diag((occ > 0).astype(float)), np.diag((occ == 2).astype(float))


def symmetrize(matrix, ops):
    """The average of W M W^T over the group ``ops``."""
    return sum(w @ matrix @ w.T for w in ops) / len(ops)


def _block_ops(ops, mo_occ):
    """``ops`` on the occupation blocks only, each block orthogonalized."""
    occ = np.asarray(mo_occ)
    blocks = [np.flatnonzero(np.isclose(occ, k)) for k in (2.0, 1.0, 0.0)]
    out = []
    for w in ops:
        b = np.zeros_like(w)
        for idx in blocks:
            if idx.size:
                u, _, vt = np.linalg.svd(w[np.ix_(idx, idx)])
                b[np.ix_(idx, idx)] = u @ vt
        out.append(b)
    return out


def symmetry_projector(ops, mo_occ):
    """The orthogonal projector onto rotation vectors invariant under ``ops``, whose frame
    must have invariant occupied spaces; None for no ops."""
    if ops is None:
        return None
    mask = rotation_mask(mo_occ)
    ops = _block_ops(ops, mo_occ)

    def project(x):
        return symmetrize(_kappa_from_vector(np.asarray(x, dtype=float), mask), ops)[mask]

    return project


def _expm_antisym(kappa):
    return expm(kappa)


def _rotate(matrix, u):
    return u.T @ matrix @ u


def _kappa_from_vector(vector, mask):
    kappa = np.zeros(mask.shape)
    kappa[mask] = vector
    return kappa - kappa.T


def orbital_gradient(h1e, eri, mo_occ, kappa=None):
    mask = rotation_mask(mo_occ)
    dm_a, dm_b = densities(mo_occ)
    if kappa is not None:
        u = _expm_antisym(kappa)
        dm_a = u @ dm_a @ u.T
        dm_b = u @ dm_b @ u.T
    v_a, v_b = general_veff(eri, dm_a, dm_b)
    f_a, f_b = h1e + v_a, h1e + v_b
    if kappa is not None:
        u = _expm_antisym(kappa)
        f_a, f_b = _rotate(f_a, u), _rotate(f_b, u)
    occ = np.asarray(mo_occ)
    occ_a, occ_b = occ > 0, occ == 2
    var_a = (~occ_a).reshape(-1, 1) & occ_a
    var_b = (~occ_b).reshape(-1, 1) & occ_b
    g = np.zeros_like(f_a)
    g[var_a] = f_a[var_a]
    g[var_b] += f_b[var_b]
    return g[mask]


def energy_and_gradient(h1e, eri, mo_occ, kappa):
    """E(exp(kappa)) and its exact derivative in the antisymmetric kappa, through the
    exponential's Frechet adjoint."""
    u = _expm_antisym(kappa)
    dm_a, dm_b = densities(mo_occ)
    da, db = u @ dm_a @ u.T, u @ dm_b @ u.T
    v_a, v_b = general_veff(eri, da, db)
    energy = float(np.einsum("ij,ji->", h1e, da + db)
                   + 0.5 * (np.einsum("ij,ji->", v_a, da) + np.einsum("ij,ji->", v_b, db)))
    du = 2 * ((h1e + v_a) @ u @ dm_a + (h1e + v_b) @ u @ dm_b)
    return energy, expm_frechet(kappa.T, du, compute_expm=False)


def _exact_gradient_vector(h1e, eri, mo_occ, x, mask):
    _, g = energy_and_gradient(h1e, eri, mo_occ, _kappa_from_vector(x, mask))
    return g[mask] - g.T[mask]


def hessian_vector(h1e, eri, mo_occ, v, step=1e-4, mask=None):
    """The orbital Hessian (orbital_hessian's half scale) times ``v``, by a central
    difference of the exact gradient."""
    mask = rotation_mask(mo_occ) if mask is None else mask
    v = np.asarray(v, dtype=float)
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return np.zeros_like(v)
    x = (step / norm) * v
    plus = _exact_gradient_vector(h1e, eri, mo_occ, x, mask)
    minus = _exact_gradient_vector(h1e, eri, mo_occ, -x, mask)
    return (plus - minus) * (norm / (4.0 * step))


def hessian_diagonal_estimate(h1e, eri, mo_occ):
    """Orbital-energy differences F_pp - F_qq per rotation and spin channel: the
    preconditioner of the iterative Hessian solves."""
    mask = rotation_mask(mo_occ)
    dm_a, dm_b = densities(mo_occ)
    v_a, v_b = general_veff(eri, dm_a, dm_b)
    f_a, f_b = np.diag(h1e + v_a), np.diag(h1e + v_b)
    occ = np.asarray(mo_occ)
    occ_a, occ_b = occ > 0, occ == 2
    var_a = (~occ_a).reshape(-1, 1) & occ_a
    var_b = (~occ_b).reshape(-1, 1) & occ_b
    d = np.zeros(mask.shape)
    d[var_a] += (f_a[:, None] - f_a[None, :])[var_a]
    d[var_b] += (f_b[:, None] - f_b[None, :])[var_b]
    return d[mask]


def lowest_hessian_modes(h1e, eri, mo_occ, nroots=1, tol=1e-8, max_cycle=200,
                         step=1e-4, dense_below=64, project=None):
    """The ``nroots`` lowest orbital-Hessian eigenpairs, by Davidson on Hessian-vector
    products (dense below ``dense_below`` rotations), within the range of ``project`` if
    given. Returns (values, vectors, converged, products)."""
    from pyscf import lib

    mask = rotation_mask(mo_occ)
    n = int(mask.sum())
    if n <= max(dense_below, 2 * nroots + 2):
        hess = orbital_hessian(h1e, eri, mo_occ, step)
        if project is None:
            w, v = np.linalg.eigh(hess)
            return w[:nroots], v[:, :nroots], True, n
        pw, pv = np.linalg.eigh(np.array([project(e) for e in np.eye(n)]).T)
        q = pv[:, pw > 0.5]
        w, v = np.linalg.eigh(q.T @ hess @ q)
        return w[:nroots], q @ v[:, :nroots], True, n
    diag = hessian_diagonal_estimate(h1e, eri, mo_occ)
    count = [0]
    keep = (lambda x: x) if project is None else project

    def aop(xs):
        count[0] += len(xs)
        return [keep(hessian_vector(h1e, eri, mo_occ, keep(x), step, mask)) for x in xs]

    def precond(r, e, x0):
        d = diag - e
        return keep(r / np.where(np.abs(d) > 1e-4, d, np.copysign(1e-4, d)))

    # Unit vectors alone can miss a whole symmetry block (Davidson never leaves the
    # space it starts in): a fixed random vector breaks the symmetry, and two extra
    # roots catch a degenerate lowest pair.
    nsolve = min(nroots + 2, n - 1)
    guesses = []
    for p in np.argsort(diag)[:nsolve + 2]:
        unit = np.zeros(n)
        unit[p] = 1.0
        guesses.append(keep(unit))
    noise = keep(np.random.default_rng(20261027).standard_normal(n))
    guesses.append(noise)
    guesses = [g / np.linalg.norm(g) for g in guesses if np.linalg.norm(g) > 1e-8]
    conv, e, x = lib.davidson1(aop, guesses, precond, tol=tol, max_cycle=max_cycle,
                               max_space=max(24, 4 * nsolve), nroots=nsolve)
    e = np.atleast_1d(e)
    x = np.asarray(x).reshape(len(e), n).T
    order = np.argsort(e)[:nroots]
    return e[order], x[:, order], bool(np.all(conv)), count[0]


def orbital_hessian(h1e, eri, mo_occ, step=1e-4):
    mask = rotation_mask(mo_occ)
    n = int(mask.sum())
    hess = np.zeros((n, n))
    for p in range(n):
        unit = np.zeros(n)
        unit[p] = step
        plus = orbital_gradient(h1e, eri, mo_occ, _kappa_from_vector(unit, mask))
        minus = orbital_gradient(h1e, eri, mo_occ, _kappa_from_vector(-unit, mask))
        hess[:, p] = (plus - minus) / (2.0 * step)
    return 0.5 * (hess + hess.T)


def stability_analysis(h1e, eri, mo_occ, step=1e-4, nroots=1, **kwargs):
    """The lowest ``nroots`` orbital-Hessian eigenpairs (lowest_hessian_modes).
    ``n_negative`` counts negative eigenvalues among those computed only."""
    eigenvalues, eigenvectors, converged, products = lowest_hessian_modes(
        h1e, eri, mo_occ, nroots=nroots, step=step, **kwargs)
    return {
        "eigenvalues": eigenvalues,
        "lowest": float(eigenvalues[0]),
        "n_negative": int((eigenvalues < -1e-8).sum()),
        "is_minimum": bool(eigenvalues[0] > -1e-8),
        "gradient_norm": float(
            np.linalg.norm(orbital_gradient(h1e, eri, mo_occ))
        ),
        "lowest_mode": eigenvectors[:, 0],
        "converged": converged,
        "hessian_vector_products": products,
    }
