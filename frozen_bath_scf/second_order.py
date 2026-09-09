import numpy as np
from scipy.linalg import expm

from pyscf import lib


def general_veff(eri, dm_a, dm_b):
    j = lib.einsum("ijkl,lk->ij", eri, dm_a + dm_b)
    k_a = lib.einsum("ikjl,lk->ij", eri, dm_a)
    k_b = lib.einsum("ikjl,lk->ij", eri, dm_b)
    return [j - k_a, j - k_b]


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


def stability_analysis(h1e, eri, mo_occ, step=1e-4):
    hess = orbital_hessian(h1e, eri, mo_occ, step)
    eigenvalues, eigenvectors = np.linalg.eigh(hess)
    return {
        "eigenvalues": eigenvalues,
        "lowest": float(eigenvalues[0]),
        "n_negative": int((eigenvalues < -1e-8).sum()),
        "is_minimum": bool((eigenvalues > -1e-8).all()),
        "gradient_norm": float(
            np.linalg.norm(orbital_gradient(h1e, eri, mo_occ))
        ),
        "lowest_mode": eigenvectors[:, 0],
    }
