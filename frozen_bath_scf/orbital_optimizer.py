import numpy as np
from scipy.optimize import minimize

from pyscf import lib

from . import second_order as so


def energy_at(h1e, eri, mo_occ, kappa=None):
    dm_a, dm_b = so.densities(mo_occ)
    if kappa is not None:
        u = so._expm_antisym(kappa)
        dm_a, dm_b = u @ dm_a @ u.T, u @ dm_b @ u.T
    return so.energy_from_dm(h1e, dm_a, dm_b, eri)


def minimize_orbitals(h1e, eri, mo_occ, x0, maxiter=2000, gtol=1e-9):
    mask = so.rotation_mask(mo_occ)

    def fun(x):
        kappa = so._kappa_from_vector(x, mask)
        return (
            float(energy_at(h1e, eri, mo_occ, kappa)),
            np.asarray(so.orbital_gradient(h1e, eri, mo_occ, kappa), dtype=float),
        )

    result = minimize(
        fun, np.asarray(x0, dtype=float), jac=True, method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 0.0, "gtol": gtol},
    )
    return result, so._kappa_from_vector(result.x, mask)


def rotate_hamiltonian(h1e, eri, u):
    h = u.T @ h1e @ u
    g = lib.einsum("pqrs,pi->iqrs", eri, u)
    g = lib.einsum("iqrs,qj->ijrs", g, u)
    g = lib.einsum("ijrs,rk->ijks", g, u)
    g = lib.einsum("ijks,sl->ijkl", g, u)
    return h, g
