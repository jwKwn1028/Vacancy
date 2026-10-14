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


def occupation_blocks(mo_occ):
    """Index arrays of the doubly occupied, singly occupied and empty orbitals."""
    occ = np.asarray(mo_occ)
    return [np.flatnonzero(block) for block in (occ == 2, (occ > 0) & (occ < 2), occ == 0)]


def canonical_block_rotation(fock, mo_occ):
    """Rotation that diagonalizes ``fock`` inside each occupation block.
    Returns the block-diagonal rotation, the canonical orbital energies in the same
    orbital positions, and the largest off-diagonal element inside a block before it.
    """
    fock = np.asarray(fock)
    n = fock.shape[0]
    rotation = np.zeros((n, n))
    energies = np.zeros(n)
    off_diagonal = 0.0
    for idx in occupation_blocks(mo_occ):
        if idx.size == 0:
            continue
        sub = fock[np.ix_(idx, idx)]
        if idx.size > 1:
            off_diagonal = max(
                off_diagonal, float(np.abs(sub - np.diag(np.diag(sub))).max())
            )
        values, vectors = np.linalg.eigh(sub)
        rotation[np.ix_(idx, idx)] = vectors
        energies[idx] = values
    return rotation, energies, off_diagonal


def averaged_fock(h1e, eri, mo_occ):
    """h + (v_a + v_b)/2 for the occupation pattern ``mo_occ`` in the current frame."""
    dm_a, dm_b = so.densities(mo_occ)
    v_a, v_b = so.general_veff(eri, dm_a, dm_b)
    return h1e + 0.5 * (v_a + v_b)


def canonical_rotation(h1e, eri, mo_occ):
    """``canonical_block_rotation`` of the Fock built from ``h1e`` and ``eri``."""
    return canonical_block_rotation(averaged_fock(h1e, eri, mo_occ), mo_occ)
