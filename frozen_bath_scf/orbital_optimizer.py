import numpy as np
from scipy.optimize import minimize

from . import second_order as so
from .df_eri import rotate_eri

energy_and_gradient = so.energy_and_gradient


def energy_at(h1e, eri, mo_occ, kappa=None):
    dm_a, dm_b = so.densities(mo_occ)
    if kappa is not None:
        u = so._expm_antisym(kappa)
        dm_a, dm_b = u @ dm_a @ u.T, u @ dm_b @ u.T
    return so.energy_from_dm(h1e, dm_a, dm_b, eri)


def minimize_orbitals(h1e, eri, mo_occ, x0, maxiter=2000, gtol=1e-9):
    mask = so.rotation_mask(mo_occ)

    def fun(x):
        energy, g = energy_and_gradient(h1e, eri, mo_occ, so._kappa_from_vector(x, mask))
        return energy, g[mask] - g.T[mask]

    result = minimize(
        fun, np.asarray(x0, dtype=float), jac=True, method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 0.0, "gtol": gtol},
    )
    return result, so._kappa_from_vector(result.x, mask)


def rotate_hamiltonian(h1e, eri, u):
    return u.T @ h1e @ u, rotate_eri(eri, u)


def _conjugate_gradient(hvp, b, precond, rtol, maxiter):
    """Preconditioned CG for H x = b.  Stops at negative curvature (returns the iterate
    so far and False).  Returns (x, positive, products)."""
    x = np.zeros_like(b)
    r = b.copy()
    z = precond * r
    p = z.copy()
    rz = float(r @ z)
    target = rtol * float(np.linalg.norm(b))
    for it in range(maxiter):
        hp = hvp(p)
        curvature = float(p @ hp)
        if curvature <= 0.0:
            return x, False, it + 1
        alpha = rz / curvature
        x = x + alpha * p
        r = r - alpha * hp
        if np.linalg.norm(r) <= target:
            return x, True, it + 1
        z = precond * r
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, True, maxiter


def newton_finish(h1e, eri, mo_occ, tol, max_cycle=8, cg_rtol=1e-3, cg_maxiter=300):
    """Newton steps at fixed occupations until the orbital gradient norm is at most ``tol``.

    Each step is taken in the current orbitals.  It finishes what minimize_orbitals
    cannot: near a minimum its energy-based line search stops where the energy change,
    about |g|^2 / H, reaches rounding.  The Newton equations are solved by preconditioned
    conjugate gradients on Hessian-vector products (so.hessian_vector), with no Hessian
    matrix.  Before the first step the lowest Hessian eigenvalue is found iteratively
    (so.lowest_hessian_modes): no step is taken unless it is positive, nor past negative
    curvature met by CG; a step is halved while it raises the energy.  Returns the
    rotation, the rotated (h1e, eri) and a record; a record that stops at a
    non-positive lowest eigenvalue carries its mode as ``lowest_mode`` (an array).
    """
    mask = so.rotation_mask(mo_occ)
    u = np.eye(len(mo_occ))
    g = so.orbital_gradient(h1e, eri, mo_occ)
    e = energy_at(h1e, eri, mo_occ)
    info = {"newton_cycles": 0, "newton_gradient_before": float(np.linalg.norm(g)),
            "newton_stop": "converged", "newton_hessian_vector_products": 0}
    while np.linalg.norm(g) > tol:
        if info["newton_cycles"] == max_cycle:
            info["newton_stop"] = "max_cycle"
            break
        if info["newton_cycles"] == 0:
            lowest, modes, _, products = so.lowest_hessian_modes(h1e, eri, mo_occ)
            info["newton_lowest_hessian"] = float(lowest[0])
            info["newton_hessian_vector_products"] += products
            if lowest[0] <= 0.0:
                info["newton_stop"] = "hessian_not_positive"
                info["lowest_mode"] = modes[:, 0]
                break
        diag = np.abs(so.hessian_diagonal_estimate(h1e, eri, mo_occ))
        precond = 1.0 / np.maximum(diag, 1e-3 * max(1.0, diag.max()))
        x, positive, products = _conjugate_gradient(
            lambda v: so.hessian_vector(h1e, eri, mo_occ, v, mask=mask), -g, precond,
            cg_rtol, cg_maxiter)
        info["newton_hessian_vector_products"] += products
        if not positive:
            info["newton_stop"] = "hessian_not_positive"
            break
        for _ in range(6):
            step = so._expm_antisym(so._kappa_from_vector(x, mask))
            h_new, eri_new = rotate_hamiltonian(h1e, eri, step)
            e_new = energy_at(h_new, eri_new, mo_occ)
            if e_new <= e + 1e-12 * max(1.0, abs(e)):
                break
            x = 0.5 * x
        else:
            info["newton_stop"] = "no_energy_decrease"
            break
        h1e, eri, e = h_new, eri_new, e_new
        u = u @ step
        g = so.orbital_gradient(h1e, eri, mo_occ)
        info["newton_cycles"] += 1
    info["newton_gradient_after"] = float(np.linalg.norm(g))
    return u, h1e, eri, info


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
