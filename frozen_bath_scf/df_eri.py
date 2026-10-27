"""Active-space two-electron integrals as density-fitting factors.

(ij|kl) = sum_P s_P L_Pij L_Pkl, with L real and symmetric in ij and s_P = +-1.  The
cluster's Cholesky vectors give exactly the ERIs the embedding used to store densely,
in naux x n^2 numbers instead of n^4.  The functions here take either form, so a dense
(n, n, n, n) array from an old checkpoint or a test still works.
"""
import numpy as np


class DFEri:

    def __init__(self, factors, sign=None):
        self.factors = np.ascontiguousarray(np.asarray(factors, dtype=float))
        if self.factors.ndim != 3 or self.factors.shape[1] != self.factors.shape[2]:
            raise ValueError("DF factors must have shape (naux, n, n), got %s"
                             % (self.factors.shape,))
        self.sign = (np.ones(len(self.factors)) if sign is None
                     else np.asarray(sign, dtype=float).copy())

    @property
    def n(self):
        return self.factors.shape[1]

    @property
    def naux(self):
        return self.factors.shape[0]

    @property
    def nbytes(self):
        return self.factors.nbytes

    def __array__(self, *args, **kwargs):
        raise TypeError("a DFEri is not converted implicitly (n^4 numbers); call "
                        "dense_eri() where the dense tensor is really wanted")

    def _blocks(self, row_bytes, block_mb=256.0):
        step = max(1, int(block_mb * 1e6 / max(row_bytes, 1)))
        for p0 in range(0, self.naux, step):
            yield p0, min(p0 + step, self.naux)

    def rotate(self, u):
        u = np.asarray(u, dtype=float)
        out = np.empty((self.naux, u.shape[1], u.shape[1]))
        for p0, p1 in self._blocks(8.0 * self.n * self.n * 2):
            out[p0:p1] = np.matmul(u.T, np.matmul(self.factors[p0:p1], u))
        return DFEri(out, self.sign)

    def take(self, idx, signs=None):
        idx = np.asarray(idx)
        out = self.factors[:, idx][:, :, idx]
        if signs is not None:
            s = np.asarray(signs, dtype=float)
            out = out * (s[:, None] * s[None, :])
        return DFEri(out, self.sign)

    def dense(self):
        return np.einsum("P,Pij,Pkl->ijkl", self.sign, self.factors, self.factors,
                         optimize=True)

    def j(self, dm):
        rho = np.einsum("Pkl,lk->P", self.factors, np.asarray(dm)) * self.sign
        return np.tensordot(rho, self.factors, axes=(0, 0))

    def k(self, dm):
        """K_ij = sum_kl (ik|jl) dm_lk for a symmetric dm, through its nonzero
        eigenvectors: sum_P s_P (L_P V) w (L_P V)^T."""
        dm = np.asarray(dm, dtype=float)
        if not np.count_nonzero(dm - np.diag(np.diag(dm))):
            w = np.diag(dm).copy()
            keep = np.flatnonzero(w)
            v = None
        else:
            w, v = np.linalg.eigh(0.5 * (dm + dm.T))
            keep = np.flatnonzero(np.abs(w) > 1e-14 * max(1.0, np.abs(w).max()))
            v = v[:, keep]
        out = np.zeros((self.n, self.n))
        if keep.size == 0:
            return out
        w = w[keep]
        for p0, p1 in self._blocks(8.0 * self.n * keep.size * 2):
            x = (self.factors[p0:p1][:, :, keep] if v is None
                 else np.matmul(self.factors[p0:p1], v))          # (nb, n, r)
            xw = x * (w[None, None, :] * self.sign[p0:p1, None, None])
            n_b = p1 - p0
            out += (xw.transpose(1, 0, 2).reshape(self.n, n_b * keep.size)
                    @ x.transpose(1, 0, 2).reshape(self.n, n_b * keep.size).T)
        return out


def as_eri(eri):
    """``eri`` as a DFEri or a real dense array (None stays None)."""
    if eri is None or isinstance(eri, DFEri):
        return eri
    return np.asarray(eri).real


def dense_eri(eri):
    return eri.dense() if isinstance(eri, DFEri) else np.asarray(eri).real


def eri_j(eri, dm):
    if isinstance(eri, DFEri):
        return eri.j(dm)
    from pyscf import lib

    return lib.einsum("ijkl,lk->ij", eri, dm)


def eri_k(eri, dm):
    if isinstance(eri, DFEri):
        return eri.k(dm)
    from pyscf import lib

    return lib.einsum("ikjl,lk->ij", eri, dm)


def rotate_eri(eri, u):
    if isinstance(eri, DFEri):
        return eri.rotate(u)
    from pyscf import lib

    g = lib.einsum("pqrs,pi->iqrs", eri, u)
    g = lib.einsum("iqrs,qj->ijrs", g, u)
    g = lib.einsum("ijrs,rk->ijks", g, u)
    return lib.einsum("ijks,sl->ijkl", g, u)


def take_eri(eri, idx, signs=None):
    if isinstance(eri, DFEri):
        return eri.take(idx, signs)
    out = eri[np.ix_(idx, idx, idx, idx)]
    if signs is not None:
        s = np.asarray(signs, dtype=float)
        out = out * (s[:, None, None, None] * s[None, :, None, None]
                     * s[None, None, :, None] * s[None, None, None, :])
    return out
