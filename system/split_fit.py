"""ERIs from the ideal crystal's k-point GDF when all four AOs lie outside the
fragment, from the Gamma GDF otherwise."""
import numpy as np


class KpointOutside:
    """The k-point GDF applied to a supercell density, in k-space.

    J~(k1,k2) = 1/N sum_P L(k1,k2)_P g(k2-k1)_P,  g(q)_P = sum_k L(k,k-q)_P . D~(-k,q-k)
    K~(k1,-k3) = 1/N sum_k2,P L(k1,k2)_P D~(k2,-k4) L(k3,k4)_P^T,  k4 = k1-k2+k3
    """

    def __init__(self, kmf, kmesh):
        from pyscf.pbc.tools.k2gamma import (to_supercell_ao_integrals,
                                             translation_vectors_for_kmesh)

        cell, kpts = kmf.cell, np.asarray(kmf.kpts)
        self.nk, self.nao = len(kpts), cell.nao_nr()
        self.phase = np.exp(1j * translation_vectors_for_kmesh(cell, list(kmesh)) @ kpts.T)
        frac = cell.get_scaled_kpts(kpts)

        def index(f):
            d = frac - f
            hit = np.where(np.abs(d - np.round(d)).max(axis=1) < 1e-8)[0]
            if len(hit) != 1:
                raise RuntimeError("k-point %s is not on the mesh" % f)
            return int(hit[0])

        nk, n = self.nk, self.nao
        self.ksum = np.array([[index(frac[a] + frac[b]) for b in range(nk)]
                              for a in range(nk)])
        self.minus = np.array([index(-frac[a]) for a in range(nk)])
        # lq[q][k] = L^(k, k+q)
        self.lq = []
        for q in range(nk):
            pairs = []
            for k in range(nk):
                blocks = []
                for lr, li, sign in kmf.with_df.sr_loop(
                        np.array([kpts[k], kpts[self.ksum[k, q]]]), compact=False):
                    if sign != 1:
                        raise NotImplementedError("GDF with a negative metric part")
                    blocks.append(np.asarray(lr) + 1j * np.asarray(li))
                pairs.append(np.concatenate(blocks).reshape(-1, n, n))
            self.lq.append(np.asarray(pairs))
        for q, mq in enumerate(self.minus):
            if self.lq[q].shape != self.lq[mq].shape:
                raise RuntimeError("q and -q have different auxiliary dimensions")
        self.density = np.asarray(to_supercell_ao_integrals(
            cell, kpts, np.asarray(kmf.make_rdm1()), kmesh=list(kmesh))).real

    def _forward(self, m):
        nk, n = self.nk, self.nao
        t = np.tensordot(self.phase.conj(), m.reshape(nk, n, nk, n), axes=(0, 0))
        return np.tensordot(t, self.phase, axes=(2, 0)).transpose(0, 1, 3, 2)

    def _backward(self, mt):
        nk, n = self.nk, self.nao
        t = np.tensordot(self.phase, mt, axes=(1, 0))
        m = np.tensordot(t, self.phase.conj(), axes=(2, 1)).transpose(0, 1, 3, 2)
        return m.reshape(nk * n, nk * n) / nk ** 2

    def get_jk(self, dm):
        nk, n = self.nk, self.nao
        ks = np.arange(nk)
        dt = self._forward(np.asarray(dm, dtype=float))
        jt = np.zeros_like(dt)
        kt = np.zeros_like(dt)
        for q in range(nk):
            km = self.ksum[ks, self.minus[q]]
            g = np.einsum("kPmn,kmn->P", self.lq[self.minus[q]],
                          dt[self.minus[ks], :, self.minus[km], :])
            jt[ks, :, self.ksum[ks, q], :] += np.einsum("kPmn,P->kmn", self.lq[q], g) / nk
        for p in range(nk):
            a = self.lq[p]
            c = np.ascontiguousarray(self.lq[self.minus[p]].transpose(0, 1, 3, 2))
            naux = a.shape[1]
            k4 = self.ksum[ks, self.minus[p]]
            for k1 in range(nk):
                dk = dt[self.ksum[k1, p]][:, self.minus[k4], :].transpose(1, 0, 2)
                u = np.matmul(dk[:, None], c).reshape(nk, naux * n, n)
                a2 = a[k1].transpose(1, 0, 2).reshape(n, naux * n)
                kt[k1][:, self.minus[ks], :] += np.matmul(a2, u).transpose(1, 0, 2) / nk
        return self._backward(jt).real, self._backward(kt).real


def fragment_aos(cell, atoms):
    mask = np.zeros(cell.nao_nr(), dtype=bool)
    for i in atoms:
        p0, p1 = cell.aoslice_by_atom()[i][2:]
        mask[p0:p1] = True
    return mask


class SplitDensityFit:
    """The Gamma GDF with its outside-outside ERIs from ``outside``."""

    def __init__(self, inside, outside, inside_aos):
        self.inside = inside
        self.outside = outside
        out = ~np.asarray(inside_aos, dtype=bool)
        self.oo = np.outer(out, out)

    def __getattr__(self, name):
        if name in ("inside", "outside", "oo"):
            raise AttributeError(name)
        return getattr(self.inside, name)

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None, with_j=True,
               with_k=True, omega=None, exxdiv=None):
        if omega is not None or kpts_band is not None or exxdiv is not None:
            raise NotImplementedError("--split: Gamma, exxdiv=None only")
        dm = np.asarray(dm)
        d_oo = np.where(self.oo, dm, 0.0)
        vj, vk = self.inside.get_jk(dm, hermi, kpts, None, with_j, with_k)
        vj_oo, vk_oo = self.inside.get_jk(d_oo, hermi, kpts, None, with_j, with_k)
        n = self.oo.shape[0]
        outs = [self.outside.get_jk(d.real) for d in d_oo.reshape(-1, n, n)]
        if with_j:
            jo = np.asarray([o[0] for o in outs]).reshape(dm.shape)
            vj = np.asarray(vj) + np.where(self.oo, jo - np.asarray(vj_oo), 0.0)
        if with_k:
            ko = np.asarray([o[1] for o in outs]).reshape(dm.shape)
            vk = np.asarray(vk) + np.where(self.oo, ko - np.asarray(vk_oo), 0.0)
        return vj, vk

    def sr_loop(self, *args, **kwargs):
        raise NotImplementedError("--split supports the cluster active ERIs only")
