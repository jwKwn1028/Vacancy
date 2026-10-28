"""Periodic all-electron derivative integrals from PySCF's own builders.

d/dA_x of x^a y^b z^c exp(-alpha r^2) centred on A is 2 alpha x^(a+1).. - a x^(a-1)..: a
derivative is the same integral over shells of ang +- 1 ("derivative shells"), recombined.
"""
import time
from functools import cached_property

import numpy as np
from pyscf import gto
from pyscf.gto.mole import ANG_OF, ATOM_OF, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_EXP


DERIVATIVE_PRECISION = 1e-12
# Tighter for the nucleus field: cheap (nuclei only), and steep core shells need it.
FIELD_PRECISION = 1e-14


def _derivative_integral_cell(cell, precision=None):
    """``cell`` with tighter screening, for integrals that are recombined into derivatives.
    """
    out = cell.copy(deep=False)
    out.precision = min(cell.precision, precision or DERIVATIVE_PRECISION)
    return out


def _fac(ang):
    return {0: 0.282094791773878143, 1: 0.488602511902919921}.get(ang, 1.0)


def cart_components(ang):
    return [(lx, ly, ang - lx - ly) for lx in range(ang, -1, -1) for ly in range(ang - lx, -1, -1)]


def derivative_cell(cell, shells):
    """A Cartesian copy of ``cell``: the derivative shells of ``shells``, then all original
    shells.  Returns (dcell, nd, rec), d chi_mu/dA_x = sum_d rec[x][d, mu] phi_d."""
    bas, env = [], list(cell._env)
    pieces = []          # (ang', nctr) of each derivative shell
    norm = {}            # (shell, ang'): the factor its stored coefficients were divided by
    for i in shells:
        b = cell._bas[i]
        ang, nprim, nctr = int(b[ANG_OF]), int(b[NPRIM_OF]), int(b[NCTR_OF])
        exps = cell._env[b[PTR_EXP]:b[PTR_EXP] + nprim]
        coef = cell._env[b[PTR_COEFF]:b[PTR_COEFF] + nprim * nctr].reshape(nctr, nprim)
        for lp, scale in ((ang + 1, 2 * exps * _fac(ang) / _fac(ang + 1)),
                          (ang - 1, _fac(ang) / _fac(ang - 1) if ang > 0 else None)):
            if lp < 0:
                continue
            c = coef * scale
            # PySCF screens by the stored coefficients: keep them, scale the result.
            norm[i, lp] = abs(c).max() / abs(coef).max()
            nb = b.copy()
            nb[ANG_OF] = lp
            nb[PTR_COEFF] = len(env)
            env.extend((c / norm[i, lp]).ravel())
            bas.append(nb)
            pieces.append((lp, nctr))
    nd = len(bas)
    dcell = cell.copy(deep=False)
    dcell.cart = True
    dcell._bas = np.asarray(np.vstack([np.asarray(bas, dtype=np.int32).reshape(-1, 8),
                                       cell._bas]), dtype=np.int32)
    dcell._env = np.asarray(env, dtype=float)

    # Recombination: rows = Cartesian derivative functions, cols = original functions.
    ndf = sum((lp + 1) * (lp + 2) // 2 * nctr for lp, nctr in pieces)
    orig_loc = cell.ao_loc_nr()
    ncol = int(sum(orig_loc[i + 1] - orig_loc[i] for i in shells))
    rec = np.zeros((3, ndf, ncol))
    row = 0
    col = 0
    for i in shells:
        b = cell._bas[i]
        ang, nctr = int(b[ANG_OF]), int(b[NCTR_OF])
        comps = cart_components(ang)
        c2s = np.eye(len(comps)) if cell.cart else gto.cart2sph(ang, normalized="sp")
        nsph = c2s.shape[1]
        blocks = {}
        for lp in (ang + 1, ang - 1):
            if lp < 0:
                continue
            blocks[lp] = row
            row += (lp + 1) * (lp + 2) // 2 * nctr
        for j in range(nctr):
            for x in range(3):
                for ci, comp in enumerate(comps):
                    up = list(comp)
                    up[x] += 1
                    r_up = blocks[ang + 1] + j * ((ang + 2) * (ang + 3) // 2) + \
                        cart_components(ang + 1).index(tuple(up))
                    rec[x, r_up, col + j * nsph:col + (j + 1) * nsph] += \
                        norm[i, ang + 1] * c2s[ci]
                    if comp[x] > 0:
                        dn = list(comp)
                        dn[x] -= 1
                        r_dn = blocks[ang - 1] + j * (ang * (ang + 1) // 2) + \
                            cart_components(ang - 1).index(tuple(dn))
                        rec[x, r_dn, col + j * nsph:col + (j + 1) * nsph] -= \
                            norm[i, ang - 1] * comp[x] * c2s[ci]
        col += nsph * nctr
    return dcell, nd, rec


def atom_shells(cell, atoms):
    return [i for i in range(cell.nbas) if int(cell._bas[i, ATOM_OF]) in set(atoms)]


def cart_to_original(cell):
    """(n_cart, n_orig) map from the Cartesian functions of ``cell`` to its own."""
    if cell.cart:
        return np.eye(cell.nao_nr())
    return cell.cart2sph_coeff(normalized="sp")


def _cart(cell):
    c = cell.copy(deep=False)
    c.cart = True
    return c


def _row_pass(pcell, auxcell, settings, nsh_rows, col_shells):
    """Raw Gamma (P|ab) on the first ``nsh_rows`` shells and the columns ``col_shells``,
    less the G = 0 term."""
    from system.fragment_fit import RowBuilder, h5_steps

    builder = RowBuilder(pcell, auxcell, settings)
    loc = pcell.ao_loc_nr()
    nao = int(loc[-1])
    c0, c1 = int(loc[col_shells[0]]), int(loc[col_shells[1]])
    out, row = [], 0
    vbar = s = None
    for v in h5_steps(builder, aosym="s1",
                       shls_slice=(0, nsh_rows, 0, pcell.nbas, 0, auxcell.nbas)):
        if row == 0:
            vbar, s = builder.g0_term()
        v = v.reshape(len(v), -1, nao)[:, :, c0:c1]
        nr = v.shape[1]
        if vbar is not None:
            v -= vbar[:, None, None] * s[row:row + nr, c0:c1]
        out.append(v)
        row += nr
    return np.concatenate(out, axis=1)


def raw_rows(cell, auxcell, atoms, settings):
    """(P|ab) for a on ``atoms`` (sorted), every b: (naux, nrow, nao)."""
    from system.fragment_fit import with_atoms

    atoms = sorted(int(i) for i in atoms)
    order = atoms + sorted(set(range(cell.natm)) - set(atoms))
    sl = cell.aoslice_by_atom()
    perm = np.concatenate([np.arange(sl[i][2], sl[i][3]) for i in order])
    pcell = with_atoms(cell, [cell._atom[i] for i in order])
    nsh = sum(int(sl[i][1] - sl[i][0]) for i in atoms)
    v = _row_pass(pcell, auxcell, settings, nsh, (0, pcell.nbas))
    return v[:, :, np.argsort(perm)]


def derivative_rows(cell, auxcell, atoms, settings, precision=None):
    """d/dA_x (P|ab) for a on ``atoms``: (3, naux, nrow, nao)."""
    cell = _derivative_integral_cell(cell, precision)
    auxcell = _derivative_integral_cell(auxcell, precision)
    shells = atom_shells(cell, atoms)
    dcell, nd, rec = derivative_cell(cell, shells)
    aux = _cart(auxcell)
    v = _row_pass(dcell, aux, settings, nd, (nd, dcell.nbas))
    v = np.tensordot(cart_to_original(auxcell), v, axes=(0, 0))
    v = v @ cart_to_original(cell)
    return np.einsum("xdm,Pdn->xPmn", rec, v, optimize=True)


def derivative_aux_cell(auxcell, atoms):
    """The derivative shells of the auxiliary functions on ``atoms``, and their
    recombination rec[x]."""
    shells = atom_shells(auxcell, atoms)
    dcell, nd, rec = derivative_cell(auxcell, shells)
    dcell._bas = dcell._bas[:nd].copy()
    return dcell, rec


def _pair_pass(cell, auxcell, settings, intor="int3c2e"):
    """Raw Gamma (P|ab) over every packed pair: (naux, npair)."""
    from system.fragment_fit import GammaBuilder, h5_steps

    builder = GammaBuilder(cell, auxcell, settings)
    if intor not in ("int3c2e", "int3c2e_sph", "int3c2e_cart"):
        builder.exclude_dd_block = False
    return np.hstack(list(h5_steps(builder, aosym="s2", intor=intor)))


def pairs_to_original(cell, v, batch=64):
    """Packed pairs of the Cartesian functions of ``cell`` -> of its own functions."""
    from pyscf import lib

    if cell.cart:
        return v
    c2s = cart_to_original(cell)
    out = np.empty((len(v), c2s.shape[1] * (c2s.shape[1] + 1) // 2))
    for i in range(0, len(v), batch):
        m = lib.unpack_tril(v[i:i + batch])
        out[i:i + batch] = lib.pack_tril(np.einsum("pi,xpq,qj->xij", c2s, m, c2s,
                                                   optimize=True))
    return out


def derivative_aux_pairs(cell, auxcell, atoms, settings):
    """d/dA_x (P|ab) for the auxiliary functions on ``atoms``: (3, n_aux_on_atoms, npair).
    """
    cell = _derivative_integral_cell(cell)
    auxcell = _derivative_integral_cell(auxcell)
    daux, rec = derivative_aux_cell(auxcell, atoms)
    v = pairs_to_original(cell, _pair_pass(_cart(cell), daux, settings))
    return np.einsum("xdP,dk->xPk", rec, v, optimize=True)


def derivative_metric_rows(cell, auxcell, atoms, others, settings):
    """d/dA_x (P|Q) for P of ``auxcell`` on ``atoms`` and Q of ``others``: (3,
    n_aux_on_atoms, n_others)."""
    from pyscf.pbc.gto.cell import conc_cell
    from system.fragment_fit import gamma_metric

    cell = _derivative_integral_cell(cell)
    auxcell = _derivative_integral_cell(auxcell)
    others = _derivative_integral_cell(others)
    daux, rec = derivative_aux_cell(auxcell, atoms)
    nd = daux.nao_nr()
    j2c = gamma_metric(cell, conc_cell(daux, _cart(others)), settings)
    block = j2c[:nd, nd:] @ cart_to_original(others)
    return np.einsum("xdP,dQ->xPQ", rec, block, optimize=True)


# --------------------------------------------------------------------------- the fragment fit

def rows_to_coords(kfit, v):
    """Raw rows (nk*naux, nrow, nao) to the k-point fit's real coordinates, as
    fragment_vectors does."""
    from pyscf.pbc.df import rsdf_builder

    nk = kfit.nk
    naux = v.shape[0] // nk
    shape = v.shape[1:]
    ft = kfit.phase[:, kfit.half].conj().T / np.sqrt(nk)
    vt = np.tensordot(ft, v.reshape(nk, -1), axes=(1, 0))
    out = {}
    for i, p in enumerate(kfit.half):
        x = vt[i].reshape(naux, -1).T
        imag = None if kfit.minus[p] == p else np.ascontiguousarray(x.imag)
        e = rsdf_builder._RSGDFBuilder.solve_cderi(None, kfit.metric[p],
                                                   np.ascontiguousarray(x.real), imag)[0]
        out[p] = e.reshape((-1,) + shape)
    return kfit.to_real(out)


def _packed_rows(rows, ao, nao):
    """Packed symmetric matrices from rows on ``ao``, mirrored, the ao-ao block summed both
    ways."""
    from pyscf import lib

    lead = rows.shape[:-2]
    m = np.zeros(lead + (nao, nao))
    m[..., ao, :] += rows
    m[..., :, ao] += np.swapaxes(rows, -1, -2)
    return lib.pack_tril(m.reshape(-1, nao, nao)).reshape(lead + (-1,))


class FitDerivative:
    """d/dR_A of a FragmentFit's ERIs and nuclear attraction, contracted with fixed
    matrices, for A among its moved atoms."""

    def __init__(self, fit):
        from pyscf.pbc.df import df as pbc_df
        from system.fragment_fit import gamma_metric, with_atoms

        self.fit = fit
        self.timings = {"vector_derivatives": 0.0, "bilinear": 0.0, "bilinear_calls": 0,
                        "adjoints": 0.0, "reverse": 0.0, "rows": 0.0, "to_coords": 0.0,
                        "aux_pairs": 0.0, "metric_rows": 0.0}
        kfit = self.kfit = fit.kfit
        cell = self.cell = getattr(fit, "row_cell", fit.cell)
        ideal = self.ideal = kfit.ideal_supercell(cell)
        self.auxcell = pbc_df.make_auxcell(ideal, kfit.auxbasis)
        self.settings = fit.settings
        moved = self.moved = list(fit.moved)
        new = [cell._atom[i] for i in moved]
        self.auxnew = pbc_df.make_auxcell(with_atoms(cell, new), kfit.auxbasis)
        if fit.low is None:
            self.nx = 0
            return
        sl = self.auxcell.aoslice_by_atom()
        self.xf = np.concatenate([np.arange(sl[i][2], sl[i][3]) for i in moved])
        nx = self.nx = len(self.xf)
        j2c = gamma_metric(cell, pbc_df.make_auxcell(
            with_atoms(cell, new + list(ideal._atom)), kfit.auxbasis), self.settings)
        dj = j2c[:nx, nx:] - j2c[nx + self.xf, nx:]
        djff = j2c[:nx, :nx] - j2c[np.ix_(nx + self.xf, nx + self.xf)]
        del j2c
        ex = np.zeros((nx, kfit.nk * kfit.naux))
        ex[np.arange(nx), self.xf] = 1.0
        self.q = np.linalg.qr(kfit.coords(ex))[0]
        self.a = kfit.aux_coords(self.xf)
        k_new = self.a + kfit.coords(dj)
        self.o = k_new - self.q @ (self.q.T @ k_new)
        s_new = self.a.T @ self.a + djff - self.o.T @ self.o
        # The fit's own eigensystem: its low-rank vectors are Lambda^-1/2 V^T W'.
        self.lam, self.vec = fit.aux_eig
        if abs(self.vec @ np.diag(self.lam) @ self.vec.T - s_new).max() > 1e-8 * abs(s_new).max():
            raise RuntimeError("the moving-auxiliary metric differs from the fit's")
        self.T = fit.low[nx:]

    # E with every column, and the low-rank vectors before Lambda^-1/2: the +- path only.
    @cached_property
    def Efull(self):
        fit = self.fit
        e = np.zeros((self.kfit.nreal, len(fit.fi), self.cell.nao_nr()))
        e[:, :, fit.ci] = fit.E
        return e

    @cached_property
    def W(self):
        return self.vec @ (np.sqrt(self.lam)[:, None] * self.fit.low[:self.nx])

    @cached_property
    def U(self):
        return self.vec @ self.fit.low[:self.nx]

    def _new_slice(self, atom):
        k = self.moved.index(atom)
        s = self.auxnew.aoslice_by_atom()[k]
        return k, int(s[2]), int(s[3])

    def atom_integrals(self, atom, axes=(0, 1, 2)):
        """The derivative integrals one atom's motion needs; ex and dpairs on ``axes`` only."""
        from pyscf.pbc.df import aft
        from pyscf.pbc.gto.cell import conc_cell

        cell, kfit, tm = self.cell, self.kfit, self.timings
        aux = conc_cell(conc_cell(self.auxcell, self.auxnew), aft._fake_nuc(cell, with_pseudo=False))
        t = time.perf_counter()
        rows = derivative_rows(cell, aux, [atom], self.settings)   # (3, naux_all, nrow, nao)
        tm["rows"] += time.perf_counter() - t
        nsc = kfit.nk * kfit.naux
        nnew = self.auxnew.nao_nr()
        t = time.perf_counter()
        ex = {x: rows_to_coords(kfit, rows[x, :nsc]) for x in axes}
        tm["to_coords"] += time.perf_counter() - t
        vnuc = -np.tensordot(cell.atom_charges(), rows[:, nsc + nnew:], axes=(0, 1))
        if self.nx == 0:
            return dict(ex=ex, vnuc=vnuc)
        dvx = rows[:, nsc:nsc + self.nx] - rows[:, self.xf]
        del rows
        k, p0, p1 = self._new_slice(atom)
        t = time.perf_counter()
        dpairs = derivative_aux_pairs(cell, self.auxnew, [k], self.settings)
        dpairs = {x: dpairs[x].copy() for x in axes}
        tm["aux_pairs"] += time.perf_counter() - t
        t = time.perf_counter()
        others = conc_cell(self.auxnew, self.auxcell)
        dmet = derivative_metric_rows(cell, self.auxnew, [k], others, self.settings)
        tm["metric_rows"] += time.perf_counter() - t
        return dict(ex=ex, dvx=dvx, vnuc=vnuc, dpairs=dpairs, dmet=dmet, slice=(p0, p1))

    def vector_derivatives(self, atom, ints, x):
        """dE (rows of E, all columns) and d(low) for d/dA_x."""
        from frozen_bath_scf import second_order as so  # noqa: F401  (keeps imports local)

        fit, kfit = self.fit, self.kfit
        nao = self.cell.nao_nr()
        ao = np.flatnonzero(_atom_mask(self.cell, atom))
        fi = fit.fi
        ex = ints["ex"][x]                              # (nreal, nrow_A, nao)
        dE = np.zeros_like(self.Efull)
        pos = {int(f): i for i, f in enumerate(fi)}
        for r, mu in enumerate(ao):
            dE[:, pos[int(mu)], :] += ex[:, r, :]
        for r, mu in enumerate(ao):
            dE[:, :, mu] += ex[:, r, fi]
        if self.nx == 0:
            return dE, None, None
        # Low-rank: the AO part (rows of A) ...
        c_ao = self.a - self.o
        pot_rows = np.einsum("vy,vmn->ymn", c_ao, ex, optimize=True)
        dw_ao = _packed_rows(ints["dvx"][x] + pot_rows, ao, nao)
        dt_ao = _packed_rows(np.einsum("vy,vmn->ymn", self.q, ex, optimize=True), ao, nao)
        # ... and the auxiliary part (A's functions in F' move).
        p0, p1 = ints["slice"]
        nx = self.nx
        dj = np.zeros((nx, kfit.nk * kfit.naux))
        dj[p0:p1] = ints["dmet"][x][:, nx:]
        djff = np.zeros((nx, nx))
        djff[p0:p1] = ints["dmet"][x][:, :nx]
        djff = djff + djff.T
        do = kfit.coords(dj)
        do = do - self.q @ (self.q.T @ do)
        dS = djff - (do.T @ self.o + self.o.T @ do)
        dw_aux = -fit.dense_potential(do, np.empty((nx, nao * (nao + 1) // 2)))
        dw_aux[p0:p1] += ints["dpairs"][x]
        lam, vec = self.lam, self.vec
        inv = (vec / np.sqrt(lam)) @ vec.T
        ldd = isqrt_divided(lam)
        dinv = vec @ ((vec.T @ dS @ vec) * ldd) @ vec.T
        dU = dinv @ self.W + inv @ (dw_ao + dw_aux)
        return dE, dU, dt_ao

    def bilinear(self, E, low, y, d):
        """Tr[y (J - K/2)[d]] with the fit's vectors replaced by (E, low), less the outside
        block's k-space exchange."""
        import copy
        from pyscf import lib

        fit = self.fit
        g = copy.copy(fit)
        g.E = np.ascontiguousarray(E)
        g.ci = np.concatenate([fit.fi, np.setdiff1d(np.arange(E.shape[2]), fit.fi)])
        g.E = np.ascontiguousarray(E[:, :, g.ci])
        g.low = low
        with lib.temporary_env(fit.kfit, exchange=lambda dt: np.zeros_like(dt)):
            j, k = g._jk(d, True, True)
        return float(np.vdot(y, j) - 0.5 * np.vdot(y, k))

    def fit_gradient(self, atom, ints, y, d, eps=1.0, axes=(0, 1, 2)):
        """d/dA of Tr[y G[d]], G the fit's J - K/2: exact central difference of a quadratic,
        two fit J/K per axis (the reference for reverse_gradient)."""
        out = np.zeros(3)
        for x in axes:
            t = time.perf_counter()
            dE, dU, dT = self.vector_derivatives(atom, ints, x)
            self.timings["vector_derivatives"] += time.perf_counter() - t
            t = time.perf_counter()
            f = []
            for sgn in (1, -1):
                low = None if dU is None else np.vstack([self.U + sgn * eps * dU,
                                                         self.T + sgn * eps * dT])
                f.append(self.bilinear(self.Efull + sgn * eps * dE, low, y, d))
            self.timings["bilinear"] += time.perf_counter() - t
            self.timings["bilinear_calls"] += 2
            out[x] = (f[0] - f[1]) / (2 * eps)
        return out

    def _contract(self, x):
        """B_v . x for every real coordinate v (x symmetric)."""
        fit, kfit = self.fit, self.kfit
        out = kfit.to_real(kfit.fit(kfit._forward(np.where(fit.oo, x, 0.0))))
        nf = len(fit.fi)
        if nf:
            xfc = x[np.ix_(fit.fi, fit.ci)]
            out += (2 * np.einsum("vfc,fc->v", fit.E, xfc)
                    - np.einsum("vfg,fg->v", fit.E[:, :, :nf], xfc[:, :nf]))
        return out

    def adjoints(self, y, d):
        """The adjoints of f = Tr[y (J - K/2)[d]] for reverse_gradient, once per (y, d).

        f = sum_v (y.B_v)(d.B_v) - 1/2 Tr(y B_v d B_v) over the fit's vectors and the
        low-rank ones (with their signs), so df/dB_v = (d.B_v) y + (y.B_v) d
        - 1/2 (d B_v y + y B_v d).  E holds B_v[f, c] = B_v[c, f]: off the fi x fi block an
        entry counts twice.  E's adjoint is kept on the fit's columns ci, as the energy keeps
        the screened ones at zero.  The low-rank adjoint is carried back through
        vector_derivatives' map to its atom-independent factors.
        """
        from pyscf import lib

        t0 = time.perf_counter()
        fit, kfit = self.fit, self.kfit
        E, fi, ci = fit.E, fit.fi, fit.ci
        nf, nao = len(fi), y.shape[0]
        co = ci[nf:]
        y = 0.5 * (y + y.T)
        d = 0.5 * (d + d.T)
        rd, ry = self._contract(d), self._contract(y)

        def rows(x):
            r = 2 * x[fi]
            r[:, fi] *= 0.5
            return r

        yf, df = rows(y)[:, ci], rows(d)[:, ci]
        ebar = np.empty((kfit.nreal, nf, len(ci)))
        # B_v x[:, fi]: the outside block in k-space (as in _k_fragment), then E's rows and
        # columns.
        src = []
        for x in (d, y):
            m = x[:, fi].copy()
            m[fi] = 0.0
            src.append((kfit.rows_forward(m), x[np.ix_(fi, fi)], x[np.ix_(co, fi)]))
        for p, kind, a, b in kfit.blocks:
            if kind != "im":
                yc = [kfit.apply(p, s[0]) for s in src]
            eb = E[a:b]
            z = []
            for c, (_, xff, xof) in zip(yc, src):
                if kind == "self":
                    zb = c.real.copy()
                else:
                    zb = np.sqrt(2) * (c.real if kind == "re" else c.imag)
                zb[:, fi] = 0.0
                zb[:, ci] += np.einsum("vfc,fg->vcg", eb, xff, optimize=True)
                zb[:, fi] += np.einsum("vfc,cg->vfg", eb[:, :, nf:], xof, optimize=True)
                z.append(zb)
            mk = np.matmul(z[0].transpose(0, 2, 1), y) + np.matmul(z[1].transpose(0, 2, 1), d)
            mk *= -1.0
            mk[:, :, fi] *= 0.5
            ebar[a:b] = mk[:, :, ci] + rd[a:b, None, None] * yf + ry[a:b, None, None] * df
        adj = {"ebar": ebar}
        if self.nx:
            nx, low = self.nx, fit.low
            w, c = np.linalg.eigh(d)
            keep = np.abs(w) > 1e-14 * max(1.0, np.abs(w).max())
            c, cw = c[:, keep], c[:, keep] * w[keep]
            dp = lib.pack_tril(2 * d - np.diag(np.diag(d)))
            yp = lib.pack_tril(2 * y - np.diag(np.diag(y)))
            diag = np.arange(nao)
            lam, vec = self.lam, self.vec
            # U = vec low[:nx], W = vec Lambda^1/2 low[:nx], inv = r vec^T.
            r = vec / np.sqrt(lam)
            ub = np.zeros((nx, low.shape[1]))       # inv Ubar, packed
            lw = np.empty((nx, nx))                 # df/d(low[:nx]) . low[:nx]
            rho = np.empty((kfit.nreal, nx))        # B_v . G_x
            grows = np.empty((2 * nx, nf, nao))     # G_x[fi]
            for x0 in range(0, 2 * nx, fit.batch):
                x1 = min(x0 + fit.batch, 2 * nx)
                lx = lib.unpack_tril(low[x0:x1])
                h = np.matmul(cw, np.matmul(y, np.matmul(lx, c)).transpose(0, 2, 1))
                del lx
                g = ((low[x0:x1] @ dp)[:, None, None] * y + (low[x0:x1] @ yp)[:, None, None] * d
                     - 0.5 * (h + h.transpose(0, 2, 1)))
                del h
                g *= fit.sign[x0:x1, None, None]
                grows[x0:x1] = g[:, fi]
                nu = max(0, min(x1, nx) - x0)
                if nu:
                    for i in range(nu):
                        rho[:, x0 + i] = self._contract(g[i])
                        g[i] *= 2
                        g[i][diag, diag] *= 0.5
                    lbar = lib.pack_tril(g[:nu])    # df/d(low[x0:x0+nu]), packed
                    lw[x0:x0 + nu] = lbar @ low[:nx].T
                    ub += r[:, x0:x0 + nu] @ lbar
                    del lbar
                del g
            ldd = isqrt_divided(lam)
            sbar = vec @ (ldd * (lw * np.sqrt(lam))) @ vec.T
            ssym = sbar + sbar.T
            dbar = self.o @ ssym + rho @ r.T
            adj.update(ub=ub, urows=np.tensordot(r, grows[:nx], axes=(1, 0)),
                       trows=grows[nx:], ssym=ssym, dbar=dbar - self.q @ (self.q.T @ dbar))
        self.timings["adjoints"] += time.perf_counter() - t0
        return adj

    def reverse_gradient(self, atom, ints, adj, axes=(0, 1, 2)):
        """d/dA of Tr[y G[d]] from adjoints(y, d): fit_gradient without the fit J/K."""
        t0 = time.perf_counter()
        fit, kfit, nx = self.fit, self.kfit, self.nx
        fi = fit.fi
        p0, p1 = (int(i) for i in self.cell.aoslice_by_atom()[atom][2:])
        s0 = int(np.searchsorted(fi, p0))
        s1 = s0 + p1 - p0
        if not np.array_equal(fi[s0:s1], np.arange(p0, p1)):
            raise ValueError("atom %d is not explicit in the fit" % atom)
        ebar, ci = adj["ebar"], fit.ci
        # dE is ex on the atom's rows of E and ex[:, :, fi] on its columns (fi leads ci).
        ecol = np.ascontiguousarray(ebar[:, :, s0:s1])
        if nx:
            # d(low) on the atom's rows: (a - o) and q times ex, mirrored.
            urows = np.ascontiguousarray(adj["urows"][:, s0:s1])
            trows = np.ascontiguousarray(adj["trows"][:, s0:s1])
            c_ao = self.a - self.o
            q0, q1 = ints["slice"]
        out = np.zeros(3)
        for x in axes:
            ex = ints["ex"][x]
            g = np.einsum("vrf,vfr->", ex[:, :, fi], ecol, optimize=True)
            for v0 in range(0, kfit.nreal, 1024):
                g += np.einsum("vrc,vrc->", ex[v0:v0 + 1024][:, :, ci], ebar[v0:v0 + 1024, s0:s1],
                               optimize=True)
            if nx:
                g += 2 * (np.vdot(np.tensordot(c_ao, ex, axes=(0, 0)), urows)
                          + np.vdot(np.tensordot(self.q, ex, axes=(0, 0)), trows))
                dmet = ints["dmet"][x]
                dj = np.zeros((nx, kfit.nk * kfit.naux))
                dj[q0:q1] = dmet[:, nx:]
                g += (2 * np.einsum("yrc,yrc->", ints["dvx"][x], urows, optimize=True)
                      + np.vdot(adj["ub"][q0:q1], ints["dpairs"][x])
                      + np.vdot(dmet[:, :nx], adj["ssym"][q0:q1])
                      - np.vdot(kfit.coords(dj), adj["dbar"]))
            out[x] = g
        self.timings["reverse"] += time.perf_counter() - t0
        return out

    def nuclear_rows_gradient(self, atom, ints, m):
        """2 sum_{mu on A, nu} M_munu dV_munu/dA from the AO derivative (all nuclei fixed)."""
        ao = np.flatnonzero(_atom_mask(self.cell, atom))
        return 2 * np.einsum("xmn,mn->x", ints["vnuc"], m[ao])


def isqrt_divided(w):
    """Divided differences of w^-1/2, (f_i - f_j)/(w_i - w_j) with f'(w_i) on the diagonal,
    in closed form.  The moving-auxiliary metric has eigenvalues down to 1e-13 (hBN): there
    differencing cancels, and a degeneracy threshold merges distinct eigenvalues."""
    s = np.sqrt(w)
    return -1.0 / (s[:, None] * s[None, :] * (s[:, None] + s[None, :]))


def _atom_mask(cell, atom):
    m = np.zeros(cell.nao_nr(), dtype=bool)
    p0, p1 = cell.aoslice_by_atom()[atom][2:]
    m[p0:p1] = True
    return m


def nucleus_field(cell, m, atoms, settings, block_mb=512):
    """d/dC of V_M(C) = sum M_ab (ab|eta_C) at each nucleus C of ``atoms``: (len(atoms), 3).
    By translational invariance, d/dC = -sum_B d/dB at a fixed nucleus."""
    from pyscf.pbc.df import aft

    fake = aft._fake_nuc(cell, with_pseudo=False)
    fake._bas = fake._bas[list(atoms)].copy()
    ms = m + m.T
    nao = cell.nao_nr()
    sl = cell.aoslice_by_atom()
    per_ao = 3 * len(atoms) * nao * 8e-6
    out = np.zeros((len(atoms), 3))
    block, nrow = [], 0
    for b in list(range(cell.natm)) + [None]:
        n = 0 if b is None else int(sl[b][3] - sl[b][2])
        if block and (b is None or (nrow + n) * per_ao > block_mb):
            rows = derivative_rows(cell, fake, block, settings, FIELD_PRECISION)
            ao = np.concatenate([np.arange(sl[a][2], sl[a][3]) for a in block])
            out -= np.einsum("xcmn,mn->cx", rows, ms[ao], optimize=True)
            block, nrow = [], 0
        if b is not None:
            block.append(b)
            nrow += n
    return out
