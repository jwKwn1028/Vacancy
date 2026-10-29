"""The Gamma density fit of a supercell whose moved atoms lie in the fragment, built from
the ideal crystal's k-point fit plus integrals that touch the moved atoms.

Pairs without a moved AO keep the k-point vectors; pairs with one get Gamma rows plus the
seam (k-point minus Gamma rows at the ideal geometry).  The moved atoms' auxiliary functions
are then moved by a rank-2|F| correction.  The nuclear attraction is assembled the same way.
The explicit parts change with the geometry, so they are built at their own precision (1e-12
by default).
"""
import contextlib
import ctypes
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.pbc.df import aft, ft_ao, rsdf_builder
from pyscf.pbc.df import df as pbc_df
from pyscf.pbc.gto.cell import conc_cell

from system.split_fit import KpointOutside, fragment_aos

# OpenBLAS's thread-count calls, by the names its builds export.
_OPENBLAS_CALLS = (("openblas_get_num_threads", "openblas_set_num_threads"),
                   ("openblas_get_num_threads64_", "openblas_set_num_threads64_"),
                   ("scipy_openblas_get_num_threads64_", "scipy_openblas_set_num_threads64_"))


_OPENBLAS = []


def _openblas():
    """(get, set) of the loaded OpenBLAS's thread count, or None."""
    if _OPENBLAS:
        return _OPENBLAS[0]
    _OPENBLAS.append(_find_openblas())
    return _OPENBLAS[0]


def _find_openblas():
    try:
        with open("/proc/self/maps") as f:
            paths = {line.split()[-1] for line in f if "openblas" in line and ".so" in line}
    except OSError:
        return None
    for path in sorted(paths):
        try:
            blas = ctypes.CDLL(path)
        except OSError:
            continue
        for get, put in _OPENBLAS_CALLS:
            if hasattr(blas, get) and hasattr(blas, put):
                return getattr(blas, get), getattr(blas, put)
    return None


def _one_blas_thread():
    """OpenBLAS on one thread from this thread: an OpenMP build follows the calling thread's
    OpenMP count (PySCF shares its runtime), a pthreads build one global count."""
    lib.num_threads(1)
    calls = _openblas()
    if calls is not None:
        calls[1](1)


@contextlib.contextmanager
def single_threaded_blas():
    """OpenBLAS on one thread inside; yields False when its calls are not found."""
    calls = _openblas()
    if calls is None:
        yield False
        return
    get, put = calls
    old, omp = int(get()), lib.num_threads()
    put(1)
    try:
        yield True
    finally:
        # An OpenMP build sets this thread's OpenMP count with it.
        put(old)
        lib.num_threads(omp)


def fragment_pairs(fi, cols):
    """Packed (lower-triangle) indices of the pairs (fi[i], cols[j]), each pair once, with
    their i and j."""
    i, j = np.meshgrid(np.arange(len(fi)), np.arange(len(cols)), indexing="ij")
    a, b = fi[i], cols[j]
    keep = ~(np.isin(b, fi) & (b > a))
    i, j, a, b = i[keep], j[keep], a[keep], b[keep]
    hi, lo = np.maximum(a, b), np.minimum(a, b)
    return hi * (hi + 1) // 2 + lo, i, j


def recording_builder(records):
    """PySCF's builder, recording itself and the metric factor of each momentum group."""

    class Recorder(rsdf_builder._RSGDFBuilder):
        def gen_uniq_kpts_groups(self, j_only, h5swap, kk_idx=None):
            records.append({"builder": self, "groups": []})
            for kpt, pairs, cd in super().gen_uniq_kpts_groups(j_only, h5swap, kk_idx):
                records[-1]["groups"].append((np.array(pairs), cd))
                yield kpt, pairs, cd

    return Recorder


def kpoint_mean_field(atoms, lattice, basis, kmesh, **kwargs):
    """build_pristine_mean_field on ``kmesh`` with the GDF metric factors recorded."""
    from system.pristine import build_pristine_mean_field

    records = []
    with lib.temporary_env(pbc_df, _RSGDFBuilder=recording_builder(records)):
        _, kmf = build_pristine_mean_field(atoms, lattice, basis, kmesh=kmesh, **kwargs)
    return kmf, KpointFit(kmf, kmesh, records)


def gamma_settings(cell, auxcell):
    """(omega, mesh, ke_cutoff) a Gamma GDF of ``cell`` with ``auxcell`` would choose."""
    omega, mesh, ke = rsdf_builder._guess_omega(auxcell, np.zeros((1, 3)), None)
    mesh = np.array(mesh)
    if cell.dimension == 2 and cell.low_dim_ft_type != "inf_vacuum":
        mesh[2] = rsdf_builder._estimate_meshz(cell)
    elif cell.dimension < 2:
        mesh[cell.dimension:] = cell.mesh[cell.dimension:]
    return omega, cell.symmetrize_mesh(mesh), ke


class GammaBuilder(rsdf_builder._RSGDFBuilder):
    """Raw Gamma (A|ab), without the metric.  Keeps room for two one-shell batches, below
    which PySCF shrinks its G blocks to 16 points, and a quarter of max_memory: PySCF
    budgets max_memory less what the process holds, down to 1 GB, and a pass in such small
    pair blocks runs several times slower on many threads."""

    def __init__(self, cell, auxcell, settings):
        super().__init__(cell, auxcell, np.zeros((1, 3)))
        self.omega, self.mesh, self.ke_cutoff = settings[0], np.array(settings[1]), settings[2]
        step = int(np.diff(cell.ao_loc_nr()).max()) * cell.nao_nr()
        room = max(2 * step * auxcell.nao_nr() * 64e-6, 0.25 * self.max_memory)
        self.max_memory = max(self.max_memory, lib.current_memory()[0] + room)

    def gen_uniq_kpts_groups(self, j_only, h5swap, kk_idx=None):
        yield np.zeros(3), np.zeros(1, dtype=np.int32), None

    def solve_cderi(self, cd_j2c, j3cR, j3cI):
        return j3cR.T, None


class RowBuilder(GammaBuilder):
    """For a slice of rows: PySCF's G=0 term would take the overlap's leading rows."""

    def gen_j3c_loader(self, h5group, kpt, kpt_ij_idx, aosym):
        with lib.temporary_env(rsdf_builder, _gaussian_int=lambda c: np.zeros(c.nao)):
            return super().gen_j3c_loader(h5group, kpt, kpt_ij_idx, aosym)

    def g0_term(self):
        cell = self.cell
        if not (self.omega != 0 and (cell.dimension == 3 or (
                cell.dimension == 2 and cell.low_dim_ft_type != "inf_vacuum"))):
            return None, None
        if self.exclude_d_aux and cell.dimension > 0:
            rs_aux = self.rs_auxcell
            chg = rsdf_builder._gaussian_int(rs_aux)
            chg[rs_aux.get_ao_type() == ft_ao.SMOOTH_BASIS] = 0
            chg = rs_aux.recontract_1d(chg[:, None]).ravel()
        else:
            chg = rsdf_builder._gaussian_int(self.auxcell)
        if self.exclude_dd_block:
            rs = self.rs_cell
            s = rs.pbc_intor("int1e_ovlp", hermi=1)
            smooth = rs.get_ao_type() == ft_ao.SMOOTH_BASIS
            s[smooth[:, None] & smooth] = 0
            s = rs.recontract(dim=2)(s)
        else:
            s = cell.pbc_intor("int1e_ovlp", hermi=1)
        return np.pi / self.omega ** 2 / cell.vol * chg, s


def with_precision(cell, precision):
    """``cell`` with its screening and lattice sums set by ``precision``."""
    c = cell.copy()
    c.precision = float(precision)
    c.build(False, False)
    return c


def with_atoms(cell, atoms):
    """``cell`` with its atoms replaced by ``atoms`` ([label, xyz Bohr])."""
    from pyscf import gto

    c = cell.copy()
    c.atom = [(label, np.asarray(xyz, dtype=float)) for label, xyz in atoms]
    c.unit = "Bohr"
    c.a = cell.lattice_vectors()
    c.charge = 0
    c.spin = int(sum(gto.charge(gto.mole._std_symbol(label)) for label, _ in atoms) % 2)
    c.build(False, False)
    return c


def h5_steps(builder, **kwargs):
    with tempfile.TemporaryDirectory(dir=lib.param.TMPDIR) as tmp:
        builder.make_j3c(os.path.join(tmp, "j3c.h5"), **kwargs)
        with h5py.File(os.path.join(tmp, "j3c.h5"), "r") as f:
            group = f["j3c/0"]
            for step in range(len(group)):
                yield np.asarray(group[str(step)])


def fragment_vectors(cell, auxcell, atoms, kfit, settings, nuclear=False):
    """M_p V^p for the rows on ``atoms``, {p: (naux_p, nrow, nao)}, and with ``nuclear`` the
    nuclear attraction's rows."""
    nk, nao = kfit.nk, cell.nao_nr()
    naux = auxcell.nao_nr() // nk
    atoms = sorted(int(i) for i in atoms)
    order = atoms + sorted(set(range(cell.natm)) - set(atoms))
    slices = cell.aoslice_by_atom()
    perm = np.concatenate([np.arange(slices[i][2], slices[i][3]) for i in order])
    inv = np.argsort(perm)
    nsh = sum(int(slices[i][1] - slices[i][0]) for i in atoms)
    nrow = sum(int(slices[i][3] - slices[i][2]) for i in atoms)
    pcell = with_atoms(cell, [cell._atom[i] for i in order])
    if nuclear:
        auxcell = conc_cell(auxcell, aft._fake_nuc(cell, with_pseudo=False))
    builder = RowBuilder(pcell, auxcell, settings)
    ft = kfit.phase[:, kfit.half].conj().T / np.sqrt(nk)
    out = {p: np.empty((kfit.lq[p].shape[1], nrow, nao),
                       dtype=float if kfit.minus[p] == p else complex) for p in kfit.half}
    nuc = np.empty((nrow, nao)) if nuclear else None
    row = 0
    vbar = s = None
    for v in h5_steps(builder, aosym="s1",
                       shls_slice=(0, nsh, 0, pcell.nbas, 0, auxcell.nbas)):
        if row == 0:
            vbar, s = builder.g0_term()
        v = v.reshape(len(v), -1, nao)
        nr = v.shape[1]
        if vbar is not None:
            v -= vbar[:, None, None] * s[row:row + nr]
        v = v[:, :, inv]
        if nuclear:
            nuc[row:row + nr] = -np.tensordot(cell.atom_charges(), v[nk * naux:], axes=(0, 0))
        vt = np.tensordot(ft, v[:nk * naux].reshape(nk, -1), axes=(1, 0))
        for i, p in enumerate(kfit.half):
            x = vt[i].reshape(naux, -1).T
            imag = None if kfit.minus[p] == p else np.ascontiguousarray(x.imag)
            e = rsdf_builder._RSGDFBuilder.solve_cderi(
                builder, kfit.metric[p], np.ascontiguousarray(x.real), imag)[0]
            out[p][:, row:row + nr] = e.reshape(-1, nr, nao)
        row += nr
    return out, nuc


def raw_pairs(cell, auxcell, settings):
    """Gamma (A|ab) for ``auxcell``, every pair: (naux, npair) packed."""
    return np.hstack(list(h5_steps(GammaBuilder(cell, auxcell, settings), aosym="s2")))


def raw_pair_difference(cell, auxcell, settings):
    """Gamma (A|ab) - (A+n|ab) for the halves of ``auxcell``, every pair: (n, npair) packed."""
    blocks = []
    for v in h5_steps(GammaBuilder(cell, auxcell, settings), aosym="s2"):
        n = len(v) // 2
        blocks.append(v[:n] - v[n:])
    return np.hstack(blocks)


def gamma_metric(cell, auxcell, settings):
    """The Gamma metric (A|B) of ``auxcell``."""
    builder = GammaBuilder(cell, auxcell, settings)
    builder.build()
    return np.asarray(builder.get_2c2e(np.zeros((1, 3)))[0])


class KpointFit(KpointOutside):
    """The k-point fit in k-space, with the metric factor of each momentum."""

    def __init__(self, kmf, kmesh, records):
        super().__init__(kmf, kmesh)
        cell, kpts = kmf.cell, np.asarray(kmf.kpts)
        self.cell, self.kmesh = cell, tuple(int(x) for x in kmesh)
        self.kpts = kpts
        self.auxbasis = kmf.with_df.auxbasis
        frac = cell.get_scaled_kpts(kpts)
        nk = self.nk
        self.metric = {}
        for pairs, cd in records[-1]["groups"]:
            ki, kj = np.divmod(pairs, nk)
            d = frac[kj] - frac[ki]
            p = {int(np.where(np.abs(frac - x - np.round(frac - x)).max(axis=1) < 1e-8)[0][0])
                 for x in d}
            if len(p) != 1 or next(iter(p)) in self.metric:
                raise RuntimeError("unexpected k-point pair grouping")
            self.metric[p.pop()] = cd
        if sorted(self.metric) != list(range(nk)):
            raise RuntimeError("a momentum has no metric factor")
        self.m = {}
        for p, (mat, negative, tag) in self.metric.items():
            if negative is not None:
                raise NotImplementedError("GDF with a negative metric part")
            self.m[p] = (scipy.linalg.solve_triangular(mat, np.eye(len(mat)), lower=True)
                         if tag == "CD" else np.asarray(mat))
        self.naux = self.m[0].shape[1]
        self.half = [p for p in range(nk) if p <= self.minus[p]]
        self.blocks, start = [], 0
        for p in self.half:
            m = self.lq[p].shape[1]
            for kind in (("self",) if self.minus[p] == p else ("re", "im")):
                self.blocks.append((p, kind, start, start + m))
                start += m
        self.nreal = start
        self._seams = {}
        self._settings = {}
        self._vnuc = None
        # Threads for exchange (None: PySCF's count).
        self.workers = None
        # Single entries, keyed by the moved atoms: (A|ab) at the ideal positions, and the
        # moving auxiliary functions' potentials outside the fragment.
        self.ideal_pairs = None
        self.outside_potentials = None

    def ideal_supercell(self, like):
        """``like`` (a supercell Cell) with every atom at its ideal position."""
        from hact_relax.geometry import ideal_supercell

        prim = [[a[0], np.asarray(a[1])] for a in self.cell._atom]
        _, coords = ideal_supercell(prim, self.cell.lattice_vectors(), self.kmesh)
        return with_atoms(like, [(like._atom[i][0], coords[i]) for i in range(like.natm)])

    def gamma_settings(self, ideal, auxcell):
        key = float(ideal.precision)
        if key not in self._settings:
            self._settings[key] = gamma_settings(ideal, auxcell)
        return self._settings[key]

    def to_real(self, z):
        out = []
        for p, kind, _, _ in self.blocks:
            if kind == "self":
                out.append(np.real(z[p]))
            elif kind == "re":
                out.append(np.sqrt(2) * z[p].real)
            else:
                out.append(np.sqrt(2) * z[p].imag)
        return np.concatenate(out)

    def to_complex(self, r):
        """c[p] for every p with sum_p B^p c^p = sum_v R_v r_v (r along axis 0)."""
        c = [None] * self.nk
        for p, kind, a, b in self.blocks:
            if kind == "self":
                c[p] = r[a:b].astype(complex)
            elif kind == "re":
                c[p] = r[a:b] / np.sqrt(2)
            else:
                c[p] = c[p] - 1j * r[a:b] / np.sqrt(2)
                c[self.minus[p]] = c[p].conj()
        return c

    def coords(self, w):
        """Coordinates v of the functions whose potentials on the auxiliary basis are w."""
        w3 = np.asarray(w).reshape(len(w), self.nk, self.naux)
        wh = np.einsum("Rp,zRP->pPz", self.phase.conj(), w3) / np.sqrt(self.nk)
        return self.to_real({p: self.m[p] @ wh[p] for p in self.half})

    def aux_coords(self, xs):
        """Coordinates v of the ideal supercell's auxiliary functions xs."""
        cell_of, aux_of = np.divmod(np.asarray(xs), self.naux)
        out = {}
        for p in self.half:
            mat, _, tag = self.metric[p]
            if tag == "CD":
                a = mat.conj().T
            else:
                m = self.m[p]
                a = m / np.einsum("ij,ij->i", m, m.conj()).real[:, None]
            out[p] = a[:, aux_of] * self.phase[cell_of, p].conj() / np.sqrt(self.nk)
        return self.to_real(out)

    def fit(self, dt):
        """g^p = sum_ab B^p_ab D_ab for every p, from D~ = _forward(D)."""
        ks = np.arange(self.nk)
        return [np.einsum("kPab,kab->P", self.lq[p],
                          dt[self.minus[ks], :, self.minus[self.ksum[ks, p]], :])
                / self.nk ** 1.5 for p in range(self.nk)]

    def potential(self, c):
        """sum_p B^p . c^p in k-space (for _backward); c[p]: (naux_p,) or (naux_p, nz)."""
        nk, n = self.nk, self.nao
        ks = np.arange(nk)
        many = np.ndim(c[0]) == 2
        mt = np.zeros((nk, n, nk, n) + (np.shape(c[0])[1:]), dtype=complex)
        for p in range(nk):
            mt[ks, :, self.ksum[ks, p]] += np.sqrt(nk) * np.tensordot(
                self.lq[p], c[p], axes=(1, 0))
        return np.moveaxis(mt, -1, 0) if many else mt

    def backward_many(self, mt):
        nk, n = self.nk, self.nao
        t = np.tensordot(mt, self.phase, axes=(-4, 1))
        t = np.tensordot(t, self.phase.conj(), axes=(-3, 1))
        t = np.moveaxis(t, (-2, -1), (-4, -2))
        return t.reshape(mt.shape[:-4] + (nk * n, nk * n)) / nk ** 2

    def exchange(self, dt):
        """K~ of a real supercell density in k-space (for _backward),

        kt[k1,-k3] = 1/N sum_p,P L^(k1,k1+p)_P D~(k1+p,-k3+p) L^(k3,k3-p)_P^T.

        A real density gives kt[-k1,-k3] = conj(kt[k1,k3]), so half the column blocks are
        built: per p, in tasks of one column block and a range of rows, over threads that
        each run BLAS on one thread.
        """
        nk, n = self.nk, self.nao
        minus, ksum = self.minus, self.ksum
        kt = np.zeros_like(dt)
        half = [k for k in range(nk) if k <= minus[k]]
        workers = int(self.workers or lib.num_threads())
        with contextlib.ExitStack() as stack:
            if workers > 1 and not stack.enter_context(single_threaded_blas()):
                workers = 1
            # About four tasks per worker: split the rows when the column blocks are few.
            nrow = -(-nk // min(nk, -(-4 * workers // len(half))))
            tasks = [(k3, slice(r, min(r + nrow, nk))) for k3 in half
                     for r in range(0, nk, nrow)]
            pool = None
            if workers > 1:
                pool = stack.enter_context(
                    ThreadPoolExecutor(workers, initializer=_one_blas_thread))
            for p in range(nk):
                a = self.lq[p]
                naux = a.shape[1]
                a2 = np.ascontiguousarray(a.transpose(0, 2, 3, 1)).reshape(nk, n, n * naux)
                b = self.lq[minus[p]]
                rows = ksum[:, p]

                def task(job):
                    k3, r = job
                    col = minus[ksum[k3, minus[p]]]
                    d = dt[rows[r], :, col, :].reshape(-1, n)
                    bt = b[k3].transpose(2, 0, 1).reshape(n, naux * n)
                    u = (d @ bt).reshape(-1, n * naux, n)
                    kt[r, :, minus[k3], :] += np.matmul(a2[r], u) / nk

                if pool is None:
                    for job in tasks:
                        task(job)
                else:
                    list(pool.map(task, tasks))
        built = {int(minus[k]) for k in half}
        for c in range(nk):
            if c not in built:
                kt[:, :, c, :] = kt[minus, :, minus[c], :].conj()
        return kt

    def rows_forward(self, m):
        nk, n = self.nk, self.nao
        return np.tensordot(self.phase.conj(), m.reshape(nk, n, -1), axes=(0, 0))

    def rows_backward(self, acc):
        nk, n = self.nk, self.nao
        return np.tensordot(self.phase, acc, axes=(1, 0)).reshape(nk * n, -1) / nk ** 1.5

    def apply(self, p, mh):
        """B^p_P M for every P, with mh = rows_forward(M): (naux_p, nao, m)."""
        nk, n = self.nk, self.nao
        naux = self.lq[p].shape[1]
        q = np.matmul(self.lq[p].reshape(nk, naux * n, n), mh[self.ksum[:, p]])
        out = np.tensordot(self.phase, q.reshape(nk, naux, n, -1), axes=(1, 0)) / nk ** 1.5
        return out.transpose(1, 0, 2, 3).reshape(naux, nk * n, -1)

    def accumulate(self, p, w, acc):
        """acc += sum_P B^p_P W_P in k-space (for rows_backward); w: (naux_p, nao, m)."""
        nk, n = self.nk, self.nao
        naux, m = w.shape[0], w.shape[-1]
        wh = np.tensordot(self.phase.conj(), w.reshape(naux, nk, n, m), axes=(0, 1))
        a = self.lq[p].transpose(0, 2, 1, 3).reshape(nk, n, naux * n)
        acc += np.matmul(a, wh[self.ksum[:, p]].reshape(nk, naux * n, m))

    def unfolded_rows(self, rows):
        """{p: B^p[rows, :]} for p in self.half, from the k-point vectors."""
        nk, n = self.nk, self.nao
        cell_of, orb_of = np.divmod(np.asarray(rows), n)
        out = {}
        for p in self.half:
            x = self.phase[cell_of].T[:, None, :, None] * self.lq[p][:, :, orb_of, :]
            t = np.tensordot(x, self.phase[:, self.ksum[:, p]].conj(), axes=(0, 1))
            t = t.transpose(0, 1, 3, 2).reshape(t.shape[0], len(rows), nk * n) / nk ** 1.5
            out[p] = t.real if self.minus[p] == p else t
        return out

    def nuclear(self):
        """The ideal crystal's nuclear attraction from the k-point cell, unfolded."""
        if self._vnuc is None:
            from pyscf.pbc.tools.k2gamma import to_supercell_ao_integrals

            vk = np.asarray(pbc_df.GDF(self.cell, self.kpts).get_nuc(self.kpts))
            self._vnuc = np.asarray(to_supercell_ao_integrals(
                self.cell, self.kpts, vk, kmesh=list(self.kmesh))).real
        return self._vnuc

    def seam_rows(self, atoms, ideal, auxcell, settings, nuclear=False):
        """k-point minus Gamma rows at the ideal geometry, per atom, and with ``nuclear``
        the nuclear attraction's."""
        seams = self._seams.setdefault(float(ideal.precision), {})
        missing = [a for a in sorted(atoms)
                   if a not in seams or (nuclear and seams[a][1] is None)]
        if missing:
            for a in missing:
                seams.pop(a, None)
            rows = np.flatnonzero(fragment_aos(ideal, missing))
            gamma, nuc = fragment_vectors(ideal, auxcell, missing, self, settings, nuclear)
            c = self.to_real(self.unfolded_rows(rows)) - self.to_real(gamma)
            cn = self.nuclear()[rows] - nuc if nuclear else None
            start = 0
            for a in missing:
                n = int(np.count_nonzero(fragment_aos(ideal, [a])))
                seams[a] = (c[:, start:start + n],
                            None if cn is None else cn[start:start + n])
                start += n
        parts = [seams[a] for a in sorted(atoms)]
        c = np.concatenate([x[0] for x in parts], axis=1)
        return c, (np.concatenate([x[1] for x in parts]) if nuclear else None)


def moved_atoms(cell, ideal, tol=1e-6):
    shift = cell.atom_coords() - ideal.atom_coords()
    frac = shift @ np.linalg.inv(cell.lattice_vectors())
    shift = (frac - np.round(frac)) @ cell.lattice_vectors()
    return [int(i) for i in np.flatnonzero(np.linalg.norm(shift, axis=1) > tol)]


class FragmentFit:
    """The fit as a GDF stand-in for a Gamma KRHF of ``cell``; ``nuclear="kpoint"``
    assembles the nuclear attraction from the k-point one."""

    def __init__(self, kfit, cell, fragment, *, nuclear="kpoint", seam=True, move_aux=True,
                 column_tol=1e-12, batch=16, precision=1e-12, explicit=()):
        if nuclear not in ("kpoint", "gamma"):
            raise ValueError("nuclear must be 'kpoint' or 'gamma'")
        if nuclear == "kpoint" and (cell.pseudo or len(cell._ecpbas)):
            raise NotImplementedError("FragmentFit: k-point nuclear attraction is all-electron")
        self.kfit, self.cell, self.nuclear = kfit, cell, nuclear
        self.fragment = tuple(sorted(int(i) for i in fragment))
        gdf = pbc_df.GDF(cell)
        gdf.auxbasis = kfit.auxbasis
        self.gdf = gdf
        # The cell the explicit parts are built in; precision None keeps cell's own.
        self.precision = None if precision is None else float(precision)
        if self.precision is not None and self.precision != cell.precision:
            cell = with_precision(cell, self.precision)
        self.row_cell = cell
        ideal = kfit.ideal_supercell(cell)
        if [cell.atom_symbol(i) for i in range(cell.natm)] != [
                ideal.atom_symbol(i) for i in range(ideal.natm)]:
            raise ValueError("the supercell does not list its atoms in the k-mesh order")
        moved = moved_atoms(cell, ideal)
        stray = sorted(set(moved) - set(self.fragment))
        if stray:
            raise ValueError("atoms %s outside the fragment have moved" % stray)
        # Explicit atoms count as moved, so a relaxation's fit keeps one structure.
        self.moved = sorted(set(moved) | {int(a) for a in explicit})
        auxcell = pbc_df.make_auxcell(ideal, kfit.auxbasis)
        self.settings = kfit.gamma_settings(ideal, auxcell)
        rows = fragment_aos(cell, self.moved)
        self.fi = np.flatnonzero(rows)
        self.oo = np.outer(~rows, ~rows)
        self.batch = int(batch)
        self.timings = {}
        self.seam_max = None
        self.aux_residual_min = None
        self.low = None
        self.vnuc = None
        kpoint = nuclear == "kpoint"
        if not self.moved:
            self.ci = self.fi
            self.E = np.zeros((kfit.nreal, 0, 0))
            if kpoint:
                t0 = time.perf_counter()
                v = kfit.nuclear()
                self.vnuc = 0.5 * (v + v.T)
                self.timings["nuclear"] = time.perf_counter() - t0
            return
        t0 = time.perf_counter()
        vectors, nuc = fragment_vectors(cell, auxcell, self.moved, kfit, self.settings, kpoint)
        e = kfit.to_real(vectors)
        del vectors
        self.timings["rows"] = time.perf_counter() - t0
        if seam:
            t0 = time.perf_counter()
            c, cn = kfit.seam_rows(self.moved, ideal, auxcell, self.settings, kpoint)
            self.timings["seam"] = time.perf_counter() - t0
            self.seam_max = float(abs(c).max())
            e += c
            if kpoint:
                nuc += cn
        keep = np.abs(e).max(axis=(0, 1)) > column_tol
        keep[self.fi] = False
        self.ci = np.concatenate([self.fi, np.flatnonzero(keep)])
        self.E = np.ascontiguousarray(e[:, :, self.ci])
        del e
        if move_aux or kpoint:
            t0 = time.perf_counter()
            dv, dn = self._pair_differences(cell, ideal, move_aux, kpoint)
            self.timings["moved_pairs"] = time.perf_counter() - t0
            if move_aux:
                self.low, self.sign = self._moving_aux(cell, ideal, auxcell, dv)
            del dv
            if kpoint:
                t0 = time.perf_counter()
                v = kfit.nuclear().copy()
                charges = cell.atom_charges()[self.moved]
                v[self.oo] -= lib.unpack_tril(charges @ dn)[self.oo]
                nuc[:, self.fi] = 0.5 * (nuc[:, self.fi] + nuc[:, self.fi].T)
                v[self.fi] = nuc
                v[:, self.fi] = nuc.T
                self.vnuc = 0.5 * (v + v.T)
                self.timings["nuclear"] = time.perf_counter() - t0

    def _pair_differences(self, cell, ideal, with_aux, with_nuc):
        """(A'|ab) - (A|ab) over every pair, for the moved atoms' auxiliary functions and
        nuclei.  (A|ab) at the ideal positions changes with the geometry only on the moved
        atoms' AOs: the k-point fit keeps the rest from the last fit with these moved atoms.
        """
        from system.periodic_derivatives import raw_rows

        kfit = self.kfit

        def half(atoms):
            c = with_atoms(cell, atoms)
            parts = ([pbc_df.make_auxcell(c, kfit.auxbasis)] if with_aux else []) + (
                [aft._fake_nuc(c, with_pseudo=False)] if with_nuc else [])
            return parts[0] if len(parts) == 1 else conc_cell(*parts)

        new = half([cell._atom[i] for i in self.moved])
        old = half([ideal._atom[i] for i in self.moved])
        key = (tuple(self.moved), float(cell.precision), with_aux, with_nuc)
        diff = raw_pairs(cell, new, self.settings)
        if kfit.ideal_pairs is None or kfit.ideal_pairs[0] != key:
            kfit.ideal_pairs = None
            pairs = raw_pairs(cell, old, self.settings)
            diff -= pairs
            kfit.ideal_pairs = (key, pairs)
        else:
            idx, i, j = fragment_pairs(self.fi, np.arange(cell.nao_nr()))
            moved_pairs = diff[:, idx]
            diff -= kfit.ideal_pairs[1]
            diff[:, idx] = moved_pairs - raw_rows(cell, old, self.moved, self.settings)[:, i, j]
        nx = diff.shape[0] - (len(self.moved) if with_nuc else 0)
        return (diff[:nx] if with_aux else None), (diff[nx:] if with_nuc else None)

    def __getattr__(self, name):
        if name in ("kfit", "gdf", "E", "fi", "ci", "oo", "low", "sign", "timings", "vnuc",
                    "nuclear"):
            raise AttributeError(name)
        if name in ("sr_loop", "get_eri", "get_ao_eri", "ao2mo", "get_mo_eri"):
            raise NotImplementedError("FragmentFit: %s" % name)
        return getattr(self.gdf, name)

    def build(self, *args, **kwargs):
        return self

    def get_nuc(self, kpts=None):
        kpts, single = aft._check_kpts(self.gdf, kpts)
        if len(kpts) != 1 or abs(kpts).max() > 1e-12:
            raise NotImplementedError("FragmentFit: Gamma only")
        if self.vnuc is None:
            t0 = time.perf_counter()
            self.vnuc = np.asarray(self.gdf.get_nuc(np.zeros(3))).real
            self.timings["nuclear"] = time.perf_counter() - t0
        return self.vnuc.copy() if single else self.vnuc[None].copy()

    def dense_potential(self, r, out, add=False, outside=None):
        """out[z] (+)= sum_v R_v r[v, z], packed: the fit's vectors combined with coordinates r.
        ``outside`` is the part off the fragment's rows, if known (packed)."""
        kfit = self.kfit
        nz = r.shape[1]
        if outside is not None:
            idx, i, j = fragment_pairs(self.fi, self.ci)
            for z0 in range(0, nz, self.batch):
                z1 = min(z0 + self.batch, nz)
                v = outside[z0:z1].copy()
                v[:, idx] = np.tensordot(r[:, z0:z1], self.E, axes=(0, 0))[:, i, j]
                if add:
                    out[z0:z1] += v
                else:
                    out[z0:z1] = v
            return out
        for z0 in range(0, nz, self.batch):
            z1 = min(z0 + self.batch, nz)
            mt = kfit.potential([x for x in kfit.to_complex(r[:, z0:z1])])
            v = np.where(self.oo, kfit.backward_many(mt).real, 0.0)
            if len(self.fi):
                fx = np.tensordot(r[:, z0:z1], self.E, axes=(0, 0))
                for z in range(z1 - z0):
                    v[z][np.ix_(self.ci, self.fi)] = fx[z].T
                    v[z][np.ix_(self.fi, self.ci)] = fx[z]
            if add:
                out[z0:z1] += lib.pack_tril(v)
            else:
                out[z0:z1] = lib.pack_tril(v)
        return out

    def _moving_aux(self, cell, ideal, auxcell, dv):
        kfit = self.kfit
        nao = cell.nao_nr()
        sl = auxcell.aoslice_by_atom()
        xf = np.concatenate([np.arange(sl[i][2], sl[i][3]) for i in self.moved])
        nx = len(xf)
        new = [cell._atom[i] for i in self.moved]
        low = np.empty((2 * nx, nao * (nao + 1) // 2))
        low[:nx] = dv
        t0 = time.perf_counter()
        j2c = gamma_metric(cell, pbc_df.make_auxcell(
            with_atoms(cell, new + list(ideal._atom)), kfit.auxbasis), self.settings)
        dj = j2c[:nx, nx:] - j2c[nx + xf, nx:]
        djff = j2c[:nx, :nx] - j2c[np.ix_(nx + xf, nx + xf)]
        del j2c
        self.timings["moved_aux_metric"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        ex = np.zeros((nx, kfit.nk * kfit.naux))
        ex[np.arange(nx), xf] = 1.0
        q = np.linalg.qr(kfit.coords(ex))[0]
        a = kfit.aux_coords(xf)
        cdj = kfit.coords(dj)
        k_new = a + cdj
        w = q.T @ k_new
        o = k_new - q @ w
        s_new = a.T @ a + djff - o.T @ o
        # T = R q; its part off the fragment's rows depends on the ideal crystal alone.
        key = (tuple(self.moved), float(cell.precision))
        cached = kfit.outside_potentials
        if cached is not None and cached[0] == key:
            self.dense_potential(q, low[nx:], outside=cached[1])
        else:
            kfit.outside_potentials = None
            self.dense_potential(q, low[nx:])
            outside = low[nx:].copy()
            outside[:, fragment_pairs(self.fi, self.ci)[0]] = 0.0
            kfit.outside_potentials = (key, outside)
        # a - o = q w - coords(dj), and R is linear.
        self.dense_potential(-cdj, low[:nx], add=True)
        for c0 in range(0, low.shape[1], 1 << 20):
            low[:nx, c0:c0 + (1 << 20)] += w.T @ low[nx:, c0:c0 + (1 << 20)]
        lam, vec = np.linalg.eigh(s_new)
        if lam.min() <= 0:
            raise RuntimeError("moving auxiliary basis: residual metric not positive definite")
        t = (vec / np.sqrt(lam)).T
        for c0 in range(0, low.shape[1], 1 << 20):
            low[:nx, c0:c0 + (1 << 20)] = t @ low[:nx, c0:c0 + (1 << 20)]
        self.timings["moved_aux_vectors"] = time.perf_counter() - t0
        self.aux_residual_min = float(lam.min())
        self.aux_eig = (lam, vec)
        return low, np.concatenate([np.ones(nx), -np.ones(nx)])

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None, with_j=True, with_k=True,
               omega=None, exxdiv=None):
        if omega is not None or kpts_band is not None or exxdiv is not None:
            raise NotImplementedError("FragmentFit: Gamma, exxdiv=None only")
        dm = np.asarray(dm)
        n = dm.shape[-1]
        vj, vk = [], []
        for d in dm.reshape(-1, n, n):
            if abs(d.imag).max() > 0 or abs(d - d.T).max() > 1e-12 * max(1.0, abs(d).max()):
                raise NotImplementedError("FragmentFit: real symmetric densities only")
            j, k = self._jk(d.real, with_j, with_k)
            vj.append(j)
            vk.append(k)
        vj = np.asarray(vj).reshape(dm.shape) if with_j else None
        vk = np.asarray(vk).reshape(dm.shape) if with_k else None
        return vj, vk

    def _jk(self, d, with_j, with_k):
        kfit, E, fi, ci = self.kfit, self.E, self.fi, self.ci
        nf = len(fi)
        dt = kfit._forward(np.where(self.oo, d, 0.0))
        j = k = None
        if with_j:
            rho = kfit.to_real(kfit.fit(dt))
            if nf:
                dfc = d[np.ix_(fi, ci)]
                rho += (2 * np.einsum("vfc,fc->v", E, dfc)
                        - np.einsum("vfg,fg->v", E[:, :, :nf], dfc[:, :nf]))
            j = np.where(self.oo, kfit._backward(kfit.potential(kfit.to_complex(rho))).real, 0.0)
            if nf:
                jx = np.einsum("vfc,v->fc", E, rho)
                j[np.ix_(ci, fi)] = jx.T
                j[np.ix_(fi, ci)] = jx
            if self.low is not None:
                dp = lib.pack_tril(d + d.T - np.diag(np.diag(d)))
                j += lib.unpack_tril((self.sign * (self.low @ dp)) @ self.low)
        if with_k:
            k = np.where(self.oo, kfit._backward(kfit.exchange(dt)).real, 0.0)
            if nf:
                k += self._k_fragment(d)
            if self.low is not None:
                w, c = np.linalg.eigh(d)
                keep = np.abs(w) > 1e-14 * max(1.0, np.abs(w).max())
                c, w = c[:, keep], w[keep]
                for x0 in range(0, len(self.low), self.batch):
                    x1 = min(x0 + self.batch, len(self.low))
                    t = np.matmul(lib.unpack_tril(self.low[x0:x1]), c)
                    tw = t * w * self.sign[x0:x1, None, None]
                    k += np.tensordot(tw, t, axes=((0, 2), (0, 2)))
        return j, k

    def _k_fragment(self, d):
        kfit, E, fi, ci = self.kfit, self.E, self.fi, self.ci
        nao, nf = d.shape[0], len(fi)
        co = ci[nf:]
        m = d[:, fi].copy()
        m[fi] = 0.0
        mh = kfit.rows_forward(m)
        dci = np.ascontiguousarray(d[:, ci])
        dff = d[np.ix_(fi, fi)]
        c_o = np.zeros((nao, len(co)))
        xff = np.zeros((nf, nf))
        xfo = np.zeros((nf, len(co)))
        xoo = np.zeros((len(co), len(co)))
        acc = np.zeros((kfit.nk, kfit.nao, nf), dtype=complex)
        zc = {}
        for p, kind, a, b in kfit.blocks:
            if kind != "im":
                y = kfit.apply(p, mh)
                y[:, fi] = 0.0
            yb = y.real if kind == "self" else np.sqrt(2) * (y.real if kind == "re" else y.imag)
            eb = E[a:b]
            nb = b - a
            eo = eb[:, :, nf:].reshape(nb * nf, -1)
            c_o += yb.transpose(1, 0, 2).reshape(nao, nb * nf) @ eo
            zb = (dci @ eb.reshape(nb * nf, -1).T).reshape(nao, nb, nf)
            zf = zb[ci].transpose(1, 0, 2).reshape(-1, nf)
            xff += eb.transpose(1, 0, 2).reshape(nf, -1) @ zf
            xfo += zb[fi].transpose(2, 1, 0).reshape(nf, nb * nf) @ eo
            xoo += eo.T @ np.matmul(dff, eb[:, :, nf:]).reshape(nb * nf, -1)
            w = zb.transpose(1, 0, 2).copy()
            w[:, fi] = 0.0
            if kind == "self":
                kfit.accumulate(p, w.astype(complex), acc)
            elif kind == "re":
                zc[p] = np.sqrt(2) * w
            else:
                kfit.accumulate(p, zc.pop(p) - 1j * np.sqrt(2) * w, acc)
        c_f = kfit.rows_backward(acc).real
        c_f[fi] = 0.0
        c = np.zeros((nao, nao))
        c[:, co] = c_o
        c[:, fi] = c_f
        k = c + c.T
        k[np.ix_(fi, fi)] += xff
        k[np.ix_(fi, co)] += xfo
        k[np.ix_(co, fi)] += xfo.T
        k[np.ix_(co, co)] += xoo
        return k
