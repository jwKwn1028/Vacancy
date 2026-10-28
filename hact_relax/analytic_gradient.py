"""Analytic nuclear gradient of the frozen-bath embedding energy (ground state),

    E = Tr[(h - V)(D_s + D_C)] + Tr[v^P Y] + E_cl + E_NN^P - E_NN^vac,
    Y = D_s + D_C/2 - D_A^P/2,   v^P = J[D^P] - K[D^P]/2   (periodic fit),

with E_cl from the cluster ERIs.  The active solution is stationary, so it is held fixed in
C_A; the chain is differentiated in reverse mode, the pristine density by a Z-vector.
"""
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg
from pyscf import lib

from system.iao_pao import partition_ao_labels


@dataclass
class Point:
    """One evaluated geometry: what the production surface built and accepted."""

    energy: float
    kmf: object
    solver: object
    active_coeff: np.ndarray
    active_occ: np.ndarray
    cluster_df: object = None    # production's cluster DF object (its Cholesky vectors)


def capture_point(surface, coords_bohr, reason="analytic-gradient-center"):
    """An accepted centre at ``coords_bohr``, with its pristine mean field, solver and
    cluster DF."""
    coords = np.asarray(coords_bohr, dtype=float)
    surface.cache.pop(surface._key(coords), None)
    energy = surface.energy(coords, reason, accept_center=True, keep_point=True)
    kept, surface.kept_centre = surface.kept_centre, None
    ref = surface.active_solution_reference
    solver = kept["solver"]
    return Point(float(energy), kept["kmf"], solver,
                 np.asarray(ref.active_coeff).real.copy(),
                 np.asarray(ref.active_occ).real.copy(), solver.cluster_df)


def _gamma(x):
    x = np.asarray(x)
    return x[0] if x.ndim == 3 else x


def invsqrt_eig(m):
    """m^(-1/2) of a symmetric positive definite matrix, with its eigensystem."""
    w, v = np.linalg.eigh(m)
    return (v / np.sqrt(w)) @ v.T, w, v


def divided_differences(f_w, w, df_w):
    """L_ij = (f_i - f_j)/(w_i - w_j), and f'(w_i) on the diagonal."""
    dw = w[:, None] - w[None, :]
    same = np.abs(dw) < 1e-10 * max(1.0, np.abs(w).max())
    # Degenerate pairs take the limit f'(w), as the diagonal does.
    return np.where(same, 0.5 * (df_w[:, None] + df_w[None, :]),
                    (f_w[:, None] - f_w[None, :]) / np.where(same, 1.0, dw))


def matfun_adjoint(v, ldd, fbar):
    """Adjoint of M -> f(M) (symmetric M = v diag(w) v^T) for the output adjoint fbar."""
    fs = 0.5 * (fbar + fbar.T)
    return v @ ((v.T @ fs @ v) * ldd) @ v.T


@dataclass
class Inputs:
    """The matrices the chain starts from, all at one geometry (Gamma, real)."""

    S: np.ndarray            # AO overlap, lattice summed
    S12: np.ndarray          # AO x MINAO overlap
    S2: np.ndarray           # MINAO overlap
    Co: np.ndarray           # pristine occupied orbitals (any orthonormal basis)
    hP: np.ndarray           # T + V_nuc of the pristine crystal
    vP: np.ndarray           # J[D^P] - K[D^P]/2 of the pristine density
    V: np.ndarray            # the vacancy nucleus' attraction, removed (Eq. 16)
    cluster_L: object        # cluster DF: ClusterDF (streamed) or a dense (naux, nao, nao)
    enuc: float              # E_NN^P - E_NN^vac
    pao_idx: np.ndarray      # AO columns of the PAOs
    frag_lo: np.ndarray      # fragment LOs (in C_lo column order)
    env_lo: np.ndarray       # environment LOs
    n_bath: int
    act_coeff: np.ndarray    # converged active orbitals (AO)
    act_occ: np.ndarray      # their occupations (0, 1, 2)


def minao_overlaps(cell, minao="minao"):
    from pyscf.lo import iao
    from pyscf.pbc import gto as pbcgto

    pmol = iao.reference_mol(cell, minao)
    s2 = np.asarray(pmol.pbc_intor("int1e_ovlp", hermi=1)).real
    s12 = np.asarray(pbcgto.cell.intor_cross("int1e_ovlp", cell, pmol)).real
    return s12, s2, pmol


def gather_inputs(point, cluster_L=None):
    """Inputs of the chain from a captured point (the solver's own matrices)."""
    solver = point.solver
    cell = solver.scell
    s12, s2, pmol = minao_overlaps(cell, solver.minao)
    pao_idx, lo_atom_ids = partition_ao_labels(cell.ao_labels(fmt=False),
                                               pmol.ao_labels(fmt=False))
    frag = np.isin(lo_atom_ids, sorted(solver.frag_atoms))
    if cluster_L is None:
        cluster_L = cluster_df_tensor(solver)
    return Inputs(
        S=np.asarray(solver.S), S12=s12, S2=s2, Co=np.asarray(solver._orbocc),
        hP=np.asarray(solver.hcore), vP=np.asarray(solver.vhf),
        V=solver._nuc_ao_integral_for_atom(), cluster_L=cluster_L,
        enuc=float(solver.Enuc), pao_idx=np.asarray(pao_idx),
        frag_lo=np.where(frag)[0], env_lo=np.where(~frag)[0],
        n_bath=int(solver.n_bath), act_coeff=point.active_coeff,
        act_occ=point.active_occ)


def cluster_mol(solver):
    from pyscf import gto as mol_gto

    scell = solver.scell
    coords = solver._defect_frame_coords()
    return mol_gto.M(atom=[[scell.atom_symbol(i), coords[i]] for i in range(scell.natm)],
                     unit="Bohr", basis=scell.basis, charge=0,
                     spin=int(round(sum(scell.atom_charges()))) % 2, verbose=0,
                     max_memory=scell.max_memory)


def cluster_df_tensor(solver):
    """The cluster DF tensor L[P, mu, nu] (molecular DF of the defect frame)."""
    from pyscf import df as mol_df

    mol = cluster_mol(solver)
    dfobj = mol_df.DF(mol)
    dfobj.build()
    nao = mol.nao_nr()
    return np.concatenate([lib.unpack_tril(np.asarray(x)).reshape(-1, nao, nao)
                           for x in dfobj.loop()])


def cluster_jk(L, d):
    """J and K of an AO density with the cluster DF tensor."""
    j = np.einsum("P,Pmn->mn", np.einsum("Pkl,kl->P", L, d), L)
    k = np.einsum("Pmk,kl,Pln->mn", L, d, L, optimize=True)
    return j, k


def _block_size(nao, factor, max_memory_mb, block_mb=None):
    """Auxiliary functions per block when each holds ``factor`` arrays of nao x nao."""
    mb = block_mb if block_mb is not None else max(256.0, max_memory_mb / 8)
    return max(1, int(mb * 1e6 / (8.0 * factor * nao * nao)))


class ClusterDF:
    """The cluster's molecular DF (defect frame), streamed over auxiliary blocks."""

    def __init__(self, solver, block_mb=None, dfobj=None):
        from pyscf import df as mol_df

        mol = cluster_mol(solver)
        if dfobj is not None and not (
                dfobj.mol.natm == mol.natm and dfobj.mol.nao_nr() == mol.nao_nr()
                and np.allclose(dfobj.mol.atom_coords(), mol.atom_coords(), atol=1e-10)):
            dfobj = None
        self.reused = dfobj is not None
        if dfobj is None:
            dfobj = mol_df.DF(mol)
            dfobj.build()
        self.df, self.mol = dfobj, dfobj.mol
        self.blksize = _block_size(self.mol.nao_nr(), 3, self.mol.max_memory, block_mb)

    def blocks(self):
        n = self.mol.nao_nr()
        for x in self.df.loop(self.blksize):
            yield lib.unpack_tril(np.asarray(x)).reshape(-1, n, n)

    def active_terms(self, C, xj, xks):
        """J of ``xj``, K of each of ``xks``, and C^T L_P C."""
        n = C.shape[0]
        j = np.zeros((n, n))
        ks = [np.zeros((n, n)) for _ in xks]
        act = []
        for lb in self.blocks():
            b = lb @ C
            a = np.matmul(C.T, b)
            act.append(a)
            j += np.tensordot(np.einsum("Pij,ij->P", a, xj), lb, axes=(0, 0))
            bt = b.transpose(1, 0, 2).reshape(n, -1)
            for k, x in zip(ks, xks):
                k += (b @ x).transpose(1, 0, 2).reshape(n, -1) @ bt.T
        return j, ks, np.concatenate(act)


def cluster_terms(cluster, C, xj, xks):
    """ClusterDF.active_terms, also for a dense cluster tensor."""
    if isinstance(cluster, ClusterDF):
        return cluster.active_terms(C, xj, xks)
    j, _ = cluster_jk(cluster, C @ xj @ C.T)
    ks = [cluster_jk(cluster, C @ x @ C.T)[1] for x in xks]
    return j, ks, np.einsum("Pmn,mi,nj->Pij", cluster, C, C, optimize=True)


# --------------------------------------------------------------------------- forward

@dataclass
class Chain:
    """The intermediates of the forward chain."""

    energy: float = 0.0
    terms: dict = field(default_factory=dict)
    vals: dict = field(default_factory=dict)


def _iao(S, S12, S2, Co):
    s1cd = scipy.linalg.cho_factor(S)
    s2cd = scipy.linalg.cho_factor(S2)
    P12 = scipy.linalg.cho_solve(s1cd, S12)
    W = S12 @ scipy.linalg.cho_solve(s2cd, S12.T)
    X = scipy.linalg.cho_solve(s1cd, W @ Co)
    fx, wx, vx = invsqrt_eig(X.T @ S @ X)
    Ct = X @ fx
    Oc = Co @ Co.T @ S
    Ot = Ct @ Ct.T @ S
    K = np.eye(len(S)) - Oc - Ot + 2 * Oc @ Ot
    A = K @ P12
    fa, wa, va = invsqrt_eig(A.T @ S @ A)
    return A @ fa, dict(P12=P12, W=W, X=X, fx=fx, wx=wx, vx=vx, Ct=Ct, Oc=Oc, Ot=Ot, K=K,
                        A=A, fa=fa, wa=wa, va=va)


def _pao(S, Ciao, pao_idx):
    n = len(S)
    Ep = np.eye(n)[:, pao_idx]
    raw = Ep - Ciao @ (Ciao.T @ S @ Ep)
    fm, wm, vm = invsqrt_eig(raw.T @ S @ raw)
    return raw @ fm, dict(Ep=Ep, raw=raw, fm=fm, wm=wm, vm=vm)


def _occupied_projector(g):
    """round(eigenvalues) applied to g, and the divided differences of that map."""
    w, v = np.linalg.eigh(g)
    n = np.round(w)
    return v @ np.diag(n) @ v.T, w, v, divided_differences(n, w, np.zeros_like(w))


def forward(inp, fixed=None, ref_U=None):
    """The energy at a fixed active solution, with its intermediates.  ``fixed`` = (d^a,
    d^b) in the basis C_A; ``ref_U`` fixes the singular vectors' signs."""
    S = inp.S
    Ciao, iao = _iao(S, inp.S12, inp.S2, inp.Co)
    Cpao, pao = _pao(S, Ciao, inp.pao_idx)
    Clo = np.hstack([Ciao, Cpao])
    DP = 2 * inp.Co @ inp.Co.T
    F, E = inp.frag_lo, inp.env_lo
    SC = S @ Clo
    gamma = SC.T @ DP @ SC
    PEF = gamma[np.ix_(E, F)] / 2
    U, s, Vt = np.linalg.svd(PEF, full_matrices=True)
    nb = inp.n_bath
    if ref_U is not None:
        sign = np.sign(np.einsum("ij,ij->j", ref_U[:, :len(s)], U[:, :len(s)]))
        U[:, :len(s)] *= sign
        Vt *= sign[:, None]
    UB, UC = U[:, :nb], U[:, nb:]
    CA = np.hstack([Clo[:, F], Clo[:, E] @ UB])
    CC = Clo[:, E] @ UC
    gAA = CA.T @ S @ DP @ S @ CA
    gCC = CC.T @ S @ DP @ S @ CC
    NA, wA, vA, LA = _occupied_projector(gAA)
    NC, wC, vC, LC = _occupied_projector(gCC)
    DAP = CA @ NA @ CA.T
    DC = CC @ NC @ CC.T

    if fixed is None:
        occ = inp.act_occ
        Da = inp.act_coeff[:, occ > 0] @ inp.act_coeff[:, occ > 0].T
        Db = inp.act_coeff[:, occ > 1.5] @ inp.act_coeff[:, occ > 1.5].T
        # The active solution in the basis C_A: fixed while the chain moves.
        SCA = S @ CA
        da, db = SCA.T @ Da @ SCA, SCA.T @ Db @ SCA
    else:
        da, db = fixed
    Da, Db = CA @ da @ CA.T, CA @ db @ CA.T
    Ds = Da + Db

    h = inp.hP - inp.V
    Y = Ds + 0.5 * DC - 0.5 * DAP
    delta = Ds - DAP
    # Every cluster density lies in span(C_A): Da, Db, DAP = C_A (da, db, NA) C_A^T.
    jd, (ka, kb, kn), bA = cluster_terms(inp.cluster_L, CA, da + db - NA, (da, db, NA))
    e_cl = (0.5 * np.vdot(jd, delta) - 0.5 * (np.vdot(ka, Da) + np.vdot(kb, Db))
            + 0.5 * np.vdot(kn, Ds) - 0.25 * np.vdot(kn, DAP))
    terms = {"one": float(np.vdot(h, Ds + DC)), "vP": float(np.vdot(inp.vP, Y)),
             "cluster": float(e_cl), "enuc": inp.enuc}
    out = Chain(energy=sum(terms.values()), terms=terms)
    out.vals.update(iao=iao, pao=pao, Ciao=Ciao, Cpao=Cpao, Clo=Clo, DP=DP, gamma=gamma,
                    PEF=PEF, U=U, s=s, Vt=Vt, UB=UB, UC=UC, CA=CA, CC=CC, gAA=gAA,
                    gCC=gCC, NA=NA, NC=NC, wA=wA, vA=vA, LA=LA, wC=wC, vC=vC, LC=LC,
                    DAP=DAP, DC=DC, da=da, db=db, Da=Da, Db=Db, Ds=Ds, Y=Y, delta=delta,
                    jd=jd, ka=ka, kb=kb, kn=kn, bA=bA, h=h)
    return out


# --------------------------------------------------------------------------- backward

@dataclass
class Adjoints:
    """dE/d(input) of the chain at fixed active solution (in the basis C_A)."""

    S: np.ndarray
    S12: np.ndarray
    S2: np.ndarray
    Co: np.ndarray
    hP: np.ndarray
    vP: np.ndarray
    V: np.ndarray
    L: np.ndarray = None
    bath_rotation: np.ndarray = None


def _svd_left_adjoint(U, s, Vt, Ubar, nb, within_bath):
    """dE/dP for P = U s V^T, given dE/dU; rotations within the complement, and within the
    bath unless ``within_bath``, are free."""
    nE = U.shape[0]
    nF = len(s)
    sf = np.zeros(nE)
    sf[:nF] = s
    m = U.T @ Ubar
    dm = m - m.T
    keep = np.ones((nE, nE), dtype=bool)
    keep[nb:, nb:] = False
    if not within_bath:
        keep[:nb, :nb] = False
    keep &= np.triu(np.ones((nE, nE), dtype=bool), 1)
    den = sf[None, :] ** 2 - sf[:, None] ** 2
    c = np.where(keep, dm / np.where(keep, den, 1.0), 0.0)
    a1 = c * sf[:, None]
    a2 = c * sf[None, :]
    return U @ a1[:nF, :].T @ Vt + U @ a2[:, :nF] @ Vt, c


def backward(inp, chain, within_bath=False, with_L=False):
    """Reverse mode through the chain: dE/d(each input) at fixed active solution."""
    v = chain.vals
    S, L = inp.S, inp.cluster_L
    F, E = inp.frag_lo, inp.env_lo
    nF = len(F)
    CA, CC, DP = v["CA"], v["CC"], v["DP"]
    Da, Db, Ds, DAP, DC = v["Da"], v["Db"], v["Ds"], v["DAP"], v["DC"]
    h = v["h"]
    jd, ka, kb, kn = v["jd"], v["ka"], v["kb"], v["kn"]
    ks = ka + kb

    # Densities.
    ea = h + inp.vP + jd - ka + 0.5 * kn
    eb = h + inp.vP + jd - kb + 0.5 * kn
    en = -0.5 * inp.vP - jd + 0.5 * ks - 0.5 * kn
    ec = h + 0.5 * inp.vP
    Sbar = np.zeros_like(S)
    DPbar = np.zeros_like(S)
    CAbar = 2 * (ea @ CA @ v["da"] + eb @ CA @ v["db"]) + 2 * en @ CA @ v["NA"]
    CCbar = 2 * ec @ CC @ v["NC"]
    for C, nbar_src, vec, ldd, Cbar_name in ((CA, en, v["vA"], v["LA"], "A"),
                                               (CC, ec, v["vC"], v["LC"], "C")):
        gbar = matfun_adjoint(vec, ldd, C.T @ nbar_src @ C)
        cb = 2 * S @ DP @ S @ C @ gbar
        if Cbar_name == "A":
            CAbar += cb
        else:
            CCbar += cb
        Z = C @ gbar @ C.T
        Sbar += Z @ S @ DP + DP @ S @ Z
        DPbar += S @ Z @ S

    # C_A = [C_lo,F, C_lo,E U_B], C_C = C_lo,E U_C.
    Clo = v["Clo"]
    UB, UC = v["UB"], v["UC"]
    Clobar = np.zeros_like(Clo)
    Clobar[:, F] += CAbar[:, :nF]
    Clobar[:, E] += CAbar[:, nF:] @ UB.T + CCbar @ UC.T
    Ubar = np.hstack([Clo[:, E].T @ CAbar[:, nF:], Clo[:, E].T @ CCbar])
    Pbar, rot = _svd_left_adjoint(v["U"], v["s"], v["Vt"], Ubar, inp.n_bath, within_bath)

    # gamma = (S C_lo)^T D^P (S C_lo), P_EF = gamma[E, F]/2.
    gbar = np.zeros((len(Clo.T), len(Clo.T)))
    gbar[np.ix_(E, F)] = 0.5 * Pbar
    gbar = 0.5 * (gbar + gbar.T)
    Clobar += 2 * S @ DP @ S @ Clo @ gbar
    Z = Clo @ gbar @ Clo.T
    Sbar += Z @ S @ DP + DP @ S @ Z
    DPbar += S @ Z @ S

    # PAO.
    niao = v["Ciao"].shape[1]
    Ciao = v["Ciao"]
    pao = v["pao"]
    raw, fm = pao["raw"], pao["fm"]
    Cpaobar = Clobar[:, niao:]
    rawbar = Cpaobar @ fm
    wm = pao["wm"]
    Mbar = matfun_adjoint(pao["vm"], divided_differences(wm ** -0.5, wm, -0.5 * wm ** -1.5),
                          raw.T @ Cpaobar)
    rawbar += 2 * S @ raw @ Mbar
    Sbar += raw @ Mbar @ raw.T
    Ep = pao["Ep"]
    Ciaobar = Clobar[:, :niao] - rawbar @ Ep.T @ S @ Ciao - S @ Ep @ rawbar.T @ Ciao
    Sbar += -Ciao @ Ciao.T @ rawbar @ Ep.T

    # IAO.
    iao = v["iao"]
    A, fa, wa = iao["A"], iao["fa"], iao["wa"]
    Abar = Ciaobar @ fa
    Mbar = matfun_adjoint(iao["va"], divided_differences(wa ** -0.5, wa, -0.5 * wa ** -1.5),
                          A.T @ Ciaobar)
    Abar += 2 * S @ A @ Mbar
    Sbar += A @ Mbar @ A.T
    P12, K, Oc, Ot, Ct, X = iao["P12"], iao["K"], iao["Oc"], iao["Ot"], iao["Ct"], iao["X"]
    Kbar = Abar @ P12.T
    P12bar = K.T @ Abar
    Obar = -Kbar + 2 * Kbar @ Ot.T
    Otbar = -Kbar + 2 * Oc.T @ Kbar
    Co = inp.Co
    Cobar = Obar @ S @ Co + S @ Obar.T @ Co
    Sbar += Co @ Co.T @ Obar
    Ctbar = Otbar @ S @ Ct + S @ Otbar.T @ Ct
    Sbar += Ct @ Ct.T @ Otbar
    fx, wx = iao["fx"], iao["wx"]
    Xbar = Ctbar @ fx
    Mbar = matfun_adjoint(iao["vx"], divided_differences(wx ** -0.5, wx, -0.5 * wx ** -1.5),
                          X.T @ Ctbar)
    Xbar += 2 * S @ X @ Mbar
    Sbar += X @ Mbar @ X.T
    Sinv = np.linalg.inv(S)
    W = iao["W"]
    Cobar += W @ Sinv @ Xbar
    Wbar = Sinv @ Xbar @ Co.T
    Sbar += -Sinv @ Xbar @ X.T
    S2inv = np.linalg.inv(inp.S2)
    S12 = inp.S12
    S12bar = (Wbar + Wbar.T) @ S12 @ S2inv + Sinv @ P12bar
    S2bar = -S2inv @ S12.T @ Wbar @ S12 @ S2inv
    Sbar += -Sinv @ P12bar @ P12.T

    # D^P = 2 Co Co^T.
    Cobar += 2 * (DPbar + DPbar.T) @ Co

    Lbar = None
    if with_L:
        if not isinstance(L, np.ndarray):
            raise ValueError("with_L needs a dense cluster tensor")
        delta = v["delta"]
        Lbar = np.zeros_like(L)
        jx = np.einsum("Pkl,kl->P", L, delta)
        for P in range(len(L)):
            lp = L[P]
            Lbar[P] = (jx[P] * delta
                       - 0.5 * (Da @ lp @ Da + Da @ lp @ Da + Db @ lp @ Db + Db @ lp @ Db)
                       + 0.5 * (Ds @ lp @ DAP + DAP @ lp @ Ds)
                       - 0.25 * 2 * DAP @ lp @ DAP)
    P1 = Ds + DC
    return Adjoints(S=Sbar, S12=S12bar, S2=S2bar, Co=Cobar, hP=P1, vP=v["Y"], V=-P1,
                    L=Lbar, bath_rotation=rot)


# --------------------------------------------------------------------------- Z-vector

@dataclass
class Response:
    """The pristine RHF's orbital response, folded into one Z-vector solve."""

    z: np.ndarray            # (nvir, nocc)
    Pz: np.ndarray           # C_v z C_o^T + transpose
    W: np.ndarray            # contracted with S^x (with a minus sign)
    Co: np.ndarray           # canonical occupied orbitals
    Cv: np.ndarray
    residual: float = 0.0        # max |A z + b|
    residual_norm: float = 0.0
    iterations: int = 0          # Krylov iterations, one fit J/K each


def pristine_g(kmf):
    """G[X] = J[X] - K[X]/2 of the pristine fit, for a real symmetric AO matrix X."""
    def g(x):
        vj, vk = kmf.get_jk(kmf.cell, np.asarray(x)[None], hermi=1)
        return (np.asarray(vj)[0] - 0.5 * np.asarray(vk)[0]).real
    return g


def zvector(S, fock, Co_chain, Cobar_chain, g, tol=1e-10, max_cycle=100):
    """The pristine RHF Z-vector, A z = -C_v^T Cobar."""
    from pyscf.scf import cphf

    e, C = scipy.linalg.eigh(fock, S)
    nocc = Co_chain.shape[1]
    Co, Cv = C[:, :nocc], C[:, nocc:]
    eo = e[:nocc]
    Q = Co.T @ S @ Co_chain
    if abs(Q @ Q.T - np.eye(nocc)).max() > 1e-8:
        raise RuntimeError("the pristine occupied space is not the aufbau one")
    Cobar = Cobar_chain @ Q.T
    nv = Cv.shape[1]
    b = Cv.T @ Cobar

    calls = [0]

    def fvind(x):
        calls[0] += 1
        out = []
        for xi in np.asarray(x).reshape(-1, nv, nocc):
            p = Cv @ xi @ Co.T
            out.append(2 * Cv.T @ g(p + p.T) @ Co)
        return np.asarray(out)

    occ = np.r_[np.full(nocc, 2.0), np.zeros(nv)]
    z = cphf.solve(fvind, e, occ, b, tol=tol, max_cycle=max_cycle)[0]
    iterations = calls[0]
    resid = (e[nocc:, None] - eo[None, :]) * z + fvind(z)[0] + b
    p = Cv @ z @ Co.T
    Pz = p + p.T
    gp = g(Pz)
    q = Cv @ z @ np.diag(eo) @ Co.T
    oo = Co.T @ Cobar
    W = 0.5 * (q + q.T) + Co @ (Co.T @ gp @ Co) @ Co.T + 0.25 * Co @ (oo + oo.T) @ Co.T
    return Response(z=z, Pz=Pz, W=W, Co=Co, Cv=Cv, residual=float(abs(resid).max()),
                    residual_norm=float(np.linalg.norm(resid)), iterations=iterations)


# --------------------------------------------------------------------------- stationarity

def active_hamiltonian(inp, chain):
    """h1e and DF ERIs in the converged active orbitals."""
    v = chain.vals
    C, CA = inp.act_coeff, v["CA"]
    # The active orbitals span C_A, so C^T L_P C comes from forward's active tensor.
    U = CA.T @ inp.S @ C
    r = C - CA @ U
    if abs(r.T @ inp.S @ r).max() > 1e-12:
        raise RuntimeError("the active orbitals leave span(C_A)")
    LC = np.einsum("Pij,ia,jb->Pab", v["bA"], U, U, optimize=True)
    jn = np.tensordot(np.einsum("Pij,ij->P", v["bA"], v["NA"]), LC, axes=(0, 0))
    h1e = C.T @ (v["h"] + inp.vP + 0.5 * v["kn"]) @ C - jn
    from frozen_bath_scf.df_eri import DFEri

    return h1e, DFEri(LC)


def polish_active(h1e, eri, occ, tol=1e-11, max_cycle=6):
    """Newton steps to an active orbital gradient below ``tol``; the rotation and a record.
    """
    from frozen_bath_scf import second_order as so
    from frozen_bath_scf.orbital_optimizer import energy_at, newton_finish

    e0 = energy_at(h1e, eri, occ)
    u, h1e_new, eri_new, newton = newton_finish(h1e, eri, occ, tol, max_cycle)
    lowest = newton.get("newton_lowest_hessian")
    if lowest is None:
        lowest = float(so.lowest_hessian_modes(h1e, eri, occ)[0][0])
    h1e, eri = h1e_new, eri_new
    return u, {"gradient_before": newton["newton_gradient_before"], "lowest_hessian": lowest,
               "cycles": newton["newton_cycles"], "stop": newton["newton_stop"],
               "gradient_after": newton["newton_gradient_after"],
               "energy_change": float(energy_at(h1e, eri, occ) - e0)}


# --------------------------------------------------------------------------- integrals
# d/dR_A at fixed M; PySCF's ip integrals are <grad mu|...>, hence the minus signs.

def _slices(cell, atoms):
    s = cell.aoslice_by_atom()
    return [(int(s[a][2]), int(s[a][3])) for a in atoms]


def grad_symmetric(ip, m, slices):
    """sum_munu M_munu d<mu|op|nu>/dR_A for an integral symmetric in mu, nu."""
    ms = m + m.T
    return np.array([-np.einsum("xij,ij->x", ip[:, p0:p1], ms[p0:p1]) for p0, p1 in slices])


def grad_overlap(cell, m, atoms):
    return grad_symmetric(np.asarray(cell.pbc_intor("int1e_ipovlp", comp=3)), m,
                          _slices(cell, atoms))


def grad_kinetic(cell, m, atoms):
    return grad_symmetric(np.asarray(cell.pbc_intor("int1e_ipkin", comp=3)), m,
                          _slices(cell, atoms))


def grad_minao(cell, pmol, m12, m2, atoms):
    """The MINAO overlaps' terms: sum M12 dS12 + sum M2 dS2."""
    from pyscf.pbc import gto as pbcgto

    ip_ao = np.asarray(pbcgto.cell.intor_cross("int1e_ipovlp", cell, pmol, comp=3))
    ip_mn = np.asarray(pbcgto.cell.intor_cross("int1e_ipovlp", pmol, cell, comp=3))
    ao, mn = _slices(cell, atoms), _slices(pmol, atoms)
    out = np.zeros((len(atoms), 3))
    for k, ((a0, a1), (m0, m1)) in enumerate(zip(ao, mn)):
        out[k] -= np.einsum("xij,ij->x", ip_ao[:, a0:a1], m12[a0:a1])
        out[k] -= np.einsum("xij,ji->x", ip_mn[:, m0:m1], m12[:, m0:m1])
    out += grad_symmetric(np.asarray(pmol.pbc_intor("int1e_ipovlp", comp=3)), m2, mn)
    return out


def grad_vacancy_attraction(solver, m, atoms):
    """sum M dV for V = -Z_v <mu|1/|r - R_v||nu> (the vacancy site does not move)."""
    from pyscf.data.elements import charge

    mol = solver._defect_frame_mol() if solver.wrap_about_vacancy else solver.scell
    out = np.zeros((len(atoms), 3))
    for idx in solver.vac_idx_sc:
        z = charge(solver.scell.atom_symbol(idx))
        with mol.with_rinv_at_nucleus(idx):
            ip = np.asarray(mol.intor("int1e_iprinv", comp=3))
        out += -z * grad_symmetric(ip, m, _slices(solver.scell, atoms))
    return out


def grad_vacancy_repulsion(solver, atoms):
    """d/dR of the vacancy nucleus' repulsion with the other atoms (Eq. 19)."""
    from pyscf.data.elements import charge

    coords = solver._defect_frame_coords()
    out = np.zeros((len(atoms), 3))
    for v in solver.vac_idx_sc:
        zv = charge(solver.scell.atom_symbol(v))
        for k, a in enumerate(atoms):
            if a == v:
                continue
            r = coords[a] - coords[v]
            out[k] -= zv * charge(solver.scell.atom_symbol(a)) * r / np.linalg.norm(r) ** 3
    return out


def grad_ewald(cell, atoms):
    from pyscf.pbc.grad import rhf as pbc_grad

    return np.asarray(pbc_grad.grad_nuc(cell))[list(atoms)]


def _aux_shell_blocks(auxmol, size, sh0=0, sh1=None):
    """Consecutive auxiliary shell ranges of at most about ``size`` functions."""
    loc = auxmol.ao_loc_nr()
    sh1 = auxmol.nbas if sh1 is None else sh1
    start = sh0
    for sh in range(sh0, sh1):
        if sh > start and loc[sh + 1] - loc[start] > size:
            yield start, sh
            start = sh
    if start < sh1:
        yield start, sh1


def grad_cluster(mol, CA, xa, xb, xn, atoms, block_mb=None, bA=None, auxmol=None):
    """d/dR of E_cl at fixed densities, streamed over auxiliary shell blocks; ``bA`` (C_A^T
    L C_A) saves the undifferentiated pass."""
    from pyscf import df as mol_df
    from pyscf.df import incore

    if auxmol is None:
        auxmol = mol_df.addons.make_auxmol(mol, mol_df.addons.make_auxbasis(mol))
    loc = auxmol.ao_loc_nr()
    n = mol.nao_nr()
    size = _block_size(n, 4, mol.max_memory, block_mb)

    def three_centre(intor, comp, rows, s0, s1):
        return incore.aux_e2(mol, auxmol, intor=intor, aosym="s1", comp=comp,
                             shls_slice=rows + (s0, s1))

    allrows = (0, mol.nbas, 0, mol.nbas)
    j2c = auxmol.intor("int2c2e", hermi=1)
    if bA is None:
        b = np.empty((auxmol.nao_nr(),) + (CA.shape[1],) * 2)
        for s0, s1 in _aux_shell_blocks(auxmol, size):
            j3c = three_centre("int3c2e", 1, allrows, s0, s1)          # (N, N, blk)
            b[loc[s0]:loc[s1]] = np.einsum("mi,mnP,nj->Pij", CA, j3c, CA, optimize=True)
        del j3c
        ct = np.einsum("PQ,Qij->Pij", np.linalg.inv(j2c), b)
    else:
        # PySCF's eigen-decomposition fallback is not differentiated here.
        try:
            low = scipy.linalg.cholesky(j2c, lower=True)
        except scipy.linalg.LinAlgError as exc:
            raise RuntimeError("the cluster DF metric is not positive definite: the energy "
                               "used PySCF's eigen-decomposition fallback") from exc
        ct = scipy.linalg.solve_triangular(low, bA.reshape(len(bA), -1), lower=True,
                                           trans="T").reshape(bA.shape)
    del j2c
    xs = xa + xb
    dl = xs - xn
    cd = np.einsum("Pij,ij->P", ct, dl)
    pairs = ((xa, xa, -0.5), (xb, xb, -0.5), (xs, xn, 0.5), (xn, xn, -0.25))
    g = np.einsum("P,ij->Pij", cd, dl)
    w = 0.5 * np.outer(cd, cd)
    for x, y, c in pairs:
        g += c * (np.matmul(np.matmul(x, ct), y) + np.matmul(np.matmul(y, ct), x))
        w += c * np.einsum("Pij,jk,Qkl,li->PQ", ct, x, ct, y, optimize=True)
    out = np.zeros((len(atoms), 3))
    for k, a in enumerate(atoms):
        sh = mol.aoslice_by_atom()[a]
        ca = CA[sh[2]:sh[3]]
        for s0, s1 in _aux_shell_blocks(auxmol, size):
            ip1 = three_centre("int3c2e_ip1", 3, (sh[0], sh[1], 0, mol.nbas), s0, s1)
            ga = np.einsum("mi,Pij,nj->mnP", ca, g[loc[s0]:loc[s1]], CA, optimize=True)
            out[k] -= 2 * np.einsum("xmnP,mnP->x", ip1, ga)
        ash = auxmol.aoslice_by_atom()[a]
        for s0, s1 in _aux_shell_blocks(auxmol, size, ash[0], ash[1]):
            ip2 = three_centre("int3c2e_ip2", 3, allrows, s0, s1)        # (3, N, N, blk)
            out[k] -= np.einsum("xmnP,mi,Pij,nj->x", ip2, CA, g[loc[s0]:loc[s1]], CA,
                                optimize=True)
        p0, p1 = ash[2], ash[3]
        ipj = auxmol.intor("int2c2e_ip1", comp=3, shls_slice=(ash[0], ash[1], 0, auxmol.nbas))
        out[k] += np.einsum("xPQ,PQ->x", ipj, w[p0:p1] + w.T[p0:p1])
    return out


# --------------------------------------------------------------------------- assembly

@dataclass
class Gradient:
    total: np.ndarray                     # (natm_moving, 3)
    parts: dict
    timings: dict
    zvector_residual: float
    stationarity: dict = field(default_factory=dict)
    zvector: dict = field(default_factory=dict)
    multipliers: dict = field(default_factory=dict)


def analytic_gradient(point, movable, axes=(0, 1, 2), polish=True, block_mb=None,
                      zvector_tol=1e-10, keep_multipliers=False):
    """dE/dR of ``movable``, which the fit must treat as moved.  ``polish`` makes the active
    solution stationary first; ``keep_multipliers`` keeps each term's contraction matrices.
    """
    import time

    from system import periodic_derivatives as pd

    timings = {}
    t0 = time.perf_counter()
    kmf, solver = point.kmf, point.solver
    fit = kmf.with_df
    if not hasattr(fit, "kfit"):
        raise NotImplementedError("analytic gradient: the pristine host must use a FragmentFit")
    missing = sorted(set(movable) - set(fit.moved))
    if missing:
        raise ValueError("atoms %s are not explicit in the fragment fit" % missing)
    cell = solver.scell
    cluster = ClusterDF(solver, block_mb, point.cluster_df)
    inp = gather_inputs(point, cluster_L=cluster)
    chain = forward(inp)
    if abs(chain.energy - point.energy) > 1e-7:
        raise RuntimeError("forward chain %.10f != production %.10f"
                           % (chain.energy, point.energy))
    stationarity = {"forward_minus_production": float(chain.energy - point.energy)}
    if polish:
        u, info = polish_active(*active_hamiltonian(inp, chain), inp.act_occ)
        stationarity.update(info)
        if info["cycles"]:
            inp.act_coeff = inp.act_coeff @ u
            chain = forward(inp)
    v = chain.vals
    adj = backward(inp, chain)
    timings["chain"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    g = pristine_g(kmf)
    cobar = adj.Co + 4 * g(v["Y"]) @ inp.Co
    resp = zvector(inp.S, solver.hcore + solver.vhf, inp.Co, cobar, g, tol=zvector_tol)
    timings["zvector"] = time.perf_counter() - t0

    ms = 0.5 * (adj.S + adj.S.T) - resp.W
    mh = adj.hP + 0.5 * resp.Pz
    my = v["Y"] + 0.5 * resp.Pz
    atoms = sorted(int(i) for i in movable)
    parts = {}
    t0 = time.perf_counter()
    _, _, pmol = minao_overlaps(cell, solver.minao)
    parts["overlap"] = grad_overlap(cell, ms, atoms)
    parts["kinetic"] = grad_kinetic(cell, mh, atoms)
    parts["minao"] = grad_minao(cell, pmol, adj.S12, adj.S2, atoms)
    parts["vacancy_attraction"] = grad_vacancy_attraction(solver, adj.V, atoms)
    parts["vacancy_repulsion"] = -grad_vacancy_repulsion(solver, atoms)
    parts["ewald"] = grad_ewald(cell, atoms)
    timings["one_electron"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    SCA = inp.S @ v["CA"]
    parts["cluster"] = grad_cluster(cluster.mol, v["CA"], v["da"], v["db"],
                                    SCA.T @ v["DAP"] @ SCA, atoms, block_mb, bA=v["bA"],
                                    auxmol=cluster.df.auxmol)
    timings["cluster"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    der = pd.FitDerivative(fit)
    timings["fit_setup"] = time.perf_counter() - t0
    fitg = np.zeros((len(atoms), 3))
    nucg = np.zeros((len(atoms), 3))
    timings["fit_integrals"] = timings["nuclear_rows"] = 0.0
    for k, a in enumerate(atoms):
        t = time.perf_counter()
        ints = der.atom_integrals(a)
        timings["fit_integrals"] += time.perf_counter() - t
        fitg[k] = der.fit_gradient(a, ints, my, v["DP"], axes=axes)
        t = time.perf_counter()
        nucg[k] = der.nuclear_rows_gradient(a, ints, mh)
        timings["nuclear_rows"] += time.perf_counter() - t
        del ints
    timings["fit_vectors"] = der.timings["vector_derivatives"]
    timings["fit_pm_jk"] = der.timings["bilinear"]
    timings["fit_and_nuclear_rows"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    field = pd.nucleus_field(cell, mh, atoms, fit.settings)
    nucg -= cell.atom_charges()[atoms][:, None] * field
    timings["nucleus_field"] = time.perf_counter() - t0
    parts["fit"] = fitg
    parts["nuclear"] = nucg
    total = sum(parts.values())
    mask = np.zeros(3, dtype=bool)
    mask[list(axes)] = True
    multipliers = {}
    if keep_multipliers:
        multipliers = {"S": ms, "hP": mh, "fit": my, "S12": adj.S12, "S2": adj.S2,
                       "V": adj.V, "Da": v["Da"], "Db": v["Db"], "DAP": v["DAP"],
                       "DP": v["DP"]}
    return Gradient(total=np.where(mask, total, 0.0), parts=parts, timings=timings,
                    zvector_residual=resp.residual, stationarity=stationarity,
                    zvector={"tol": zvector_tol, "iterations": resp.iterations,
                             "residual_max": resp.residual,
                             "residual_norm": resp.residual_norm,
                             "fit_pm_jk_calls": der.timings["bilinear_calls"]},
                    multipliers=multipliers)
