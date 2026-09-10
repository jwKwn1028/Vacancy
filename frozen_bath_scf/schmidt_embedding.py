import numpy as np

from pyscf import lib
from pyscf import gto as mol_gto
from pyscf import df as mol_df
from pyscf.data.elements import charge as element_charge
from pyscf.pbc import df as pdf
from pyscf.pbc.tools import k2gamma
from pyscf.scf import rohf

from system.iao_pao import iao_pao_supercell

from .frozen_rhf import Frozen_RHF
from .frozen_rohf import Frozen_ROHF


def _sq(x):
    x = np.asarray(x)
    return x[0] if x.ndim == 3 else x


class _SchmidtEmbedding:

    def _configure(self, kmf, kmesh, *, vac_species, n_frag, charge=0,
                   minao="minao", bath_tol=1e-8, max_cycle=200, level_shift=0.0,
                   verbose=0, defect=True, eri_mode="periodic",
                   seed_vacancy=False, vacancy_index=None,
                   fixed_n_bath=None, fixed_fragment_atoms=None,
                   compute_core_energy=True, auxbasis=None):
        self.kmf = kmf
        self.kmesh = list(kmesh)
        self.vac_species = vac_species
        self.n_frag = n_frag
        self.charge = charge
        self.minao = minao
        self.bath_tol = bath_tol
        self.max_cycle = max_cycle
        self.level_shift = level_shift
        self.verbose = verbose
        self.defect = defect
        self.eri_mode = eri_mode
        self.seed_vacancy = seed_vacancy
        self.vacancy_index = vacancy_index
        self.fixed_n_bath = fixed_n_bath
        self.fixed_fragment_atoms = fixed_fragment_atoms
        self.compute_core_energy = bool(compute_core_energy)
        self.auxbasis = auxbasis

        self.mo_coeff = self.mo_occ = self.active_orb = None
        self.mo_occ_D = None
        self.nelecas = 0
        core_initial = 0 if self.compute_core_energy else None
        self.E1e_core = self.E2e_core = self.Enuc = core_initial
        self.eri_active = None
        self._c_def = None

    def build(self):
        self._build_supercell()
        self._build_localized_orbitals()
        self._build_active_space()
        self.get_active_occ()
        self.get_h1e()
        self.get_g2e()
        self.heff = self.h1e_lo_lo + self.vbath_lo_lo
        self.enuc = self.get_enuc() if self.compute_core_energy else None
        if not self.compute_core_energy:
            self.Enuc = None
        self._seed_initial_occ()

    def _reusable_pristine_hcore(self, scell, gdf, *, gamma_full_cell):
        if not gamma_full_cell:
            return None

        ecpbas = getattr(scell, "_ecpbas", ())
        if getattr(scell, "pseudo", None) or (ecpbas is not None and len(ecpbas)):
            return None

        cache = getattr(self.kmf, "_embedding_pristine_hcore_cache", None)
        if cache is None:
            return None
        if not isinstance(cache, dict) or cache.get("version") != 1:
            raise RuntimeError("unrecognized pristine hcore cache")
        if cache.get("cell") is not scell:
            raise RuntimeError("pristine hcore cache belongs to a different cell")
        if cache.get("with_df") is not gdf:
            raise RuntimeError(
                "pristine hcore cache belongs to a different density fit"
            )

        kpts = np.asarray(self.kmf.kpts)
        cached_kpts = np.asarray(cache.get("kpts"))
        if kpts.shape != (1, 3) or not np.allclose(kpts, 0.0, atol=1e-12):
            return None
        if cached_kpts.shape != kpts.shape or not np.array_equal(cached_kpts, kpts):
            raise RuntimeError("pristine hcore cache k-points changed")

        snapshots = (
            ("lattice", cache.get("lattice_vectors"), scell.lattice_vectors()),
            ("coordinates", cache.get("atom_coords"), scell.atom_coords()),
            ("nuclear charges", cache.get("atom_charges"), scell.atom_charges()),
        )
        for name, cached, current in snapshots:
            if cached is None or not np.array_equal(
                np.asarray(cached), np.asarray(current)
            ):
                raise RuntimeError("pristine hcore cache %s changed" % name)

        nao = int(scell.nao_nr())
        if cache.get("nao") != nao:
            raise RuntimeError("pristine hcore cache AO dimension changed")
        hcore = _sq(cache.get("hcore"))
        if hcore.shape != (nao, nao) or not np.all(np.isfinite(hcore)):
            raise RuntimeError(
                "pristine hcore cache has invalid shape or non-finite values"
            )
        return np.asarray(hcore).real

    def _build_supercell(self):
        gamma_full_cell = tuple(int(x) for x in self.kmesh) == (1, 1, 1)
        mf_sc = self.kmf if gamma_full_cell else k2gamma.k2gamma(
            self.kmf, kmesh=self.kmesh
        )
        scell = mf_sc.cell
        self.scell = scell
        self.vacancy = scell
        self.ncells = int(np.prod(self.kmesh))

        if gamma_full_cell and getattr(self.kmf, "with_df", None):
            gdf = self.kmf.with_df
            reused = getattr(gdf, "auxbasis", None)
            if self.auxbasis is not None and reused != self.auxbasis:
                raise ValueError(
                    "pristine KRHF was density-fitted with auxbasis=%r but the "
                    "embedding was given auxbasis=%r"
                    % (reused, self.auxbasis)
                )
            if getattr(gdf, "_cderi", None) is None:
                gdf.build()
        else:
            gdf = pdf.GDF(scell)
            gdf.auxbasis = self.auxbasis
            gdf.build()
        self.gdf = gdf

        mo = _sq(mf_sc.mo_coeff).real
        occ = np.asarray(mf_sc.mo_occ)
        if occ.ndim == 2 and occ.shape[0] == 1:
            occ = occ[0]
        self.S = np.asarray(scell.pbc_intor("int1e_ovlp", hermi=1)).real

        orbocc = mo[:, occ > 0]
        self._orbocc = orbocc

        cached_hcore = self._reusable_pristine_hcore(
            scell, gdf, gamma_full_cell=gamma_full_cell
        )
        if cached_hcore is not None:
            self.hcore = np.array(cached_hcore, copy=True).real
        else:
            kin = np.asarray(scell.pbc_intor("int1e_kin", hermi=1)).real
            nuc = _sq(gdf.get_nuc()).real
            self.hcore = kin + nuc

        dm = (orbocc * occ[occ > 0]) @ orbocc.T
        self._dm_ao = dm
        vj, vk = gdf.get_jk(dm, hermi=1, exxdiv=None)
        self.vhf = (_sq(vj) - 0.5 * _sq(vk)).real
        self.Enuc_P = scell.energy_nuc() if self.compute_core_energy else None

    def _build_localized_orbitals(self):
        C_lo, n_iao, lo_atom_ids = iao_pao_supercell(
            self.scell, self._orbocc, ovlp=self.S, minao=self.minao,
            return_atom_ids=True,
        )
        self.C_lo = C_lo
        self.n_iao = n_iao
        self.lo_atom_ids = lo_atom_ids
        SC = self.S @ C_lo
        self.gamma_lo = SC.T @ self._dm_ao @ SC

    def _build_active_space(self):
        frag_mask = self._fragment_lo_mask()
        gamma = self.gamma_lo
        nlo = gamma.shape[0]

        fidx = np.where(frag_mask)[0]
        eidx = np.where(~frag_mask)[0]
        vacancy_los = np.where(
            np.isin(self.lo_atom_ids, self.fragment_center_idx)
        )[0]
        missing_vacancy_los = np.setdiff1d(vacancy_los, fidx)
        if vacancy_los.size == 0 or missing_vacancy_los.size:
            raise RuntimeError(
                "vacancy-centred LOs were excluded from the fragment side of "
                "the Schmidt SVD: vacancy_los=%s missing=%s"
                % (vacancy_los.tolist(), missing_vacancy_los.tolist())
            )
        perm = np.concatenate([fidx, eidx])
        nF = len(fidx)
        g = gamma[np.ix_(perm, perm)]

        P = g / 2.0
        P_EF = P[nF:, :nF]
        if P_EF.shape[1] != nF:
            raise RuntimeError("Schmidt SVD lost fragment columns")
        U, s, _ = np.linalg.svd(P_EF, full_matrices=True)
        if self.fixed_n_bath is None:
            n_bath = int(np.sum(s > self.bath_tol))
        else:
            n_bath = int(self.fixed_n_bath)
            if not 0 <= n_bath <= U.shape[1]:
                raise ValueError(
                    "fixed_n_bath=%d does not fit the %d-orbital environment"
                    % (n_bath, U.shape[1])
                )

        U_lo_eo = np.eye(nlo)
        U_lo_eo[nF:, nF:] = U
        C_eo = self.C_lo[:, perm] @ U_lo_eo
        g_eo = U_lo_eo.T @ g @ U_lo_eo

        active_eo = np.zeros(nlo, dtype=bool)
        active_eo[:nF] = True
        active_eo[nF:nF + n_bath] = True

        mo_coeff, mo_occ, active_mask = self._natural_orbitals(
            C_eo, g_eo, active_eo
        )

        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self._mo_occ_build = mo_occ.copy()
        self.active_orb = active_mask
        self.n_frag = nF
        self.n_bath = n_bath
        self._validate_active_space()

    def _validate_active_space(self):
        return None

    @staticmethod
    def _natural_orbitals(C_eo, g_eo, active_eo):
        nlo = g_eo.shape[0]
        core_eo = ~active_eo
        mo_coeff = np.empty_like(C_eo)
        mo_occ = np.zeros(nlo)
        active_mask = np.zeros(nlo, dtype=bool)

        wa, va = np.linalg.eigh(g_eo[np.ix_(active_eo, active_eo)])
        wc, vc = np.linalg.eigh(g_eo[np.ix_(core_eo, core_eo)])
        n_act = int(active_eo.sum())

        mo_coeff[:, :n_act] = C_eo[:, active_eo] @ va
        mo_coeff[:, n_act:] = C_eo[:, core_eo] @ vc
        mo_occ[:n_act] = np.round(wa)
        mo_occ[n_act:] = np.round(wc)
        active_mask[:n_act] = True
        return mo_coeff, mo_occ, active_mask

    def _fragment_lo_mask(self):
        scell = self.scell
        coords = scell.atom_coords()
        x = coords[:, 0]

        species = np.array([scell.atom_symbol(i) for i in range(scell.natm)])
        requested = self.vacancy_index
        if requested is None:
            center_x = 0.5 * (x.min() + x.max())
            cand = np.where(species == self.vac_species)[0]
            if cand.size == 0:
                raise ValueError(
                    f"vac_species {self.vac_species!r} not found in supercell"
                )
            vac_idx = [int(cand[np.argmin(np.abs(x[cand] - center_x))])]
        else:
            requested = int(requested)
            if not 0 <= requested < scell.natm:
                raise IndexError(
                    f"vacancy_index={requested} outside supercell with "
                    f"{scell.natm} atoms"
                )
            if species[requested] != self.vac_species:
                raise ValueError(
                    f"vacancy_index={requested} is {species[requested]!r}, "
                    f"expected vac_species={self.vac_species!r}"
                )
            vac_idx = [requested]
        self.fragment_center_idx = vac_idx
        self.vac_idx_sc = vac_idx if self.defect else []

        if self.fixed_fragment_atoms is None:
            order = np.argsort(np.abs(x - x[vac_idx[0]]))
            frag_atoms = set(int(a) for a in order[: self.n_frag])
        else:
            frag_atoms = set(int(a) for a in self.fixed_fragment_atoms)
            invalid = sorted(a for a in frag_atoms if not 0 <= a < scell.natm)
            if invalid or len(frag_atoms) != self.n_frag:
                raise ValueError(
                    "fixed fragment atoms %s do not define n_frag=%d atoms"
                    % (sorted(frag_atoms), self.n_frag)
                )
        if not set(vac_idx).issubset(frag_atoms):
            raise RuntimeError(
                "vacancy site was excluded from fragment atoms: vacancy=%s "
                "fragment=%s" % (vac_idx, sorted(frag_atoms))
            )
        self.frag_atoms = frag_atoms
        mask = self._fragment_mask_from_atoms(frag_atoms)
        vacancy_lo_mask = np.isin(self.lo_atom_ids, vac_idx)
        if not np.any(vacancy_lo_mask) or not np.all(mask[vacancy_lo_mask]):
            raise RuntimeError(
                "vacancy-centred localized orbitals were excluded from the "
                "fragment mask"
            )
        return mask

    def _fragment_mask_from_atoms(self, frag_atoms):
        frag_atoms = np.asarray(sorted(frag_atoms), dtype=int)
        mask = np.isin(self.lo_atom_ids, frag_atoms)
        aoslice = self.scell.aoslice_by_atom()
        expected = int(sum(aoslice[at][3] - aoslice[at][2] for at in frag_atoms))
        actual = int(np.sum(mask))
        if actual != expected:
            raise RuntimeError(
                "atom-labelled fragment LO count is inconsistent with the AO "
                f"basis: got {actual}, expected {expected} for atoms "
                f"{frag_atoms.tolist()}"
            )
        return mask

    def get_eri_act(self):
        Cact = np.asarray(self.mo_coeff[:, self.active_orb]).real
        if self.eri_mode == "cluster":
            self.eri_active = self._eri_act_cluster(Cact)
            return

        nao = self.scell.nao_nr()
        Lij_chunks = []
        for LpqR, LpqI, sign in self.gdf.sr_loop(
            [np.zeros(3), np.zeros(3)], compact=False
        ):
            Lpq = (LpqR + 1j * LpqI).reshape(-1, nao, nao)
            Lij_chunks.append(lib.einsum("Lpq,pi,qj->Lij", Lpq, Cact, Cact))
        Lij = np.concatenate(Lij_chunks, axis=0)
        self.eri_active = lib.einsum("Lij,Lkl->ijkl", Lij, Lij).real

    def _eri_act_cluster(self, Cact):
        scell = self.scell
        mol = mol_gto.M(
            atom=[[scell.atom_symbol(i), scell.atom_coord(i)]
                  for i in range(scell.natm)],
            unit="Bohr",
            basis=scell.basis,
            charge=0,
            spin=int(round(sum(scell.atom_charges()))) % 2,
            verbose=0,
            max_memory=scell.max_memory,
        )
        if mol.nao_nr() != scell.nao_nr():
            raise ValueError(
                "molecular-cluster AO count (%d) != supercell AO count (%d); "
                "C_lo cannot be reused.  eri_mode='cluster' assumes an "
                "all-electron basis matching the supercell."
                % (mol.nao_nr(), scell.nao_nr())
            )

        dfobj = mol_df.DF(mol)
        dfobj.build()
        nao = mol.nao_nr()
        Lij_chunks = []
        for Lpq in dfobj.loop():
            Lpq = lib.unpack_tril(np.asarray(Lpq)).reshape(-1, nao, nao)
            Lij_chunks.append(lib.einsum("Lpq,pi,qj->Lij", Lpq, Cact, Cact))
        Lij = np.concatenate(Lij_chunks, axis=0)
        return lib.einsum("Lij,Lkl->ijkl", Lij, Lij).real

    def _nuc_ao_integral_for_atom(self, symmetric=True):
        nao = self.scell.nao_nr()
        h1e_nuc = np.zeros((nao, nao))
        for idx in self.vac_idx_sc:
            Z = element_charge(self.scell.atom_symbol(idx))
            with self.scell.with_rinv_at_nucleus(idx):
                V = (self.scell.intor_symmetric("int1e_rinv") if symmetric
                     else self.scell.intor("int1e_rinv"))
            h1e_nuc += -Z * V
        return h1e_nuc

    def _nuclear_repulsion_with_atom(self):
        coords = self.scell.atom_coords()
        Z = [element_charge(self.scell.atom_symbol(i)) for i in range(self.scell.natm)]
        vac = set(self.vac_idx_sc)
        Enuc_at = 0.0
        for ivac in self.vac_idx_sc:
            for iat in range(self.scell.natm):
                if iat == ivac:
                    continue
                r = np.linalg.norm(coords[ivac] - coords[iat])
                term = Z[ivac] * Z[iat] / r
                Enuc_at += 0.5 * term if iat in vac else term
        return Enuc_at

    def _vac_charge_sum(self):
        return sum(element_charge(self.scell.atom_symbol(i)) for i in self.vac_idx_sc)

    def _n_active_electrons(self):
        n_core = int(round(np.sum(self.mo_occ[~self.active_orb])))
        n_total_vac = self.scell.nelectron - self._vac_charge_sum() - self.charge
        return int(round(n_total_vac - n_core))

    def _vacancy_populations(self, c_act):
        c_act = np.asarray(c_act).real
        SC = self.S @ c_act
        aoslice = self.scell.aoslice_by_atom()
        pops = np.zeros(c_act.shape[1])
        for idx in self.vac_idx_sc:
            s0, s1 = aoslice[idx][2], aoslice[idx][3]
            pops += np.einsum("up,up->p", c_act[s0:s1], SC[s0:s1])
        return pops

    def _seed_defect_orbital(self, c_act, pop_tol=0.1):
        pops = self._vacancy_populations(c_act)
        vac_orb = int(np.argmax(pops))
        if pops[vac_orb] < pop_tol:
            self._c_def = None
            return None
        self._c_def = np.asarray(c_act).real[:, vac_orb].copy()
        return vac_orb

    def _track_defect_orbital(self, c_act):
        if self._c_def is None:
            return None
        c_act = np.asarray(c_act).real
        ovlp = np.abs(self._c_def @ (self.S @ c_act))
        vac_orb = int(np.argmax(ovlp))
        self._c_def = c_act[:, vac_orb].copy()
        return vac_orb

    def _pristine_seeded_occ_indices(self, n_take):
        e_diag = np.diag(self.heff)
        pristine = np.asarray(self._mo_occ_build)[self.active_orb]
        occupied = [int(i) for i in np.where(pristine > 0)[0]]
        empty = [int(i) for i in np.where(pristine <= 0)[0]]
        n_take = int(n_take)
        if n_take <= 0:
            return np.zeros(0, dtype=int)
        if len(occupied) > n_take:
            occupied.sort(key=lambda i: e_diag[i])
            occupied = occupied[:n_take]
        elif len(occupied) < n_take:
            empty.sort(key=lambda i: e_diag[i])
            occupied = occupied + empty[: n_take - len(occupied)]
        return np.asarray(sorted(occupied), dtype=int)

    def _occ_indices_with_vacancy(self, energies, vac_orb, n_take):
        order_e = np.argsort(energies)
        if vac_orb is None:
            return order_e[:n_take]
        chosen = [int(vac_orb)]
        for i in order_e:
            if len(chosen) >= n_take:
                break
            if int(i) != vac_orb:
                chosen.append(int(i))
        return np.array(chosen, dtype=int)


class SchmidtEmbeddedRHF(_SchmidtEmbedding, Frozen_RHF):

    def __init__(self, kmf, kmesh, vac_species, n_frag, **kwargs):
        rohf.ROHF.__init__(self, kmf.cell)
        self._configure(kmf, kmesh, vac_species=vac_species, n_frag=n_frag,
                        **kwargs)
        self.build()

    def get_active_occ(self):
        self.nelecas = self._n_active_electrons()

    def _seed_initial_occ(self):
        nocc = self.nelecas // 2
        if self.seed_vacancy and nocc > 0 and self.vac_idx_sc:
            e_diag = np.diag(self.heff)
            vac_orb = self._seed_defect_orbital(
                self.mo_coeff[:, self.active_orb]
            )
            occ = np.zeros(e_diag.size)
            occ[self._occ_indices_with_vacancy(e_diag, vac_orb, nocc)] = 2
        else:
            occ = np.zeros(int(np.sum(self.active_orb)))
            occ[self._pristine_seeded_occ_indices(nocc)] = 2
        self.mo_occ[self.active_orb] = occ

    def get_occ(self, mo_energy, mo_coeff):
        if not self.seed_vacancy or not self.vac_idx_sc:
            return super().get_occ(mo_energy, mo_coeff)
        nocc = self.nelecas // 2
        mo_occ = np.zeros_like(mo_energy)
        if nocc > 0:
            vac_orb = self._track_defect_orbital(mo_coeff)
            mo_occ[self._occ_indices_with_vacancy(mo_energy, vac_orb, nocc)] = 2
        return mo_occ


class SchmidtEmbeddedROHF(_SchmidtEmbedding, Frozen_ROHF):

    def __init__(self, kmf, kmesh, vac_species, n_frag, spin=1,
                 level_shift=0.4, **kwargs):
        rohf.ROHF.__init__(self, kmf.cell)
        self._configure(kmf, kmesh, vac_species=vac_species, n_frag=n_frag,
                        level_shift=level_shift, **kwargs)
        self.spin = spin
        self.build()

    def get_active_occ(self):
        nelec = self._n_active_electrons()
        self.nelecas = [(nelec + self.spin) // 2, (nelec - self.spin) // 2]

    def _seed_initial_occ(self):
        ncore, nopen = min(self.nelecas), abs(self.nelecas[0] - self.nelecas[1])
        e_diag = np.diag(self.heff)
        if self.seed_vacancy and (ncore + nopen) > 0 and self.vac_idx_sc:
            vac_orb = self._seed_defect_orbital(
                self.mo_coeff[:, self.active_orb]
            )
            idx = self._occ_indices_with_vacancy(e_diag, vac_orb, ncore + nopen)
        else:
            idx = self._pristine_seeded_occ_indices(ncore + nopen)
        idx = idx[np.argsort(e_diag[idx])]
        occ = np.zeros(e_diag.size)
        occ[idx[:ncore]] = 2
        occ[idx[ncore:ncore + nopen]] = 1
        self.mo_occ_D = self.mo_occ.copy()
        self.mo_occ_D[self.active_orb] = occ

    def get_occ(self, mo_energy, mo_coeff):
        if not self.seed_vacancy or not self.vac_idx_sc:
            return super().get_occ(mo_energy, mo_coeff)
        if self.nelecas[0] > self.nelecas[1]:
            nocc, ncore = self.nelecas
        else:
            ncore, nocc = self.nelecas
        nopen = nocc - ncore
        mo_occ = np.zeros_like(mo_energy)
        vac_orb = self._track_defect_orbital(mo_coeff)
        idx = self._occ_indices_with_vacancy(mo_energy, vac_orb, ncore + nopen)
        idx = idx[np.argsort(mo_energy[idx])]
        mo_occ[idx[:ncore]] = 2
        mo_occ[idx[ncore:ncore + nopen]] = 1
        return mo_occ
