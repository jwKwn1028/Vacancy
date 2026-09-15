import numpy as np

from pyscf import lib
from pyscf.scf import hf, rohf

from .frozen_active_space import FrozenActiveSpace


class Frozen_ROHF(FrozenActiveSpace, rohf.ROHF):
    def kernel(self, conv_tol=1e-8, conv_tol_grad=None,
               start_rotation=None, start_occ=None):
        if conv_tol_grad is None:
            conv_tol_grad = np.sqrt(conv_tol)

        mo_coeff_active = self.mo_coeff[:, self.active_orb]
        mo_occ_active = self.mo_occ_D[self.active_orb]

        if start_rotation is None:
            h1e = self.get_hcore()
            vhf, eri_active = self.get_veff(mo_occ_active)
        else:
            (
                mo_coeff_active,
                mo_occ_active,
                h1e,
                vhf,
                eri_active,
            ) = self._start_frame(mo_coeff_active, start_rotation, start_occ)

        e_elec = self.energy_elec(h1e, vhf, mo_occ_active)
        scf_conv = False
        eye = np.eye(h1e.shape[0])

        if self.max_cycle <= 0:
            fock = self.get_fock(h1e, eye, vhf, np.diag(mo_occ_active))
            mo_energy_active, U = self.eig(fock, eye)
            mo_coeff_active = lib.einsum("ui, ij -> uj", mo_coeff_active, U)
            mo_occ_active = self.get_occ(mo_energy_active, mo_coeff_active)
            return (scf_conv, e_elec, mo_energy_active, mo_coeff_active,
                    mo_occ_active, None, None)

        if isinstance(self.diis, lib.diis.DIIS):
            mf_diis = self.diis
        elif self.diis:
            mf_diis = self.DIIS(self, self.diis_file)
            mf_diis.space = self.diis_space
            mf_diis.rollback = self.diis_space_rollback
            mf_diis.damp = self.diis_damp
        else:
            mf_diis = None

        mo_occ = mo_occ_active
        mo_coeff = mo_coeff_active
        Cacc = eye.copy()

        for cycle in range(self.max_cycle):
            last_hf_e = e_elec

            dm = self.make_rdm1(mo_occ)
            fock = self.get_fock(h1e, eye, vhf, dm, cycle, level_shift_factor=0.0)
            focka, fockb = fock.focka, fock.fockb
            dm_tot = dm[0] + dm[1]

            if mf_diis is not None and cycle >= self.diis_start_cycle:
                f0 = Cacc.dot(np.asarray(fock)).dot(Cacc.conj().T)
                d0 = Cacc.dot(dm_tot).dot(Cacc.conj().T)
                fock = Cacc.conj().T.dot(mf_diis.update(eye, d0, f0)).dot(Cacc)

            if abs(self.level_shift) > 1e-4:
                fock = hf.level_shift(eye, dm_tot * .5, np.asarray(fock),
                                      self.level_shift)
            fock = lib.tag_array(np.asarray(fock), focka=focka, fockb=fockb)

            mo_energy, U = self.eig(fock, eye)
            mo_coeff = lib.einsum("ui, ij -> uj", mo_coeff, U)
            Cacc = Cacc.dot(U)
            mo_occ = self.get_occ(mo_energy, mo_coeff)

            h1e = self.get_hcore(h1e, U)
            vhf, eri_active = self.get_veff(mo_occ, eri_active, U)
            e_elec = self.energy_elec(h1e, vhf, mo_occ)

            dm = self.make_rdm1(mo_occ)
            fock = self.get_fock(h1e, eye, vhf, dm, cycle, level_shift_factor=0.0)
            norm_gorb = np.linalg.norm(self.get_grad(eye, mo_occ, fock))

            if self.scf_callback is not None:
                self.scf_callback(cycle + 1, e_elec, e_elec - last_hf_e, norm_gorb)

            if abs(e_elec - last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
                scf_conv = True
                break

        if scf_conv:
            mo_energy, U = self.eig(fock, eye)
            mo_coeff = lib.einsum("ui, ij -> uj", mo_coeff, U)
            mo_occ = self.get_occ(mo_energy, mo_coeff)

            h1e = self.get_hcore(h1e, U)
            vhf, eri_active = self.get_veff(mo_occ, eri_active, U)
            e_elec = self.energy_elec(h1e, vhf, mo_occ)

        return [scf_conv, e_elec, mo_energy, mo_coeff, mo_occ, h1e, eri_active]

    def get_veff(self, mo_occ, eri=None, U=None):
        eri_active = self.eri_active if eri is None else self._rotate_eri(eri, U)

        mo_occ_a = (mo_occ > 0).astype(np.double)
        mo_occ_b = (mo_occ == 2).astype(np.double)

        vhf_a = (
            lib.einsum("ijkk, k -> ij", eri_active, mo_occ_a)
            + lib.einsum("ijkk, k -> ij", eri_active, mo_occ_b)
            - lib.einsum("ikkj, k -> ij", eri_active, mo_occ_a)
        )
        vhf_b = (
            lib.einsum("ijkk, k -> ij", eri_active, mo_occ_b)
            + lib.einsum("ijkk, k -> ij", eri_active, mo_occ_a)
            - lib.einsum("ikkj, k -> ij", eri_active, mo_occ_b)
        )
        return [vhf_a, vhf_b], eri_active

    def energy_elec(self, h1e, vhf, mo_occ):
        mo_occa = (mo_occ > 0).astype(np.double)
        mo_occb = (mo_occ == 2).astype(np.double)
        e_1e = (np.sum(np.diagonal(h1e) * mo_occa)
                + np.sum(np.diagonal(h1e) * mo_occb))
        e_2e = (np.sum(np.diagonal(vhf[0]) * mo_occa)
                + np.sum(np.diagonal(vhf[1]) * mo_occb)) * 0.5
        return e_1e + e_2e

    def get_occ(self, mo_energy, mo_coeff):
        if self.nelecas[0] > self.nelecas[1]:
            nocc, ncore = self.nelecas
        else:
            ncore, nocc = self.nelecas
        return self._fill_rohf_occ(mo_energy, mo_energy, ncore, nocc - ncore)

    def _fill_rohf_occ(self, mo_energy, mo_energy_a, ncore, nopen):
        mo_occ = np.zeros_like(mo_energy)
        open_idx = []
        core_sort = np.argsort(mo_energy)
        core_idx = core_sort[:ncore]
        if nopen > 0:
            open_idx = core_sort[ncore:]
            open_sort = np.argsort(mo_energy_a[open_idx])
            open_idx = open_idx[open_sort[:nopen]]
        mo_occ[core_idx] = 2
        mo_occ[open_idx] = 1
        return mo_occ

    def make_rdm1(self, mo_occ):
        return [np.diag((mo_occ > 0).astype(np.double)),
                np.diag((mo_occ == 2).astype(np.double))]
