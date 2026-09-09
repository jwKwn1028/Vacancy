import numpy as np

from pyscf import lib
from pyscf.scf import hf

from .frozen_active_space import FrozenActiveSpace


class Frozen_RHF(FrozenActiveSpace, hf.SCF):
    def kernel(self, conv_tol=1e-10, conv_tol_grad=None,
               start_rotation=None, start_occ=None):
        if conv_tol_grad is None:
            conv_tol_grad = np.sqrt(conv_tol)

        mo_coeff_active = self.mo_coeff[:, self.active_orb]
        mo_occ_active = self.mo_occ[self.active_orb]

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

        mo_occ = mo_occ_active
        mo_coeff = mo_coeff_active

        for cycle in range(self.max_cycle):
            last_hf_e = e_elec

            fock = self.get_fock(h1e, eye, vhf, np.diag(mo_occ), cycle)
            mo_energy, U = self.eig(fock, eye)
            mo_coeff = lib.einsum("ui, ij -> uj", mo_coeff, U)
            mo_occ = self.get_occ(mo_energy, mo_coeff)

            h1e = self.get_hcore(h1e, U)
            vhf, eri_active = self.get_veff(mo_occ, eri_active, U)
            e_elec = self.energy_elec(h1e, vhf, mo_occ)

            fock = self.get_fock(h1e, eye, vhf, np.diag(mo_occ))
            norm_gorb = np.linalg.norm(
                self.get_grad(
                    eye,
                    mo_occ,
                    self.get_fock(
                        h1e, eye, vhf, np.diag(mo_occ), cycle,
                        level_shift_factor=0.0,
                    ),
                )
            )

            if abs(e_elec - last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
                scf_conv = True
                break

        if scf_conv:
            mo_energy, U = self.eig(fock, eye)
            mo_occ = self.get_occ(mo_energy, mo_coeff)

            h1e = self.get_hcore(h1e, U)
            vhf, eri_active = self.get_veff(mo_occ, eri_active, U)
            e_elec = self.energy_elec(h1e, vhf, mo_occ)

        return [scf_conv, e_elec, mo_energy, mo_coeff, mo_occ, h1e, eri_active]

    def get_veff(self, mo_occ, eri=None, U=None):
        eri_active = self.eri_active if eri is None else self._rotate_eri(eri, U)
        vj = lib.einsum("ijkk, k -> ij", eri_active, mo_occ)
        vk = lib.einsum("ikkj, k -> ij", eri_active, mo_occ)
        return vj - 0.5 * vk, eri_active

    def energy_elec(self, h1e, vhf, mo_occ):
        e_1e = np.sum(np.diagonal(h1e) * mo_occ)
        e_2e = np.sum(np.diagonal(vhf) * mo_occ) * 0.5
        return e_1e + e_2e

    def get_occ(self, mo_energy, mo_coeff):
        e_idx = np.argsort(mo_energy)
        mo_occ = np.zeros_like(mo_energy)
        mo_occ[e_idx[: self.nelecas // 2]] = 2
        return mo_occ
