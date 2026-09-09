import numpy as np

from pyscf import lib


class FrozenActiveSpace:
    def get_h1e(self):
        h1e_ao_ao = self.hcore.copy()
        h1e_ao_ao -= self._nuc_ao_integral_for_atom()

        C_active = self.mo_coeff[:, self.active_orb]
        if self.compute_core_energy:
            C_core = self.mo_coeff[:, ~self.active_orb]
            h1e_core = lib.einsum("iu, uv, vj -> ij", C_core.T, h1e_ao_ao, C_core)
            self.E1e_core = np.sum(
                np.diagonal(h1e_core) * self.mo_occ[~self.active_orb]
            )
        else:
            self.E1e_core = None
        self.h1e_lo_lo = lib.einsum(
            "iu, uv, vj -> ij", C_active.T, h1e_ao_ao, C_active
        )

    def get_g2e(self):
        self.get_eri_act()

        vhf_ao_ao_P = self.vhf
        if self.compute_core_energy:
            vhf_mo_mo_P = lib.einsum(
                "iu, uv, vj -> ij", self.mo_coeff.T, vhf_ao_ao_P, self.mo_coeff
            )
            vhf_active_P = vhf_mo_mo_P[np.ix_(self.active_orb, self.active_orb)]
            E2e_pristine = np.sum(np.diagonal(vhf_mo_mo_P) * self.mo_occ * 0.5)
        else:
            C_active = self.mo_coeff[:, self.active_orb]
            vhf_active_P = lib.einsum(
                "iu, uv, vj -> ij", C_active.T, vhf_ao_ao_P, C_active
            )

        occ_active = self.mo_occ[self.active_orb]
        vhf_mo_mo_active = (
            lib.einsum("ijkk, k -> ij", self.eri_active, occ_active)
            - 0.5 * lib.einsum("ikkj, k -> ij", self.eri_active, occ_active)
        )
        vhf_mo_mo_int = vhf_active_P - vhf_mo_mo_active

        self.vbath_lo_lo = vhf_mo_mo_int
        if self.compute_core_energy:
            E2e_active = np.sum(
                np.diagonal(vhf_mo_mo_active) * occ_active * 0.5
            )
            E2e_int = np.sum(np.diagonal(vhf_mo_mo_int) * occ_active)
            self.E2e_core = E2e_pristine - E2e_active - E2e_int
        else:
            self.E2e_core = None

    def get_enuc(self):
        self.Enuc = self.Enuc_P - self._nuclear_repulsion_with_atom()
        return self.Enuc

    def get_hcore(self, h1e=None, U=None):
        if h1e is None:
            return self.heff
        h1e = lib.einsum("kl, lj -> kj", h1e, U)
        return lib.einsum("ik, kj -> ij", U.T, h1e)

    @staticmethod
    def _rotate_eri(eri, U):
        eri = lib.einsum("ir, rspq -> ispq", U.T, eri)
        eri = lib.einsum("js, ispq -> ijpq", U.T, eri)
        eri = lib.einsum("ijpq, pk -> ijkq", eri, U)
        return lib.einsum("ijkq, ql -> ijkl", eri, U)

    def _start_frame(self, mo_coeff_active, start_rotation, start_occ):
        U0 = np.asarray(start_rotation, dtype=float)
        nact = mo_coeff_active.shape[1]
        if U0.shape != (nact, nact):
            raise ValueError(
                "start_rotation has shape %s, expected (%d, %d)"
                % (U0.shape, nact, nact)
            )
        if start_occ is None or np.asarray(start_occ).size != nact:
            raise ValueError(
                "start_rotation requires a matching start_occ of length %d" % nact
            )
        mo_coeff_active = lib.einsum("ui, ij -> uj", mo_coeff_active, U0)
        mo_occ_active = np.asarray(start_occ, dtype=float).copy()
        h1e = self.get_hcore(self.heff, U0)
        vhf, eri_active = self.get_veff(mo_occ_active, self.eri_active, U0)
        return mo_coeff_active, mo_occ_active, h1e, vhf, eri_active
