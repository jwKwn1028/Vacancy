"""CASCI on the embedded active-space Hamiltonian.

``H_act^V`` (defect.pdf Eq. 14) is handed to us as an MO-basis one-electron matrix
``h1e``, the active two-electron integrals ``eri``, and the core energy
``E1e_core + E2e_core + Enuc``.  PySCF's CASCI wants a mean-field object, so we wrap
those in a dummy ``scf.RHF`` whose "AO" basis already IS the relaxed active-MO basis.

Ported verbatim from the sibling repository's ``frozen_bath_scf/nv_casci.py``.
"""
import numpy as np

from pyscf import gto, scf, ao2mo, mcscf, fci


def _fake_mf(h1e, eri, ecore, n_active_elec):
    """Wrap an MO-basis active Hamiltonian (h1e, eri, ecore) in a mean field
    that PySCF CASCI can consume.  mo_coeff = identity: the 'AO' basis of this
    fake system already IS the relaxed active-MO basis.

    ``energy_nuc`` returns ``ecore``, so every ``mc.e_tot`` is already the TOTAL
    energy of the defect model -- nothing is added afterwards.
    """
    nact = h1e.shape[0]
    mol = gto.M(verbose=0)
    mol.nelectron = int(n_active_elec)
    mol.incore_anyway = True

    fmf = scf.RHF(mol)
    fmf.get_hcore = lambda *args: h1e
    fmf.get_ovlp = lambda *args: np.eye(nact)
    fmf._eri = ao2mo.restore(8, np.asarray(eri), nact)
    fmf.energy_nuc = lambda *args: ecore
    fmf.mo_coeff = np.eye(nact)
    occ = np.zeros(nact)
    occ[: n_active_elec // 2] = 2.0                 # aufbau doubly-occupied guess
    fmf.mo_occ = occ
    return fmf


def _run_casci(fmf, ncas, ncas_elec, two_s, nroots, ncore):
    """One spin-pure CASCI (fixed S) returning the per-root total energies,
    <S^2> diagnostics, and CI vectors.

    Note that the CI vector SHAPE is spin-sector dependent -- it is
    ``(n_alpha_strings, n_beta_strings)`` -- so vectors from different ``two_s``
    are not comparable.  Callers that overlap CI vectors across geometries must
    hold ``two_s`` fixed.
    """
    na = (ncas_elec + two_s) // 2
    nb = (ncas_elec - two_s) // 2
    mc = mcscf.CASCI(fmf, ncas, (na, nb))
    mc.ncore = ncore
    mc.fcisolver = fci.direct_spin1.FCI()
    mc.fcisolver.nroots = nroots
    # Pin the spin to S(S+1) so triplet/singlet roots do not mix or drop out.
    # A large shift keeps wrong-spin contaminants well out of the low-lying roots.
    fci.addons.fix_spin_(mc.fcisolver, shift=1.0, ss=two_s / 2.0 * (two_s / 2.0 + 1.0))
    mc.kernel()

    e_tot = np.atleast_1d(mc.e_tot)
    ci = mc.ci if nroots > 1 else [mc.ci]
    ss = [fci.spin_op.spin_square(c, ncas, (na, nb))[0] for c in ci]
    return [(e, s, c, na, nb) for e, s, c in zip(e_tot.tolist(), ss, ci)]
