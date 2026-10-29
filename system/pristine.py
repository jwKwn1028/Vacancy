import os

import numpy as np


_PRISTINE_HCORE_CACHE_ATTR = "_embedding_pristine_hcore_cache"


def _install_pristine_hcore_capture(kmf):
    previous_post_kernel = kmf.post_kernel

    def capture_hcore(environment):
        hcore = environment.get("h1e")
        if hcore is not None:
            cell = kmf.cell
            setattr(
                kmf,
                _PRISTINE_HCORE_CACHE_ATTR,
                {
                    "version": 1,
                    "hcore": np.array(hcore, copy=True),
                    "cell": cell,
                    "with_df": getattr(kmf, "with_df", None),
                    "kpts": np.array(kmf.kpts, copy=True),
                    "lattice_vectors": np.array(cell.lattice_vectors(), copy=True),
                    "atom_coords": np.array(cell.atom_coords(), copy=True),
                    "atom_charges": np.array(cell.atom_charges(), copy=True),
                    "nao": int(cell.nao_nr()),
                },
            )
        return previous_post_kernel(environment)

    kmf.post_kernel = capture_hcore


def plateau_convergence(conv_tol, conv_tol_grad, factor, cycles=3):
    """PySCF check_convergence: the usual test, or a gradient stalled within ``factor`` of
    ``conv_tol_grad`` after ``cycles`` cycles of converged energy (a numerical floor, e.g.
    hBN 9x9's Gamma host).  ``check.floor`` is the gradient accepted that way."""
    history = []

    def check(envs):
        de, g = abs(envs["e_tot"] - envs["last_hf_e"]), float(envs["norm_gorb"])
        tol, gtol = envs["conv_tol"], envs["conv_tol_grad"]
        if envs["mf"].cycles:
            # The extra cycle after convergence: PySCF's own looser test.
            return de < tol or g < gtol
        if de < tol and g < gtol:
            return True
        history.append((de, g))
        window = history[-cycles:]
        if (len(window) == cycles and all(d < conv_tol for d, _ in window)
                and all(x < factor * conv_tol_grad for _, x in window)
                and window[-1][1] > 0.9 * window[0][1]):
            check.floor = g
            return True
        return False

    check.floor = None
    return check


def build_pristine_mean_field(
    atoms,
    lattice,
    basis,
    *,
    kmesh=(1, 1, 1),
    pseudo=None,
    charge=0,
    spin=0,
    exxdiv=None,
    verbose=0,
    max_memory=4000,
    conv_tol=1e-10,
    conv_tol_grad=1e-7,
    max_cycle=200,
    dm0=None,
    auxbasis=None,
    callback=None,
    chkfile=None,
    split=None,
    fit_records=None,
    fit_precision=1e-12,
    fit_explicit=(),
    floor_factor=None,
):
    """The pristine KRHF. ``split`` = (outside, atoms) replaces the Gamma GDF: a
    KpointOutside gives the split fit, a KpointFit the fragment fit (``fit_precision``,
    ``fit_explicit``). ``fit_records`` collects the GDF's metric factors for a KpointFit.
    ``floor_factor`` accepts a stalled gradient (plateau_convergence); the gradient
    accepted that way is ``kmf.converged_at_floor``."""
    from pyscf.pbc import gto, scf

    cell = gto.Cell()
    cell.atom = atoms
    cell.a = np.asarray(lattice, dtype=float)
    cell.unit = "Bohr"
    cell.basis = basis
    cell.pseudo = pseudo
    cell.charge = int(charge)
    cell.spin = int(spin)
    cell.verbose = int(verbose)
    cell.max_memory = int(max_memory)
    cell.build()

    kpts = cell.make_kpts(list(kmesh))
    kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv).density_fit(auxbasis=auxbasis)
    if split is not None:
        from system.fragment_fit import FragmentFit, KpointFit
        from system.split_fit import SplitDensityFit, fragment_aos

        if len(kpts) != 1:
            raise ValueError("split integrals need the Gamma supercell")
        outside, atoms = split
        if isinstance(outside, KpointFit):
            kmf.with_df = FragmentFit(outside, cell, atoms, precision=fit_precision,
                                      explicit=fit_explicit)
        else:
            kmf.with_df = SplitDensityFit(kmf.with_df, outside, fragment_aos(cell, atoms))
    kmf.conv_tol = float(conv_tol)
    kmf.conv_tol_grad = float(conv_tol_grad)
    kmf.max_cycle = int(max_cycle)
    kmf.max_memory = int(max_memory)
    if floor_factor:
        kmf.check_convergence = plateau_convergence(kmf.conv_tol, kmf.conv_tol_grad,
                                                    float(floor_factor))

    if dm0 is not None:
        dm0 = np.asarray(dm0)
        expected = (len(kpts), cell.nao_nr(), cell.nao_nr())
        if dm0.shape == expected[1:] and len(kpts) == 1:
            dm0 = dm0.reshape(expected)
        if dm0.shape != expected:
            raise ValueError(
                "initial KRHF density has shape %s, expected %s"
                % (dm0.shape, expected)
            )

    _install_pristine_hcore_capture(kmf)
    if callback is not None:
        kmf.callback = callback
    if fit_records is None:
        kmf.kernel(dm0=dm0)
    else:
        from pyscf import lib
        from pyscf.pbc.df import df as pbc_df
        from system.fragment_fit import recording_builder

        with lib.temporary_env(pbc_df, _RSGDFBuilder=recording_builder(fit_records)):
            kmf.kernel(dm0=dm0)
    kmf.converged_at_floor = getattr(kmf.check_convergence, "floor", None)
    if not kmf.converged:
        raise RuntimeError("pristine KRHF did not converge")
    if chkfile is not None:
        kmf.dump_chk(os.fspath(chkfile))
    return cell, kmf
