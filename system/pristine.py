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
):
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
    kmf.conv_tol = float(conv_tol)
    kmf.conv_tol_grad = float(conv_tol_grad)
    kmf.max_cycle = int(max_cycle)
    kmf.max_memory = int(max_memory)

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
    kmf.kernel(dm0=dm0)
    if not kmf.converged:
        raise RuntimeError("pristine KRHF did not converge")
    return cell, kmf
