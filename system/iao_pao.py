import numpy as np


def lowdin_invsqrt(metric, tol=1e-12, require_full_rank=False):
    metric = np.asarray(metric)
    w, v = np.linalg.eigh(metric)
    keep = w > tol
    if require_full_rank and not np.all(keep):
        raise np.linalg.LinAlgError(
            "projected PAO metric is rank deficient: "
            f"rank {int(np.sum(keep))} < {w.size}; "
            f"smallest eigenvalue {float(w.min()):.3e}"
        )
    return (v[:, keep] / np.sqrt(w[keep])) @ v[:, keep].conj().T


def partition_ao_labels(full_labels, reference_labels):
    full = [tuple(label) for label in full_labels]
    reference = [tuple(label) for label in reference_labels]
    full_set = set(full)
    missing = [label for label in reference if label not in full_set]
    if missing:
        raise ValueError(f"MINAO labels absent from the full AO basis: {missing[:3]}")
    if len(set(reference)) != len(reference):
        raise ValueError("MINAO AO labels are not unique")

    reference_set = set(reference)
    pao_indices = np.asarray(
        [idx for idx, label in enumerate(full) if label not in reference_set],
        dtype=int,
    )
    lo_atom_ids = np.asarray(
        [label[0] for label in reference] + [full[idx][0] for idx in pao_indices],
        dtype=int,
    )
    return pao_indices, lo_atom_ids


def atom_preserving_pao_complement(c_iao, overlap, pao_ao_indices, tol=1e-12):
    c_iao = np.asarray(c_iao)
    overlap = np.asarray(overlap)
    pao_ao_indices = np.asarray(pao_ao_indices, dtype=int)
    nao = overlap.shape[0]
    if overlap.shape != (nao, nao) or c_iao.shape[0] != nao:
        raise ValueError("incompatible AO dimensions in the IAO/PAO construction")
    if (pao_ao_indices.ndim != 1
            or len(np.unique(pao_ao_indices)) != pao_ao_indices.size
            or np.any(pao_ao_indices < 0)
            or np.any(pao_ao_indices >= nao)):
        raise ValueError("PAO AO indices must be unique and in range")

    projector_perp = np.eye(nao) - c_iao @ (c_iao.conj().T @ overlap)
    raw_pao = projector_perp[:, pao_ao_indices]
    pao_metric = raw_pao.conj().T @ overlap @ raw_pao
    c_pao = raw_pao @ lowdin_invsqrt(pao_metric, tol=tol, require_full_rank=True)

    orth_error = np.max(np.abs(
        c_pao.conj().T @ overlap @ c_pao - np.eye(c_pao.shape[1])
    )) if c_pao.shape[1] else 0.0
    cross_error = np.max(np.abs(
        c_iao.conj().T @ overlap @ c_pao
    )) if c_iao.shape[1] and c_pao.shape[1] else 0.0
    if max(orth_error, cross_error) > 1e-8:
        raise np.linalg.LinAlgError(
            "IAO/PAO orthogonalization failed: "
            f"PAO error={orth_error:.3e}, IAO-PAO error={cross_error:.3e}"
        )
    return c_pao


def iao_pao_supercell(scell, orbocc, ovlp=None, minao="minao", return_atom_ids=False):
    from pyscf.lo import iao

    if ovlp is None:
        ovlp = scell.pbc_intor("int1e_ovlp", hermi=1)
    S = np.asarray(ovlp).real
    nao = S.shape[0]
    orbocc = np.asarray(orbocc).real

    a = np.asarray(iao.iao(scell, orbocc, minao=minao)).real
    n_iao = a.shape[1]
    reference = iao.reference_mol(scell, minao)
    full_labels = scell.ao_labels(fmt=False)
    reference_labels = reference.ao_labels(fmt=False)
    if len(reference_labels) != n_iao:
        raise RuntimeError(
            f"IAO/reference size mismatch: {n_iao} columns vs "
            f"{len(reference_labels)} MINAO labels"
        )
    C_iao = a @ lowdin_invsqrt(a.conj().T @ S @ a)

    pao_ao_indices, lo_atom_ids = partition_ao_labels(full_labels, reference_labels)
    if pao_ao_indices.size != nao - n_iao:
        raise RuntimeError(
            f"PAO label count mismatch: {pao_ao_indices.size} != {nao - n_iao}"
        )
    C_pao = atom_preserving_pao_complement(C_iao, S, pao_ao_indices)

    C_ao_lo = np.hstack([C_iao, C_pao])
    if return_atom_ids:
        return C_ao_lo, n_iao, lo_atom_ids
    return C_ao_lo, n_iao
