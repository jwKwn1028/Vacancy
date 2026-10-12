import itertools

import numpy as np


SHELL_TOL_BOHR = 1e-4

_NEIGHBOUR_SHIFTS = np.array(list(itertools.product((-1, 0, 1), repeat=3)))


def minimum_image_offsets(coords, lattice, center):
    """Vector from atom ``center`` to the nearest image of every atom.

    ``coords`` and ``lattice`` (rows are lattice vectors) share one length unit.
    Rounding the fractional coordinates only lands in the cell centred on the
    atom; for a skewed lattice such as hBN's 60 degree one that is not the
    nearest image near the cell's corners, so the neighbouring images are
    searched too.
    """
    coords = np.asarray(coords, dtype=float)
    lattice = np.asarray(lattice, dtype=float)
    frac = np.linalg.solve(lattice.T, (coords - coords[int(center)]).T).T
    frac -= np.round(frac)
    candidates = (frac[:, None, :] + _NEIGHBOUR_SHIFTS[None]) @ lattice
    nearest = np.argmin(np.linalg.norm(candidates, axis=2), axis=1)
    return candidates[np.arange(len(coords)), nearest]


def minimum_image_distances(coords, lattice, center):
    """Distance of every atom from atom ``center`` under the nearest-image rule."""
    return np.linalg.norm(minimum_image_offsets(coords, lattice, center), axis=1)


def coordination_shells(coords, lattice, center, tol=SHELL_TOL_BOHR):
    """Atom indices grouped into shells of equal nearest-image distance."""
    distances = minimum_image_distances(coords, lattice, center)
    order = np.argsort(distances, kind="stable")
    shells = []
    for index in order:
        if shells and distances[index] - distances[shells[-1][-1]] < tol:
            shells[-1].append(int(index))
        else:
            shells.append([int(index)])
    return shells


def nearest_atoms(coords, lattice, center, n_atoms, tol=SHELL_TOL_BOHR):
    """The ``n_atoms`` atoms nearest ``center``, which comes first.

    Refuses a count that cuts through a shell: the atoms of a shell are
    equivalent, so keeping some of them would be an arbitrary choice that breaks
    the symmetry of the defect.
    """
    n_atoms = int(n_atoms)
    shells = coordination_shells(coords, lattice, center, tol)
    if not 1 <= n_atoms <= sum(len(shell) for shell in shells):
        raise ValueError(
            "N_Frag=%d does not fit %d atoms"
            % (n_atoms, sum(len(shell) for shell in shells))
        )
    chosen = []
    for shell in shells:
        if len(chosen) >= n_atoms:
            break
        if len(chosen) + len(shell) > n_atoms:
            complete = np.cumsum([len(s) for s in shells])
            raise ValueError(
                "N_Frag=%d splits a coordination shell of %d equivalent atoms; "
                "complete shells give N_Frag in %s"
                % (n_atoms, len(shell), complete[:8].tolist())
            )
        chosen.extend(shell)
    return chosen
