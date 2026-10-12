import numpy as np


A_HBN = 2.50
VACUUM_HBN = 30.0
HBN_CELLS = 9


def hbn_primitive_lattice(a=A_HBN, vacuum=VACUUM_HBN):
    """Primitive lattice vectors as rows, Angstrom."""
    return np.array([
        [a, 0.0, 0.0],
        [a / 2.0, a * np.sqrt(3.0) / 2.0, 0.0],
        [0.0, 0.0, vacuum],
    ])


def hbn_primitive_atoms(a=A_HBN, vacuum=VACUUM_HBN):
    """[(symbol, xyz)] of the 2-atom primitive cell, Angstrom."""
    a1, a2, _ = hbn_primitive_lattice(a, vacuum)
    z = np.array([0.0, 0.0, vacuum / 2.0])
    return [("B", z.copy()), ("N", (a1 + a2) / 3.0 + z)]


def hbn_supercell_lattice(ncells, a=A_HBN, vacuum=VACUUM_HBN):
    """Lattice vectors of the ncells x ncells x 1 supercell as rows, Angstrom."""
    return np.diag([int(ncells), int(ncells), 1]) @ hbn_primitive_lattice(a, vacuum)


def hbn_supercell(ncells, a=A_HBN, vacuum=VACUUM_HBN):
    """(symbols, coords, lattice) of the ncells x ncells x 1 supercell, Angstrom."""
    ncells = int(ncells)
    if ncells < 1:
        raise ValueError("ncells must be positive")
    lattice = hbn_primitive_lattice(a, vacuum)
    atoms = hbn_primitive_atoms(a, vacuum)
    images = np.array([[i, j, 0] for i in range(ncells) for j in range(ncells)])
    shifts = images @ lattice
    symbols = [symbol for _ in shifts for symbol, _ in atoms]
    coords = np.array([xyz + shift for shift in shifts for _, xyz in atoms])
    return symbols, coords, hbn_supercell_lattice(ncells, a, vacuum)


def central_boron_index(ncells):
    """Supercell index of the boron in image (ncells//2, ncells//2).

    For 9 x 9 this is atom 80, the vacancy site of the sibling repository's run.
    """
    ncells = int(ncells)
    middle = ncells // 2
    return 2 * (ncells * middle + middle)
