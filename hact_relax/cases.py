from __future__ import annotations

from dataclasses import dataclass

from pyscf import lib


BOHR = lib.param.BOHR
FULL_CHAIN_KMESH = (1, 1, 1)
CHAIN_CELLS = 21
R_ANG = 1.7
FRAGMENTS = (1, 3, 5, 7, 9)


@dataclass(frozen=True)
class VacancyCase:
    vacancy: str
    charge: int
    spin: int
    seed_vacancy: bool
    # CAS window for --state excited.  The Schmidt step produces the EMBEDDING
    # active space (tens of orbitals); CASCI runs in a smaller frontier window cut
    # inside it, placed at orbitals [ncore, ncore + ncas) of the Fock-energy-sorted
    # active orbitals, with ncore = (n_active_elec - ncas_elec) // 2.  The window
    # SIZE is a property of the defect, not of the embedding, so it lives here.
    ncas: int
    ncas_elec: int
    level_shift: float


CASES = {
    "vhn": VacancyCase("H", 0, 1, False, 10, 9, 0.1),
    "vhc": VacancyCase("H", -1, 2, True, 10, 8, 0.4),
    "vln": VacancyCase("Li", 0, 1, False, 8, 7, 0.2),
    "vlc": VacancyCase("Li", -1, 0, True, 8, 6, 0.4),
}


def parse_cases(value: str) -> list[str]:
    values = list(CASES) if value == "all" else value.split(",")
    values = list(dict.fromkeys(v.strip() for v in values if v.strip()))
    unknown = sorted(set(values) - set(CASES))
    if unknown:
        raise ValueError("unknown case(s): %s" % ", ".join(unknown))
    return values


def parse_fragments(value: str) -> list[int]:
    values = list(FRAGMENTS) if value == "all" else [
        int(v) for v in value.split(",")
    ]
    values = list(dict.fromkeys(values))
    unknown = sorted(set(values) - set(FRAGMENTS))
    if unknown:
        raise ValueError("fragment size(s) must be 1,3,5,7,9; got %s" % unknown)
    return values


def fragment_relax_radius(n_frag: int, r_ang: float = R_ANG) -> float:
    if int(n_frag) < 1:
        raise ValueError("n_frag must be positive")
    return max((int(n_frag) - 1) // 2, 1) * float(r_ang)
