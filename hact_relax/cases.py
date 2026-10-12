from __future__ import annotations

from dataclasses import dataclass

from pyscf import lib


BOHR = lib.param.BOHR
FULL_CHAIN_KMESH = (1, 1, 1)
CHAIN_CELLS = 21
R_ANG = 1.7
FRAGMENTS = (1, 3, 5, 7, 9)

LIH1D = "lih1d"
HBN2D = "hbn2d"
HOST_FRAGMENTS = {
    LIH1D: FRAGMENTS,
    HBN2D: (1, 4, 10, 13, 19),
}
HOST_DEFAULT_BASIS = {LIH1D: "pob-tzvp", HBN2D: "cc-pvdz"}
HOST_DEFAULT_FD_AXES = {LIH1D: "x", HBN2D: "xy"}
HOST_DEFAULT_WRAP = {LIH1D: False, HBN2D: True}


@dataclass(frozen=True)
class VacancyCase:
    vacancy: str
    charge: int
    spin: int
    seed_vacancy: bool
    ncas: int
    ncas_elec: int
    level_shift: float
    host: str = LIH1D
    cas_selection: str = "vacancy-local"


CASES = {
    "vhn": VacancyCase("H", 0, 1, False, 10, 9, 0.1),
    "vhc": VacancyCase("H", -1, 2, True, 10, 8, 0.4),
    "vln": VacancyCase("Li", 0, 1, False, 8, 7, 0.2),
    "vlc": VacancyCase("Li", -1, 0, True, 8, 6, 0.4),
    "vbc": VacancyCase("B", -1, 2, False, 6, 10, 0.4, host=HBN2D,
                       cas_selection="energy"),
}


def parse_cases(value: str) -> list[str]:
    # "all" keeps its original meaning: the four LiH-1D vacancies.
    values = ([name for name, case in CASES.items() if case.host == LIH1D]
              if value == "all" else value.split(","))
    values = list(dict.fromkeys(v.strip() for v in values if v.strip()))
    unknown = sorted(set(values) - set(CASES))
    if unknown:
        raise ValueError("unknown case(s): %s" % ", ".join(unknown))
    return values


def case_host(case_names) -> str:
    """The one host the requested cases share."""
    hosts = sorted({CASES[name].host for name in case_names})
    if len(hosts) != 1:
        raise ValueError(
            "cases %s mix the hosts %s; run each host separately"
            % (", ".join(case_names), ", ".join(hosts))
        )
    return hosts[0]


def parse_fragments(value: str, allowed=FRAGMENTS) -> list[int]:
    allowed = tuple(int(v) for v in allowed)
    values = list(allowed) if value == "all" else [
        int(v) for v in value.split(",")
    ]
    values = list(dict.fromkeys(values))
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(
            "fragment size(s) must be %s; got %s"
            % (",".join(str(v) for v in allowed), unknown)
        )
    return values


def fragment_relax_radius(n_frag: int, r_ang: float = R_ANG) -> float:
    if int(n_frag) < 1:
        raise ValueError("n_frag must be positive")
    return max((int(n_frag) - 1) // 2, 1) * float(r_ang)
