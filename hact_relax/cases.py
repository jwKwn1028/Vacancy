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
    level_shift: float


CASES = {
    "vhn": VacancyCase("H", 0, 1, False, 0.1),
    "vhc": VacancyCase("H", -1, 2, True, 0.4),
    "vln": VacancyCase("Li", 0, 1, False, 0.2),
    "vlc": VacancyCase("Li", -1, 0, True, 0.4),
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


def parse_relax_radius(value: str) -> float | None:
    text = str(value).strip().lower()
    if text == "all":
        return None
    radius = float(text)
    if radius <= 0.0:
        raise ValueError("--relax-radius must be positive or 'all'")
    return radius
