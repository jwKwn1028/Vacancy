from __future__ import annotations

import os
import sys
from pathlib import Path


def _early_int_option(name: str, default: int) -> int:
    prefix = name + "="
    for i, value in enumerate(sys.argv[1:]):
        if value.startswith(prefix):
            return int(value.split("=", 1)[1])
        if value == name and i + 2 <= len(sys.argv) - 1:
            return int(sys.argv[i + 2])
    return int(default)


THREADS = _early_int_option(
    "--threads", int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
)
if THREADS < 1:
    raise SystemExit("--threads must be positive")
for _thread_env in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env] = str(THREADS)


REPO = Path(__file__).resolve().parents[1]
if os.fspath(REPO) not in sys.path:
    sys.path.insert(0, os.fspath(REPO))
