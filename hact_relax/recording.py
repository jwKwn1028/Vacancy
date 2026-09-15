from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import resource
import threading
import time
from typing import Any, Iterator

import numpy as np


_STATUS_MB = {
    "VmRSS": "rss_mb",
    "VmSize": "virtual_memory_mb",
    "RssAnon": "anonymous_rss_mb",
    "RssFile": "file_rss_mb",
}
_MEMINFO_MB = {
    "MemTotal": "system_memory_total_mb",
    "MemAvailable": "system_memory_available_mb",
    "SwapTotal": "system_swap_total_mb",
    "SwapFree": "system_swap_free_mb",
}
_IO_COUNTS = {
    "syscr": "io_read_syscalls",
    "syscw": "io_write_syscalls",
    "read_bytes": "io_read_bytes",
    "write_bytes": "io_write_bytes",
}


def _proc_values(path: str, wanted: dict[str, str]) -> dict[str, float]:
    values: dict[str, float] = {}
    try:
        with open(path) as handle:
            for line in handle:
                name, _, rest = line.partition(":")
                key = wanted.get(name.strip())
                fields = rest.split()
                if key is not None and fields:
                    values[key] = float(fields[0])
    except OSError:
        pass
    return values


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return os.fspath(value)
    return str(value)


def resource_snapshot() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    snapshot: dict[str, Any] = {
        "cpu_user_seconds": round(usage.ru_utime, 3),
        "cpu_system_seconds": round(usage.ru_stime, 3),
        "cpu_seconds": round(usage.ru_utime + usage.ru_stime, 3),
        "child_cpu_seconds": round(children.ru_utime + children.ru_stime, 3),
        "peak_rss_mb": round(usage.ru_maxrss / 1024.0, 3),
        "minor_page_faults": int(usage.ru_minflt),
        "major_page_faults": int(usage.ru_majflt),
        "voluntary_context_switches": int(usage.ru_nvcsw),
        "involuntary_context_switches": int(usage.ru_nivcsw),
        "cpu_affinity": len(os.sched_getaffinity(0)),
        "load_average": [round(value, 2) for value in os.getloadavg()],
    }
    for key, value in _proc_values("/proc/self/status", _STATUS_MB).items():
        snapshot[key] = round(value / 1024.0, 3)
    for key, value in _proc_values("/proc/meminfo", _MEMINFO_MB).items():
        snapshot[key] = round(value / 1024.0, 3)
    for key, value in _proc_values("/proc/self/io", _IO_COUNTS).items():
        snapshot[key] = int(value)
    threads = _proc_values("/proc/self/status", {"Threads": "process_threads"})
    if threads:
        snapshot["process_threads"] = int(threads["process_threads"])
    try:
        snapshot["open_file_descriptors"] = len(os.listdir("/proc/self/fd"))
    except OSError:
        pass
    return snapshot


class JsonlWriter:

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    def write(self, row: dict[str, Any]) -> None:
        line = json.dumps(row, default=jsonable, sort_keys=True)
        with self._lock:
            with self.path.open("a") as handle:
                handle.write(line + "\n")


class ResourceProfiler:

    def __init__(
        self,
        writer: JsonlWriter,
        interval_seconds: float,
        configuration: dict[str, Any] | None = None,
    ):
        self.writer = writer
        self.interval_seconds = float(interval_seconds)
        self.configuration = dict(configuration or {})
        self.phase_name = "initializing"
        self.started = time.time()
        self._cpu = 0.0
        self._sampled = self.started
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _row(self, event: str, **fields: Any) -> dict[str, Any]:
        snapshot = resource_snapshot()
        now = time.time()
        interval = max(now - self._sampled, 1e-12)
        cpu = float(snapshot["cpu_seconds"])
        sample_cpu = cpu - self._cpu
        allocation = max(int(self.configuration.get("configured_threads") or 1), 1)
        self._sampled, self._cpu = now, cpu
        row = {
            "event": event,
            "phase": self.phase_name,
            "elapsed_wall_seconds": round(now - self.started, 6),
            "sample_interval_seconds": round(interval, 6),
            "sample_cpu_seconds": round(sample_cpu, 6),
            "sample_cpu_utilization_of_allocation_percent":
                round(100.0 * sample_cpu / (interval * allocation), 2),
        }
        row.update(self.configuration)
        row.update(snapshot)
        row.update(fields)
        return row

    def start(self) -> None:
        self.writer.write(self._row("profiler_start"))
        if self.interval_seconds <= 0.0:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.writer.write(self._row("periodic_resource_sample"))

    def phase(self, name: str) -> None:
        self.phase_name = str(name)
        self.writer.write(self._row("phase_start"))

    @contextlib.contextmanager
    def stage(self, name: str, **fields: Any) -> Iterator[None]:
        started = time.time()
        cpu = float(resource_snapshot()["cpu_seconds"])
        try:
            yield
        finally:
            snapshot = resource_snapshot()
            self.writer.write(self._row(
                "stage_end",
                stage=str(name),
                stage_wall_seconds=round(time.time() - started, 6),
                stage_cpu_seconds=round(float(snapshot["cpu_seconds"]) - cpu, 6),
                **fields,
            ))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 5.0)
            self._thread = None
        self.writer.write(self._row("profiler_stop"))


class RunRecorder:

    def __init__(
        self,
        output_dir: Path,
        config: dict[str, Any],
        *,
        resource_interval_seconds: float = 30.0,
    ):
        self.output_dir = Path(output_dir)
        self.evaluations_path = self.output_dir / "evaluations.jsonl"
        self.scf_iterations_path = self.output_dir / "scf_iterations.jsonl"
        self.resources_path = self.output_dir / "resources.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.config_path = self.output_dir / "config.json"
        self.pristine_chkfile = self.output_dir / "pristine.chk"
        self.hact_chkfile = self.output_dir / "hact.chk"

        self._evaluations = JsonlWriter(self.evaluations_path)
        self._scf_iterations = JsonlWriter(self.scf_iterations_path)
        self._resources = JsonlWriter(self.resources_path)
        self.profiler = ResourceProfiler(
            self._resources,
            resource_interval_seconds,
            {
                "configured_threads": config.get("threads"),
                "configured_memory_limit_mb": config.get("memory_mb"),
            },
        )

        self.evaluation_count = 0
        self.started = time.time()
        self.resource_at_start = resource_snapshot()

        self.config = dict(config)
        self.config.update({
            "evaluation_profile_jsonl": os.fspath(self.evaluations_path),
            "scf_iteration_profile_jsonl": os.fspath(self.scf_iterations_path),
            "resource_profile_jsonl": os.fspath(self.resources_path),
            "resource_profile_interval_seconds": float(resource_interval_seconds),
            "resource_at_start": self.resource_at_start,
        })
        self._write_json(self.config_path, self.config)
        self.profiler.start()

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=jsonable) + "\n"
        )

    def next_evaluation(self) -> int:
        self.evaluation_count += 1
        return self.evaluation_count

    def evaluation(self, row: dict[str, Any]) -> None:
        self._evaluations.write({
            "threads": self.config.get("threads"),
            **row,
            **resource_snapshot(),
        })

    def scf_iteration(self, row: dict[str, Any]) -> None:
        self._scf_iterations.write(
            {"event": "scf_iteration", **row, **resource_snapshot()}
        )

    def phase(self, name: str) -> None:
        self.profiler.phase(name)

    def stage(self, name: str, **fields: Any):
        return self.profiler.stage(name, **fields)

    def summary(self, fields: dict[str, Any]) -> Path:
        summary = dict(self.config)
        summary.update(fields)
        summary.update({
            "unique_energy_evaluations": self.evaluation_count,
            "run_wall_seconds": round(time.time() - self.started, 6),
            "resource_at_start": self.resource_at_start,
            "resource_at_end": resource_snapshot(),
        })
        self._write_json(self.summary_path, summary)
        return self.summary_path

    def close(self) -> None:
        self.profiler.stop()


class NullRecorder:

    output_dir = None
    pristine_chkfile = None
    hact_chkfile = None
    evaluation_count = 0

    def next_evaluation(self) -> int:
        return 0

    def evaluation(self, row: dict[str, Any]) -> None:
        return None

    def scf_iteration(self, row: dict[str, Any]) -> None:
        return None

    def phase(self, name: str) -> None:
        return None

    def stage(self, name: str, **fields: Any):
        return contextlib.nullcontext()

    def summary(self, fields: dict[str, Any]) -> None:
        return None

    def close(self) -> None:
        return None
