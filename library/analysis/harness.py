import subprocess
from pathlib import Path

import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "app" / "benchmark"
DATA = ROOT / "data"
BASELINE = DATA / "baseline.txt"


def run_timing(graph_path: Path, representation: str) -> str:
    result = subprocess.run(
        [str(BINARY), str(graph_path), representation, "time"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def run_memory(graph_path: Path, representation: str) -> float:
    proc = subprocess.Popen(
        [str(BINARY), str(graph_path), representation, "mem"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    line = proc.stdout.readline()
    if line.strip() != "READY":
        # process died before signalling ready — surface its actual error
        proc.wait()
        err = proc.stderr.read()
        raise RuntimeError(
            f"benchmark mem failed for {graph_path} ({representation}): "
            f"{err.strip() or 'no output, exit ' + str(proc.returncode)}"
        )

    rss_bytes = psutil.Process(proc.pid).memory_info().rss

    proc.stdin.write("\n")
    proc.stdin.flush()
    proc.wait()

    return rss_bytes / (1024 * 1024)


def run_report(graph_path: Path, representation: str) -> str:
    result = subprocess.run(
        [str(BINARY), str(graph_path), representation, "report"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def representations_for(spec: str) -> list[str]:
    if spec == "both":
        return ["list", "matrix"]
    if spec in ("list", "matrix"):
        return [spec]
    raise ValueError(f"unknown representation spec: {spec!r}")


def collect_one(name: str, rep_spec: str) -> pd.DataFrame:
    """Compute all rows for a single graph, per its representation spec."""
    rows = []
    graph_path = DATA / f"{name}.txt"
    reps = representations_for(rep_spec)

    for skipped in sorted({"list", "matrix"} - set(reps)):
        rows.append(
            {"graph": name, "representation": skipped, "metric": "skipped", "value": 1}
        )

    for rep in reps:
        # timing
        for line in run_timing(graph_path, rep).strip().splitlines():
            parts = line.split(",")
            rows.append(
                {
                    "graph": name,
                    "representation": rep,
                    "metric": f"{parts[2]}_micros",
                    "value": float(parts[3]),
                }
            )
        # memory
        mem_raw = run_memory(graph_path, rep)
        mem_base = run_memory(BASELINE, rep)
        for metric, value in [
            ("memory_mb_raw", mem_raw),
            ("memory_mb_baseline", mem_base),
            ("memory_mb_graph", mem_raw - mem_base),
        ]:
            rows.append(
                {"graph": name, "representation": rep, "metric": metric, "value": value}
            )
        # report
        for line in run_report(graph_path, rep).strip().splitlines():
            if not line:
                continue
            parts = line.split(",")
            rows.append(
                {
                    "graph": name,
                    "representation": rep,
                    "metric": parts[2],
                    "value": float(parts[3]),
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    config = pd.read_csv(ROOT / "config.csv")
    out = DATA / "results.csv"

    for _, cfg_row in config.iterrows():
        name = cfg_row["graph"]
        print(f"processing {name} ...", flush=True)
        try:
            df = collect_one(name, cfg_row["representation"])
        except Exception as e:
            print(f"  FAILED for {name}: {e}", flush=True)
            continue
        # dump this graph's rows immediately, appending to the growing file
        df.to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"  wrote {len(df)} rows for {name}", flush=True)
