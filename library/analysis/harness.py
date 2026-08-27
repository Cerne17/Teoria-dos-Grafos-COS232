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
        text=True,
    )
    line = proc.stdout.readline()
    assert line.strip() == "READY", f"expected READY, got {line!r}"

    rss_bytes = psutil.Process(proc.pid).memory_info().rss

    proc.stdin.write("\n")
    proc.stdin.flush()
    proc.wait()

    return rss_bytes / (1024 * 1024)


def collect(graph_files: list[str]) -> pd.DataFrame:
    rows = []
    for graph_file in graph_files:
        graph_path = DATA / graph_file
        name = graph_file.replace(".txt", "")

        for rep in ["list", "matrix"]:
            # --- timing: parse the two CSV rows the binary printed ---
            for line in run_timing(graph_path, rep).strip().splitlines():
                parts = line.split(",")
                algo = parts[2]  # "bfs" | "dfs"
                avg = float(parts[3])
                rows.append(
                    {
                        "graph": name,
                        "representation": rep,
                        "metric": f"{algo}_micros",
                        "value": avg,
                    }
                )

            # --- memory: raw, baseline, and the isolated difference ---
            mem_raw = run_memory(graph_path, rep)
            mem_base = run_memory(BASELINE, rep)
            for metric, value in [
                ("memory_mb_raw", mem_raw),
                ("memory_mb_baseline", mem_base),
                ("memory_mb_graph", mem_raw - mem_base),
            ]:
                rows.append(
                    {
                        "graph": name,
                        "representation": rep,
                        "metric": metric,
                        "value": value,
                    }
                )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    graph_list = [
        "grafo_1.txt",
        # "grafo_2.txt",
        # "grafo_3.txt",
        # "grafo_4.txt",
        # "grafo_5.txt",
    ]
    # grafo_2..5 commented out during development (large, slow)

    df = collect(graph_list)
    print(df.to_string(index=False))
    df.to_csv(DATA / "analysis_results.csv", index=False)
