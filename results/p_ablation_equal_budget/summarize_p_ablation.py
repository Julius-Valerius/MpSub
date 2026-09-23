"""Build auditable per-run and summary tables for the equal-budget p ablation."""

from __future__ import annotations

import csv
import math
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MAIN_RUNS = ROOT / "results" / "scaling" / "data" / "main_results_per_run.csv"
MANIFEST = HERE / "manifest.csv"

ACC_RE = re.compile(
    r"\{'accuracy':\s*([0-9.]+),\s*'dev_accuracy':\s*([0-9.]+)\}"
)
LOSS_RE = re.compile(r"'eval_loss':\s*([0-9.eE+-]+)")


def read_manifest() -> dict[tuple[int, int], dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return {
            (int(row["p"]), int(row["seed"])): row
            for row in csv.DictReader(handle)
        }


def parse_log(path: Path) -> tuple[float, float, float]:
    text = path.read_text(errors="ignore")
    accuracies = ACC_RE.findall(text)
    losses = LOSS_RE.findall(text)
    if not accuracies or not losses:
        raise ValueError(f"Incomplete run log: {path}")
    test_acc, dev_acc = map(float, accuracies[-1])
    return float(losses[-1]), dev_acc, test_acc


def p20_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with MAIN_RUNS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not (
                row["model"] == "opt-125m"
                and row["method"] == "MpSub"
                and row["hyperparameter"] == "1e-1"
            ):
                continue
            rows.append(
                {
                    "p": 20,
                    "seed": int(row["seed"]),
                    "steps": 200,
                    "forward_per_step": 42,
                    "actual_training_forwards": int(row["forward_budget"]),
                    "final_dev_loss": float(row["final_dev_loss"]),
                    "final_dev_accuracy": float(row["final_dev_accuracy"]),
                    "final_test_accuracy": float(row["final_test_accuracy"]),
                    "source": "results/scaling/data/main_results_per_run.csv",
                }
            )
    if len(rows) != 3:
        raise ValueError(f"Expected three reusable p=20 runs, found {len(rows)}")
    return rows


def new_rows(manifest: dict[tuple[int, int], dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (p, seed), entry in sorted(manifest.items()):
        log = HERE / entry["log"]
        loss, dev_acc, test_acc = parse_log(log)
        rows.append(
            {
                "p": p,
                "seed": seed,
                "steps": int(entry["steps"]),
                "forward_per_step": int(entry["forward_per_step"]),
                "actual_training_forwards": int(entry["actual_training_forwards"]),
                "final_dev_loss": loss,
                "final_dev_accuracy": dev_acc,
                "final_test_accuracy": test_acc,
                "source": f"results/p_ablation_equal_budget/{entry['log']}",
            }
        )
    return rows


def write_per_run(rows: list[dict[str, object]]) -> None:
    fields = [
        "p",
        "seed",
        "steps",
        "forward_per_step",
        "actual_training_forwards",
        "final_dev_loss",
        "final_dev_accuracy",
        "final_test_accuracy",
        "source",
    ]
    with (HERE / "p_ablation_per_run.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (int(item["p"]), int(item["seed"]))):
            writer.writerow(row)


def write_summary(rows: list[dict[str, object]]) -> None:
    fields = [
        "p",
        "num_seeds",
        "training_forwards_min",
        "training_forwards_max",
        "dev_loss_mean",
        "dev_loss_min",
        "dev_loss_max",
        "dev_accuracy_mean",
        "dev_accuracy_min",
        "dev_accuracy_max",
        "test_accuracy_mean",
        "test_accuracy_min",
        "test_accuracy_max",
    ]
    groups: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(int(row["p"]), []).append(row)

    summary: list[dict[str, object]] = []
    for p, group in sorted(groups.items()):
        if len(group) != 3:
            raise ValueError(f"p={p} has {len(group)} runs; expected three")
        seed_set = {int(row["seed"]) for row in group}
        if seed_set != {0, 1, 2}:
            raise ValueError(f"p={p} has seeds {sorted(seed_set)}")

        forwards = [int(row["actual_training_forwards"]) for row in group]
        dev_loss = [float(row["final_dev_loss"]) for row in group]
        dev_acc = [float(row["final_dev_accuracy"]) for row in group]
        test_acc = [float(row["final_test_accuracy"]) for row in group]
        summary.append(
            {
                "p": p,
                "num_seeds": len(group),
                "training_forwards_min": min(forwards),
                "training_forwards_max": max(forwards),
                "dev_loss_mean": statistics.mean(dev_loss),
                "dev_loss_min": min(dev_loss),
                "dev_loss_max": max(dev_loss),
                "dev_accuracy_mean": statistics.mean(dev_acc),
                "dev_accuracy_min": min(dev_acc),
                "dev_accuracy_max": max(dev_acc),
                "test_accuracy_mean": statistics.mean(test_acc),
                "test_accuracy_min": min(test_acc),
                "test_accuracy_max": max(test_acc),
            }
        )

    with (HERE / "p_ablation_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    manifest = read_manifest()
    rows = new_rows(manifest) + p20_rows()
    write_per_run(rows)
    write_summary(rows)
    print(f"wrote {len(rows)} runs across {len(set(int(r['p']) for r in rows))} p values")


if __name__ == "__main__":
    main()
