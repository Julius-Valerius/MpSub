"""Equal-forward-budget ablation for the MpSub subspace dimension.

The earlier exploratory table used 100 iterations for every value of p.  Since
one MpSub iteration costs 2p+2 training-objective forward evaluations, that
comparison assigned a different budget to every row.  This driver instead
gives every configuration the largest whole number of iterations that does not
exceed FORWARD_BUDGET (8400 by default), repeats every p with three data seeds,
and stores the complete stdout/stderr stream for every run.

The already completed p=20, delta0=0.1 runs in results/hp_cost are reused by
the reporting script; set P_VALUES to include 20 only when an independent
rerun is desired.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LLM_DIR = ROOT / "llm"
OUT_DIR = Path(
    os.environ.get(
        "MPSUB_ABLATION_OUT",
        str(ROOT / "results" / "p_ablation_equal_budget"),
    )
).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

PYTHON = os.environ.get("MPSUB_PYTHON", sys.executable)
_bundled_model = ROOT / "model" / "opt-125m"
_legacy_model = Path(
    r"C:\Users\31203\Desktop\Haoyu-Pengcheng\2DMoSub\hf_cache\opt-125m"
)
MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        str(_bundled_model if _bundled_model.is_dir() else _legacy_model),
    )
)
# Model weights are loaded read-only from MODEL_PATH.  Runtime dataset caches
# and lock files belong inside this project so the experiment never depends on
# write access to an old Hugging Face cache.
HF_HOME = Path(os.environ.get("MPSUB_HF_HOME", str(OUT_DIR / "hf_runtime_cache")))

FORWARD_BUDGET = int(os.environ.get("FORWARD_BUDGET", "8400"))
P_VALUES = [int(v) for v in os.environ.get("P_VALUES", "5 10 15 25 30").split()]
SEEDS = [int(v) for v in os.environ.get("SEEDS", "0 1 2").split()]
WORKERS = int(os.environ.get("WORKERS", "1"))


@dataclass(frozen=True)
class Job:
    p: int
    seed: int
    steps: int
    forward_per_step: int
    actual_forwards: int
    log: Path
    output_dir: Path
    result_file: Path


def make_jobs() -> list[Job]:
    jobs: list[Job] = []
    # Interleave small and large p values so partial results already cover the
    # shape of the curve if a long run is interrupted.
    ordered = sorted(P_VALUES)
    interleaved: list[int] = []
    while ordered:
        interleaved.append(ordered.pop(0))
        if ordered:
            interleaved.append(ordered.pop(-1))
    for seed in SEEDS:
        for p in interleaved:
            forward_per_step = 2 * p + 2
            steps = FORWARD_BUDGET // forward_per_step
            jobs.append(
                Job(
                    p=p,
                    seed=seed,
                    steps=steps,
                    forward_per_step=forward_per_step,
                    actual_forwards=steps * forward_per_step,
                    log=OUT_DIR / f"p{p}_s{seed}.log",
                    output_dir=OUT_DIR / f"p{p}_s{seed}",
                    result_file=OUT_DIR / f"p{p}_s{seed}_metrics.json",
                )
            )
    return jobs


def is_complete(job: Job) -> bool:
    if not job.log.is_file() or job.log.stat().st_size == 0:
        return False
    text = job.log.read_text(errors="ignore")
    return "'accuracy'" in text and "'dev_accuracy'" in text


def command(job: Job) -> list[str]:
    # One Trainer evaluation at the final step records final development loss.
    # The framework then records test and development accuracies, as in the
    # existing main experiments.
    return [
        PYTHON,
        "-u",
        "run_lozo.py",
        "--model_name",
        "facebook/opt-125m",
        "--model_path",
        str(MODEL_PATH),
        "--task_name",
        "CB",
        "--output_dir",
        str(job.output_dir),
        "--result_file",
        str(job.result_file),
        "--tag",
        f"p_ablation_equal_p{job.p}_s{job.seed}",
        "--train_set_seed",
        str(job.seed),
        "--num_train",
        "100",
        "--num_dev",
        "50",
        "--num_eval",
        "100",
        "--logging_steps",
        str(max(1, job.steps // 20)),
        "--max_steps",
        str(job.steps),
        "--trainer",
        "LOZO",
        "--per_device_train_batch_size",
        "8",
        "--lr_scheduler_type",
        "constant",
        "--learning_rate",
        "0",
        "--evaluation_strategy",
        "steps",
        "--eval_steps",
        str(job.steps),
        "--save_strategy",
        "no",
        "--train_as_classification",
        "--use_psub",
        "True",
        "--psub_p",
        str(job.p),
        "--psub_delta0",
        "1e-1",
        "--psub_nsteps",
        "1",
        "--psub_kind",
        "grad",
        "--psub_use_torch",
        "True",
    ]


def write_manifest(jobs: list[Job]) -> None:
    path = OUT_DIR / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "p",
                "seed",
                "forward_budget",
                "forward_per_step",
                "steps",
                "actual_training_forwards",
                "budget_shortfall",
                "status_at_launch",
                "log",
            ]
        )
        for job in sorted(jobs, key=lambda j: (j.p, j.seed)):
            writer.writerow(
                [
                    job.p,
                    job.seed,
                    FORWARD_BUDGET,
                    job.forward_per_step,
                    job.steps,
                    job.actual_forwards,
                    FORWARD_BUDGET - job.actual_forwards,
                    "complete" if is_complete(job) else "pending",
                    job.log.name,
                ]
            )


def main() -> int:
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Local OPT-125M model not found: {MODEL_PATH}")

    jobs = make_jobs()
    write_manifest(jobs)
    queue = [job for job in jobs if not is_complete(job)]
    print(
        f"[plan] {len(queue)} of {len(jobs)} configurations pending; "
        f"budget={FORWARD_BUDGET}; workers={WORKERS}",
        flush=True,
    )
    for job in jobs:
        print(
            f"  p={job.p:2d} seed={job.seed}: {job.steps:3d} steps x "
            f"{job.forward_per_step:2d} = {job.actual_forwards} forwards",
            flush=True,
        )

    env = dict(os.environ)
    env.update(
        HF_HOME=str(HF_HOME),
        HF_DATASETS_CACHE=str(HF_HOME / "datasets"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )

    running: list[tuple[subprocess.Popen[bytes], Job, object]] = []
    failures: list[tuple[Job, int]] = []
    started = time.time()

    while queue or running:
        while queue and len(running) < WORKERS:
            job = queue.pop(0)
            job.output_dir.mkdir(parents=True, exist_ok=True)
            handle = job.log.open("wb")
            proc = subprocess.Popen(
                command(job),
                cwd=LLM_DIR,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running.append((proc, job, handle))
            print(
                f"[{time.time() - started:7.0f}s] start p={job.p} seed={job.seed} "
                f"({len(queue)} queued)",
                flush=True,
            )
            time.sleep(10)

        time.sleep(10)
        active: list[tuple[subprocess.Popen[bytes], Job, object]] = []
        for proc, job, handle in running:
            code = proc.poll()
            if code is None:
                active.append((proc, job, handle))
                continue
            handle.close()
            ok = code == 0 and is_complete(job)
            print(
                f"[{time.time() - started:7.0f}s] "
                f"{'complete' if ok else 'FAILED'} p={job.p} seed={job.seed}",
                flush=True,
            )
            if not ok:
                failures.append((job, code))
        running = active

    write_manifest(jobs)
    elapsed = (time.time() - started) / 3600
    print(f"[done] elapsed={elapsed:.2f} h; failures={len(failures)}", flush=True)
    for job, code in failures:
        print(f"  p={job.p} seed={job.seed}: return code {code}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
