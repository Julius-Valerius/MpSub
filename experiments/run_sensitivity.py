#!/usr/bin/env python
"""Runner for MpSub initial radius sensitivity sweep.

Runs MpSub over delta0 in ['1e-3', '1e-2', '3e-2', '1e-1', '3e-1'] x 3 seeds.
Completed logs are automatically detected and skipped.
Existing MeZO logs in results/hp_cost are reused.
Upon completion, paper/make_figures.py is called to generate fig_sensitivity.pdf.
"""

from __future__ import annotations

import os
import sys
import glob
import time
import subprocess
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLM_DIR = ROOT / "llm"
OUT_DIR = ROOT / "results" / "hp_cost"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT / "model" / "opt-125m"

PY = sys.executable
WORKERS = int(os.environ.get("WORKERS", "4"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0 1 2").split()]
PDELTAS = os.environ.get("PDELTAS", "1e-3 1e-2 3e-2 1e-1 3e-1").split()

ENV = dict(os.environ)
if MODEL_DIR.is_dir():
    ENV.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
ENV.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def done(log_path: Path) -> bool:
    if not log_path.is_file() or log_path.stat().st_size == 0:
        return False
    with open(log_path, errors="ignore") as f:
        return "'accuracy'" in f.read()


def make_jobs():
    jobs = []
    common = [
        "--model_name", "facebook/opt-125m",
        "--task_name", "CB",
        "--num_train", "100",
        "--num_dev", "50",
        "--num_eval", "100",
        "--per_device_train_batch_size", "8",
        "--lr_scheduler_type", "constant",
        "--save_strategy", "no",
        "--train_as_classification",
        "--evaluation_strategy", "steps",
    ]
    if MODEL_DIR.is_dir():
        common += ["--model_path", str(MODEL_DIR)]

    for d in PDELTAS:
        for sd in SEEDS:
            log = OUT_DIR / f"psub_d{d}_s{sd}.log"
            argv = [
                PY, "-u", "run_lozo.py",
                "--output_dir", str(OUT_DIR / f"ps_{d}_{sd}"),
                "--tag", f"hpc_ps_{d}_{sd}",
                "--train_set_seed", str(sd),
                "--logging_steps", "5",
                "--max_steps", "200",
                "--trainer", "LOZO",
                "--learning_rate", "0",
                "--eval_steps", "25",
                "--use_psub", "True",
                "--psub_p", "20",
                "--psub_delta0", d,
                "--psub_nsteps", "1",
                "--psub_kind", "grad",
                "--psub_use_torch", "True",
            ] + common
            jobs.append((log, argv))
    return jobs


def run_pool():
    all_jobs = make_jobs()
    jobs = [(lg, av) for lg, av in all_jobs if not done(lg)]
    print(f"[plan] {len(jobs)} of {len(all_jobs)} runs remaining, WORKERS={WORKERS}")

    running, failed, t0 = [], [], time.time()
    queue = list(jobs)
    while queue or running:
        while queue and len(running) < WORKERS:
            log, argv = queue.pop(0)
            fh = open(log, "w", encoding="utf-8")
            p = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, env=ENV, cwd=str(LLM_DIR))
            running.append((p, log, fh))
            print(f"[{time.time()-t0:5.0f}s] START {log.name} ({len(queue)} queued)", flush=True)
            time.sleep(6)  # stagger CUDA startup
        time.sleep(10)
        still = []
        for p, log, fh in running:
            if p.poll() is None:
                still.append((p, log, fh))
                continue
            fh.close()
            ok = (p.returncode == 0) and done(log)
            status = "OK  " if ok else f"FAIL(rc={p.returncode})"
            print(f"[{time.time()-t0:5.0f}s] {status} {log.name}", flush=True)
            if not ok:
                failed.append(log)
        running = still

    print(f"[done] Finished in {(time.time()-t0)/60:.1f} minutes, {len(failed)} failures.")
    if failed:
        print("Failed logs:", [f.name for f in failed])


def summarize_and_plot():
    import re

    def _acc(path):
        for ln in open(path, errors="ignore"):
            m = re.search(r"\{'accuracy':\s*([0-9.]+),\s*'dev_accuracy':\s*([0-9.]+)\}", ln)
            if m:
                return float(m.group(1))
        return None

    print("\n" + "=" * 55)
    print("           EXPERIMENT SUMMARY")
    print("=" * 55)
    for d in PDELTAS:
        accs = [_acc(f) for f in glob.glob(str(OUT_DIR / f"psub_d{d}_s*.log"))]
        accs = [a for a in accs if a is not None]
        if accs:
            print(f"delta0 = {d:<8} Mean = {st.mean(accs):.4f} (Min: {min(accs):.4f}, Max: {max(accs):.4f}, N={len(accs)})")
        else:
            print(f"delta0 = {d:<8} (no completed runs)")
    print("=" * 55)

    plot_script = ROOT / "paper" / "make_figures.py"
    if plot_script.is_file():
        print("[plot] Regenerating fig_sensitivity.pdf...")
        subprocess.run([PY, str(plot_script), str(OUT_DIR), str(ROOT / "paper")])
        out_fig = ROOT / "paper" / "fig_sensitivity.pdf"
        if out_fig.is_file():
            print(f"[plot] SUCCESS: {out_fig} generated!")


if __name__ == "__main__":
    run_pool()
    summarize_and_plot()
