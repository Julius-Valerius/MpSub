"""Matched-compute total-cost experiment, process-pool driver.

Runs 30 configurations at 8400 forward passes each: MeZO over five learning
rates x three seeds at 4200 steps, and MpSub over five initial radii x three
seeds at 200 steps with p = 20. Log file names match the serial shell driver,
so the two are interchangeable and either can resume the other's run;
completed logs are detected and skipped.

Per worker the resident cost is the fp32 125M weights (0.5 GB) plus forward
activations at batch size 8 (0.5-1 GB) plus the CUDA context (~0.4 GB).
Wall-clock per step is not comparable under contention, but the experiment
measures accuracy at a fixed forward-pass count, which contention does not
affect.

Place next to run_mezo.py and run_lozo.py. The model path is taken from
MODEL_PATH if set, otherwise from the local candidates below, otherwise
weights are fetched online.

Usage: python run_hp_cost_parallel.py
       WORKERS=8 python run_hp_cost_parallel.py
       ONLY=mezo python run_hp_cost_parallel.py
"""
import os
import sys
import re
import glob
import time
import subprocess
import statistics as st

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)

PY = sys.executable
OUT = os.path.join('result', 'hp_cost')
os.makedirs(OUT, exist_ok=True)

_cands = [os.environ.get('MODEL_PATH'),
          os.path.join(BASE, '..', 'hf_cache', 'opt-125m'),
          '/root/autodl-tmp/opt-125m',
          os.path.join(BASE, 'opt-125m')]
MODEL_PATH = next((c for c in _cands if c and os.path.isdir(c)), None)
ENV = dict(os.environ)
if MODEL_PATH:
    ENV.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if 'HF_HOME' not in ENV:
        ENV['HF_HOME'] = os.path.abspath(os.path.join(BASE, '..', 'hf_cache'))
else:
    ENV.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
print('[env] python=%s' % PY)
print('[env] MODEL_PATH=%s' % (MODEL_PATH or '(none, fetching online)'))

WORKERS = int(os.environ.get('WORKERS', 5))
MSTEPS = int(os.environ.get('MSTEPS', 4200))
SEEDS = [int(s) for s in os.environ.get('SEEDS', '0 1 2').split()]
MLRS = os.environ.get('MLRS', '1e-4 1e-5 1e-6 1e-7 1e-8').split()
PDELTAS = os.environ.get('PDELTAS', '1e-1 3e-1 1.0 3.0 10.0').split()
ONLY = os.environ.get('ONLY', '')

COMMON = ['--model_name', 'facebook/opt-125m', '--task_name', 'CB',
          '--num_train', '100', '--num_dev', '50', '--num_eval', '100',
          '--per_device_train_batch_size', '8',
          '--lr_scheduler_type', 'constant', '--save_strategy', 'no',
          '--train_as_classification', '--evaluation_strategy', 'steps']
if MODEL_PATH:
    COMMON += ['--model_path', MODEL_PATH]


def make_jobs():
    jobs = []
    if ONLY != 'psub':
        for lr in MLRS:
            for sd in SEEDS:
                log = os.path.join(OUT, 'mezo_lr%s_s%d.log' % (lr, sd))
                argv = [PY, '-u', 'run_mezo.py',
                        '--output_dir', os.path.join(OUT, 'mz_%s_%d' % (lr, sd)),
                        '--tag', 'hpc_mz_%s_%d' % (lr, sd),
                        '--train_set_seed', str(sd),
                        '--logging_steps', '50', '--max_steps', str(MSTEPS),
                        '--trainer', 'zo', '--learning_rate', lr,
                        '--zo_eps', '1e-3', '--eval_steps', '300'] + COMMON
                jobs.append((log, argv))
    if ONLY != 'mezo':
        for d in PDELTAS:
            for sd in SEEDS:
                log = os.path.join(OUT, 'psub_d%s_s%d.log' % (d, sd))
                argv = [PY, '-u', 'run_lozo.py',
                        '--output_dir', os.path.join(OUT, 'ps_%s_%d' % (d, sd)),
                        '--tag', 'hpc_ps_%s_%d' % (d, sd),
                        '--train_set_seed', str(sd),
                        '--logging_steps', '5', '--max_steps', '200',
                        '--trainer', 'LOZO', '--learning_rate', '0',
                        '--eval_steps', '25',
                        '--use_psub', 'True', '--psub_p', '20',
                        '--psub_delta0', d, '--psub_nsteps', '1',
                        '--psub_kind', 'grad', '--psub_use_torch', 'True'] + COMMON
                jobs.append((log, argv))
    return jobs


def done(log):
    if not os.path.isfile(log) or os.path.getsize(log) == 0:
        return False
    with open(log, errors='ignore') as f:
        return "'accuracy'" in f.read()


def main():
    all_jobs = make_jobs()
    jobs = [(lg, av) for lg, av in all_jobs if not done(lg)]
    print('[plan] %d of %d configurations to run, WORKERS=%d'
          % (len(jobs), len(all_jobs), WORKERS))
    running, failed, t0 = [], [], time.time()
    queue = list(jobs)
    while queue or running:
        while queue and len(running) < WORKERS:
            log, argv = queue.pop(0)
            fh = open(log, 'w')
            p = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT,
                                 env=ENV, cwd=BASE)
            running.append((p, log, fh))
            print('[%5.0fs] start %-36s (%d queued)'
                  % (time.time() - t0, os.path.basename(log), len(queue)), flush=True)
            time.sleep(8)          # stagger CUDA initialization
        time.sleep(10)
        still = []
        for p, log, fh in running:
            if p.poll() is None:
                still.append((p, log, fh))
                continue
            fh.close()
            ok = (p.returncode == 0) and done(log)
            print('[%5.0fs] %s %s' % (time.time() - t0,
                  'ok  ' if ok else 'FAIL(rc=%s)' % p.returncode,
                  os.path.basename(log)), flush=True)
            if not ok:
                failed.append(log)
        running = still
    print('[done] %.1f minutes, %d failed%s'
          % ((time.time() - t0) / 60, len(failed),
             (': ' + ', '.join(os.path.basename(f) for f in failed)) if failed else ''))
    summarize()


def _acc(path):
    a = None
    for ln in open(path, errors='ignore'):
        m = re.search(r"\{'accuracy':\s*([0-9.]+),\s*'dev_accuracy':\s*([0-9.]+)\}", ln)
        if m:
            a = float(m.group(1))
    return a


def summarize():
    print()
    print('==== total cost including hyperparameter search ====')
    best = {}
    for name, pat, key in [
            ('MeZO', os.path.join(OUT, 'mezo_lr*_s*.log'), r'lr([0-9e.+-]+)_s(\d+)'),
            ('psub', os.path.join(OUT, 'psub_d*_s*.log'), r'd([0-9e.+-]+)_s(\d+)')]:
        groups = {}
        for f in sorted(glob.glob(pat)):
            m = re.search(key, os.path.basename(f))
            a = _acc(f) if m else None
            if a is not None:
                groups.setdefault(m.group(1), []).append(a)
        rows = [(hp, st.mean(v), min(v), max(v), len(v)) for hp, v in groups.items()]
        for hp, mean, lo, hi, ns in sorted(rows, key=lambda r: -r[1]):
            print('%-6s %-8s mean=%-7.3f (%.3f~%.3f, n=%d)' % (name, hp, mean, lo, hi, ns))
        if rows:
            b = max(rows, key=lambda r: r[1])
            best[name] = b
            print('  %s best: %s mean=%.3f | %d grid points => %dx8400 forwards'
                  % (name, b[0], b[1], len(rows), len(rows)))
        print()
    d01 = [a for a in (_acc(f) for f in glob.glob(os.path.join(OUT, 'psub_d1e-1_s*.log')))
           if a is not None]
    if d01 and 'MeZO' in best:
        pm, mm = st.mean(d01), best['MeZO'][1]
        print('psub at the default delta0=0.1 mean=%.3f vs tuned MeZO mean=%.3f, '
              'difference=%+.3f' % (pm, mm, pm - mm))
        # Pre-registered criterion, fixed before the runs.
        print('criterion difference >= -0.02: %s' % ('met' if pm - mm >= -0.02
                                                     else 'not met'))


if __name__ == '__main__':
    main()
