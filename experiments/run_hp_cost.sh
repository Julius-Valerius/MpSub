#!/bin/bash
# Matched-compute total-cost experiment, serial driver.
#
# Every configuration receives 8400 forward passes: MeZO at 4200 steps
# (2 forwards per step) over five learning rates, MpSub at 200 steps with
# p = 20 (42 forwards per step) over five initial radii, three seeds each.
# The MpSub grid brackets the representability window from the ulp analysis,
# with three values inside it and two outside. Log names match
# run_hp_cost_parallel.py, and logs already containing an accuracy line are
# skipped, so an interrupted run resumes by re-issuing the same command.
#
# Set CODE_DIR to the directory holding run_mezo.py and run_lozo.py, PY to
# the interpreter, and MODEL_PATH to a local copy of the model.
#
#   bash run_hp_cost.sh
#   MEZO_ONLY=1 bash run_hp_cost.sh
#   PSUB_ONLY=1 bash run_hp_cost.sh
cd "${CODE_DIR:-$(dirname "$0")/../llm}" || exit 1

PY=${PY:-python}
MODEL_PATH=${MODEL_PATH:-}
OUT=${OUT:-result/hp_cost}
mkdir -p $OUT
if [ -n "$MODEL_PATH" ]; then
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    MODEL_ARG="--model_path $MODEL_PATH"
else
    MODEL_ARG=""
fi

SEEDS=${SEEDS:-"0 1 2"}
MSTEPS=${MSTEPS:-4200}
MLRS=${MLRS:-"1e-4 1e-5 1e-6 1e-7 1e-8"}
PDELTAS=${PDELTAS:-"1e-1 3e-1 1.0 3.0 10.0"}

if [ "${PSUB_ONLY:-0}" != "1" ]; then
for LR in $MLRS; do
for SD in $SEEDS; do
    LOG=$OUT/mezo_lr${LR}_s${SD}.log
    if [ -s "$LOG" ] && grep -q "'accuracy'" "$LOG"; then
        echo "=== skip (done): $LOG ==="; continue
    fi
    echo "=== MeZO lr=$LR seed=$SD ${MSTEPS} steps ($((MSTEPS*2)) forwards) ==="
    $PY -u run_mezo.py \
        --model_name facebook/opt-125m $MODEL_ARG \
        --task_name CB \
        --output_dir $OUT/mz_${LR}_${SD} --tag hpc_mz_${LR}_${SD} \
        --train_set_seed $SD --num_train 100 --num_dev 50 --num_eval 100 \
        --logging_steps 50 --max_steps $MSTEPS \
        --trainer zo --learning_rate $LR --zo_eps 1e-3 \
        --per_device_train_batch_size 8 --lr_scheduler_type constant \
        --evaluation_strategy steps --eval_steps 300 --save_strategy no \
        --train_as_classification \
        2>&1 | tee $LOG | grep -E "eval_loss|accuracy"
done
done
fi

if [ "${MEZO_ONLY:-0}" != "1" ]; then
for D in $PDELTAS; do
for SD in $SEEDS; do
    LOG=$OUT/psub_d${D}_s${SD}.log
    if [ -s "$LOG" ] && grep -q "'accuracy'" "$LOG"; then
        echo "=== skip (done): $LOG ==="; continue
    fi
    echo "=== psub p=20 delta0=$D seed=$SD 200 steps (8400 forwards) ==="
    $PY -u run_lozo.py \
        --model_name facebook/opt-125m $MODEL_ARG \
        --task_name CB \
        --output_dir $OUT/ps_${D}_${SD} --tag hpc_ps_${D}_${SD} \
        --train_set_seed $SD --num_train 100 --num_dev 50 --num_eval 100 \
        --logging_steps 5 --max_steps 200 --trainer LOZO \
        --per_device_train_batch_size 8 --lr_scheduler_type constant --learning_rate 0 \
        --evaluation_strategy steps --eval_steps 25 --save_strategy no \
        --train_as_classification \
        --use_psub True --psub_p 20 --psub_delta0 $D --psub_nsteps 1 \
        --psub_kind grad --psub_use_torch True \
        2>&1 | tee $LOG | grep -E "eval_loss|accuracy"
done
done
fi

echo ""
echo "==== total cost including hyperparameter search ===="
OUT=$OUT $PY - <<'PYEOF'
import glob, re, os
import statistics as st
OUT = os.environ.get('OUT', 'result/hp_cost')

def acc(path):
    a = None
    for ln in open(path, errors='ignore'):
        m = re.search(r"\{'accuracy':\s*([0-9.]+),\s*'dev_accuracy':\s*([0-9.]+)\}", ln)
        if m:
            a = float(m.group(1))
    return a

def table(pat, key):
    groups = {}
    for f in sorted(glob.glob(pat)):
        m = re.search(key + r'([0-9e.+-]+)_s(\d+)', os.path.basename(f))
        if not m:
            continue
        a = acc(f)
        if a is not None:
            groups.setdefault(m.group(1), []).append(a)
    return [(hp, st.mean(v), min(v), max(v), len(v)) for hp, v in groups.items()]

best = {}
for name, pat, key in [('MeZO', os.path.join(OUT, 'mezo_lr*_s*.log'), 'lr'),
                       ('psub', os.path.join(OUT, 'psub_d*_s*.log'), 'd')]:
    rows = table(pat, key)
    for hp, mean, lo, hi, ns in sorted(rows, key=lambda r: -r[1]):
        print('%-6s %-8s mean=%-7.3f (%.3f~%.3f, n=%d)' % (name, hp, mean, lo, hi, ns))
    if rows:
        b = max(rows, key=lambda r: r[1])
        best[name] = b
        print('  %s best: %s=%s mean=%.3f | %d grid points => %dx8400 forwards'
              % (name, key, b[0], b[1], len(rows), len(rows)))
    print()
d01 = [a for a in (acc(f) for f in glob.glob(os.path.join(OUT, 'psub_d1e-1_s*.log')))
       if a is not None]
if d01 and 'MeZO' in best:
    pm, mm = st.mean(d01), best['MeZO'][1]
    print('psub at the default delta0=0.1 mean=%.3f vs tuned MeZO mean=%.3f, '
          'difference=%+.3f' % (pm, mm, pm - mm))
    print('criterion difference >= -0.02: %s' % ('met' if pm - mm >= -0.02 else 'not met'))
PYEOF
