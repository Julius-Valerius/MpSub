#!/bin/bash
# Convergence curves at matched forward-pass budget.
#
# With batch size fixed, MeZO spends 2 forwards per step and MpSub with p = 20
# spends 42, so 4200 MeZO steps and 200 MpSub steps are the same budget of
# 8400 forwards; the summary aligns both on the forward-pass axis. Logs
# containing an accuracy line are skipped, so the script resumes on rerun.
#
# Set CODE_DIR to the directory holding run_mezo.py and run_lozo.py, PY to
# the interpreter, and MODEL_PATH to a local copy of the model.
#
#   bash run_mezo_horizon.sh
#   LRS="1e-6 3e-7" bash run_mezo_horizon.sh
#   RUN_PSUB=1 bash run_mezo_horizon.sh
#   STEPS=3000 SEEDS="0" bash run_mezo_horizon.sh
cd "${CODE_DIR:-$(dirname "$0")/../llm}" || exit 1

PY=${PY:-python}
MODEL_PATH=${MODEL_PATH:-}
OUT=${OUT:-result/mezo_horizon}
mkdir -p $OUT
if [ -n "$MODEL_PATH" ]; then
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    MODEL_ARG="--model_path $MODEL_PATH"
else
    MODEL_ARG=""
fi

STEPS=${STEPS:-4200}
LRS=${LRS:-"1e-6"}
SEEDS=${SEEDS:-"0 1 2"}

for LR in $LRS; do
for SD in $SEEDS; do
    LOG=$OUT/mezo_lr${LR}_seed${SD}.log
    if [ -s "$LOG" ] && grep -q "'accuracy'" "$LOG"; then
        echo "=== skip (done): $LOG ==="; continue
    fi
    echo "=== MeZO lr=$LR seed=$SD steps=$STEPS ($((STEPS*2)) forwards) ==="
    $PY -u run_mezo.py \
        --model_name facebook/opt-125m $MODEL_ARG \
        --task_name CB \
        --output_dir $OUT/mezo_lr${LR}_s${SD} --tag mezo_hz_lr${LR}_s${SD} \
        --train_set_seed $SD --num_train 100 --num_dev 50 --num_eval 100 \
        --logging_steps 25 --max_steps $STEPS \
        --trainer zo --learning_rate $LR --zo_eps 1e-3 \
        --per_device_train_batch_size 8 --lr_scheduler_type constant \
        --evaluation_strategy steps --eval_steps 250 --save_strategy no \
        --train_as_classification \
        2>&1 | tee $LOG | grep -E "eval_loss|accuracy"
done
done

if [ "${RUN_PSUB:-0}" = "1" ]; then
for D in 1e-1 1.0; do
for SD in $SEEDS; do
    LOG=$OUT/psub_d${D}_seed${SD}.log
    if [ -s "$LOG" ] && grep -q "'accuracy'" "$LOG"; then
        echo "=== skip (done): $LOG ==="; continue
    fi
    echo "=== psub p=20 delta0=$D seed=$SD 200 steps (8400 forwards) ==="
    $PY -u run_lozo.py \
        --model_name facebook/opt-125m $MODEL_ARG \
        --task_name CB \
        --output_dir $OUT/psub_d${D}_s${SD} --tag psub_hz_d${D}_s${SD} \
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
echo "==== summary, x axis in forward passes (MeZO step*2, psub step*42) ===="
OUT=$OUT $PY - <<'PYEOF'
import glob, re, os
import statistics as st
OUT = os.environ.get('OUT', 'result/mezo_horizon')

def parse(path):
    evs, acc = [], None
    for ln in open(path, errors='ignore'):
        m = re.search(r"'eval_loss':\s*([0-9.]+)", ln)
        if m:
            evs.append(float(m.group(1)))
        m = re.search(r"\{'accuracy':\s*([0-9.]+),\s*'dev_accuracy':\s*([0-9.]+)\}", ln)
        if m:
            acc = (float(m.group(1)), float(m.group(2)))
    return evs, acc

for pat in [os.path.join(OUT, 'mezo_lr*_seed*.log'),
            os.path.join(OUT, 'psub_d*_seed*.log')]:
    accs = []
    for f in sorted(glob.glob(pat)):
        evs, acc = parse(f)
        tail = ' -> '.join('%.3f' % v for v in evs[-4:]) if evs else '(no eval)'
        a = 'test=%.3f dev=%.2f' % acc if acc else '(incomplete)'
        print('%-44s %s | eval tail: %s' % (os.path.basename(f), a, tail))
        if acc:
            accs.append(acc[0])
    if accs:
        print('  test acc: mean=%.3f min=%.3f max=%.3f (n=%d seeds)'
              % (st.mean(accs), min(accs), max(accs), len(accs)))
    print()
PYEOF
