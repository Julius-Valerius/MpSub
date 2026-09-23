# MpSub: A Momentum $p$-Dimensional Subspace Trust-Region Method for Derivative-Free Fine-Tuning of Large Language Models

Official PyTorch implementation and reproduction package for the paper:
> **MpSub: A Momentum $p$-Dimensional Subspace Trust-Region Method for Derivative-Free Fine-Tuning of Large Language Models**  
> *Yuyang Wang, Haoyu Yao, and Pengcheng Xie*

---

## 🌟 Overview

Full-parameter fine-tuning of large language models (LLMs) via zeroth-order (ZO) optimization eliminates the need for backpropagation, reducing the training memory footprint to that of inference. However, existing ZO methods (such as MeZO) rely on a delicate learning rate that must be tuned separately across models and tasks.

**MpSub** introduces a momentum-guided, $p$-dimensional subspace trust-region framework tailored for high-dimensional black-box language model tuning:
- **Momentum + Exploration**: Spans a $p$-dimensional search subspace where $\bm{d}_1$ inherits the normalized displacement of the latest accepted step, while $\bm{d}_2, \dots, \bm{d}_p$ provide isotropic random exploration.
- **Tuning-Free Step Size**: The adaptive trust-region radius $\Delta_k$ dynamically regulates both the sampling resolution and the coordinate step size based on model-objective agreement ($\rho_k \ge \eta$). A universal default $\Delta_0 = 10^{-1}$ achieves accuracy competitive with grid-searched MeZO across multiple model scales.
- **Inference-Level Memory**: Search directions are regenerated deterministically on-device via PRNG seeds, avoiding materialization of the $n \times p$ frame matrix in GPU memory. Auxiliary memory is strictly $\mathcal{O}(n)$, identical to inference.
- **Matched-Budget Efficiency**: Under a matched budget of 8,400 training forward passes, MpSub matches or outperforms MeZO on OPT-125M and OPT-350M on CommitmentBank without a costly hyperparameter sweep, saving $3\times$ to $5\times$ compute.

---

## 📁 Repository Structure

```text
MpSub/
├── llm/                      # Core optimizer and LLM fine-tuning framework
│   ├── psub_torch.py         # GPU in-place MpSub solver (fast path)
│   ├── psub_solver.py        # CPU reference solver
│   ├── LOZOtrainer.py        # Custom Trainer integrating MpSub
│   ├── run_lozo.py           # Training entry point for MpSub / ZO methods
│   ├── run_mezo.py           # Training entry point for MeZO baseline
│   ├── data/cb/              # CommitmentBank dataset splits
│   ├── test_psub_torch.py    # Unit tests for GPU in-place operations & recovery
│   ├── test_psub_local.py    # Local integration tests
│   └── requirements.txt      # Pinned dependency environment
│
├── experiments/              # Experiment driver and reproduction scripts
│   ├── run_hp_cost.sh        # Grid comparison on OPT-125M (MpSub vs MeZO)
│   ├── run_sensitivity.py    # Initial radius sensitivity runner across 3 seeds
│   ├── run_p_ablation_equal_budget.py # Subspace dimension p ablation suite
│   └── run_mezo_horizon.sh   # Extended MeZO baseline runs
│
├── results/                  # Raw evaluation logs and structured summaries
│   ├── hp_cost/              # Logs for MeZO lr sweeps and MpSub delta0 sweeps
│   ├── scaling/              # OPT-125M and OPT-350M matched-budget results & curves
│   └── p_ablation_equal_budget/ # Equal-budget ablation results across seeds
│
├── paper/                    # Paper manuscript, styles, and figures
│   ├── mpsub.tex             # SIAM format LaTeX source
│   ├── mpsub.pdf             # Precompiled manuscript
│   ├── refs.bib              # Bibliography
│   ├── make_figures.py       # Script to regenerate paper figures from logs
│   ├── fig_convergence_models.pdf # Figure 1: Convergence trajectories
│   ├── fig_sensitivity.pdf        # Figure 2: Step-size sensitivity
│   └── fig_p_ablation.pdf         # Figure 3: Subspace dimension ablation
│
├── surrogate/                # Standalone synthetic & toy problem benchmarks
│   ├── surrogate_history_subspace.py
│   ├── surrogate_dimension_model.py
│   ├── surrogate_estimation_steprule.py
│   └── surrogate_eval_recycle.py
│
├── .gitignore                # Git ignore configuration
└── README.md                 # This document
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.11 (strictly recommended; `tokenizers==0.13.3` provides pre-built wheels for cp311)
- CUDA 11.8+ or compatible GPU environment

```bash
# Clone the repository
git clone https://github.com/PengchengXieLSEC/MpSub.git
cd MpSub

# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r llm/requirements.txt
```

Verify the installation and optimizer implementation with unit tests:
```bash
python llm/test_psub_torch.py
python llm/test_psub_local.py
```

---

## 🚀 Quick Start

### Fine-Tuning OPT-125M with MpSub
To run MpSub on CommitmentBank with the default radius $\Delta_0 = 0.1$ and subspace dimension $p = 20$ (200 steps $\times$ 42 passes = 8,400 forward passes):

```bash
cd llm
python run_lozo.py \
    --model_name facebook/opt-125m \
    --task_name CB \
    --num_train 100 \
    --num_dev 50 \
    --num_eval 100 \
    --per_device_train_batch_size 8 \
    --lr_scheduler_type constant \
    --learning_rate 0 \
    --max_steps 200 \
    --eval_steps 25 \
    --use_psub True \
    --psub_p 20 \
    --psub_delta0 0.1 \
    --psub_nsteps 1 \
    --psub_use_torch True \
    --train_as_classification \
    --output_dir ./out/opt-125m-mpsub
```

### Running the MeZO Baseline
```bash
python run_mezo.py \
    --model_name facebook/opt-125m \
    --task_name CB \
    --num_train 100 \
    --num_dev 50 \
    --num_eval 100 \
    --per_device_train_batch_size 8 \
    --lr_scheduler_type constant \
    --learning_rate 1e-6 \
    --max_steps 4200 \
    --eval_steps 300 \
    --zo_eps 1e-3 \
    --train_as_classification \
    --output_dir ./out/opt-125m-mezo
```

---

## 📊 Reproducing Paper Results

### 1. Main Matched-Budget Comparison (Table 1 & Figure 1)
Results for OPT-125M and OPT-350M across three random seeds under the 8,400 forward-pass budget are summarized in `results/scaling/`:
- `results/scaling/SUMMARY_opt-125m.txt`
- `results/scaling/SUMMARY_opt-350m.txt`

### 2. Step-Size Sensitivity (Figure 2)
To reproduce the initial radius sweep across $\Delta_0 \in \{10^{-3}, 10^{-2}, 3\times 10^{-2}, 10^{-1}, 3\times 10^{-1}\}$:
```bash
python experiments/run_sensitivity.py
```

### 3. Subspace Dimension Ablation (Table 2)
To evaluate the impact of subspace dimension $p \in \{5, 10, 15, 20, 25, 30\}$:
```bash
python experiments/run_p_ablation_equal_budget.py
```

### 4. Regenerating Paper Figures
```bash
python paper/make_figures.py results/hp_cost paper
```

---

## 📝 Citation

If you find this work or codebase helpful in your research, please cite:

```bibtex
@article{wang2026mpsub,
  title   = {MpSub: A Momentum $p$-Dimensional Subspace Trust-Region Method for Derivative-Free Fine-Tuning of Large Language Models},
  author  = {Wang, Yuyang and Yao, Haoyu and Xie, Pengcheng},
  journal = {Preprint},
  year    = {2026}
}
```

---

## 📄 License
This project is released under the [Apache-2.0 License](LICENSE).
