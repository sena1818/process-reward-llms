# V0 Experiment Runbook

> **Historical encoder-sanity runbook.** The strict main experiment is now
> Qwen2.5-Math-1.5B LoRA on a shared node-balanced cohort. Use
> [`TRAINING_AUDIT_AND_UNICLUSTER_GUIDE.md`](TRAINING_AUDIT_AND_UNICLUSTER_GUIDE.md)
> for current commands and compute guidance.

## What is trained

The V0 reward model is a bidirectional encoder plus one scalar head:

```text
problem + previous steps + candidate step
    -> RoBERTa/DeBERTa encoder
    -> scalar reward logit
```

The encoder and scalar head are fine-tuned end to end. This does **not** train
the solver/generator LLM and it does not use RL. Pointwise, pairwise, and hybrid
runs use the same scorer architecture so the comparison isolates the training
objective.

## Data pipeline

From the project root:

```bash
python scripts/01_build_splits.py
python scripts/02_build_pairs.py
python scripts/03_build_pointwise.py
python scripts/validate_v0_data.py
```

The full processed data is about 1.36 GB. If moving to a cloud machine, it is
usually cheaper to transfer the roughly 455 MB raw JSONL files and rebuild the
processed files there.

## Environment and smoke tests

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python scripts/04_train_pointwise.py --smoke
python scripts/05_train_pairwise.py --smoke
python scripts/06_train_hybrid.py --smoke
```

Each smoke run uses 64 training examples, sequence length 128, and five update
steps. It checks model download, tokenization, forward/backward, checkpointing,
and both loss paths; it is not a meaningful experiment.

## Main small-scale experiment

The checked-in encoder configs now use the same strict cohort as the main
experiment: `roberta-base`, sequence length 512, 47,852 training nodes, and
5,751 validation nodes.

```bash
python scripts/04_train_pointwise.py
python scripts/05_train_pairwise.py
python scripts/06_train_hybrid.py
python scripts/07_eval_all.py
```

The hybrid command trains all four required runs:

```text
hybrid_v0_lam0.1
hybrid_v0_lam0.3
hybrid_v0_lam0.5
hybrid_v0_lam1.0
```

`07_eval_all.py` scores every run on pointwise, pairwise, and first-error data.
For each checkpoint, it calibrates the step-classification and first-error
thresholds using validation data only, fixes them, and then evaluates test
data. It selects the main hybrid lambda by validation first-error within +/-1,
with validation pairwise accuracy as tie-breaker.

## Recommended model progression

1. `roberta-base` (about 125M parameters): V0 default and easiest baseline.
2. `microsoft/deberta-v3-base` (86M backbone plus a large embedding table,
   about 184M total): a stronger encoder replication if compute permits.
3. Optional only after V0: a 1.5B math LLM with LoRA/QLoRA and a scalar head.

Do not replace the main controlled comparison with a 7B model. One base encoder
plus the six V0 objectives already provides a defensible seminar experiment;
model-scale ablations are secondary.

## Hardware expectations

- 8 GB GPU: batch size 1-2, fp16, gradient checkpointing, accumulation; V0 is
  feasible but the six-run sweep will be slow.
- 16-24 GB GPU: recommended for the full V0 sweep; increase batch size after a
  smoke run and monitor peak memory.
- 24 GB GPU: also sufficient for an optional 1.5B QLoRA extension.
- 48 GB GPU: useful only if adding a 7B QLoRA extension; not required for V0.

Keep the pointwise/pairwise/hybrid data budgets and backbone identical in the
main table. Otherwise a result can be caused by model or data scale rather than
the pairwise objective.
