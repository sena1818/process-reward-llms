from __future__ import annotations

import copy
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from prm_pref.data.datasets import ExactPrefixNodeCollator, IndexedJsonlDataset
from prm_pref.data.input_packing import build_input_packer
from prm_pref.models.encoder_reward_model import (
    RewardModel,
    build_reward_model,
    load_checkpoint_payload,
    save_reward_checkpoint,
)
from prm_pref.training.configuration import (
    training_signature,
    validate_training_config,
)
from prm_pref.training.losses import (
    hybrid_loss,
    nodewise_pairwise_loss,
    nodewise_pointwise_loss,
)
from prm_pref.utils.seed import seed_everything


VALID_MODES = {"pointwise", "pairwise", "hybrid"}


def _path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _device_from_config(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_device(
    tokens: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in tokens.items()
    }


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision not in {"fp16", "bf16"}:
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _build_grad_scaler(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision == "fp16"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _build_scheduler(
    optimizer: AdamW,
    *,
    total_steps: int,
    warmup_ratio: float,
) -> LambdaLR:
    warmup_steps = int(total_steps * warmup_ratio)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(total_steps - step, 0)
        decay_steps = max(total_steps - warmup_steps, 1)
        return float(remaining) / float(decay_steps)

    return LambdaLR(optimizer, multiplier)


def _forward_node_losses(
    *,
    model: RewardModel,
    batch: dict,
    mode: str,
    device: torch.device,
    lambda_pair: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = model(**_to_device(batch["tokens"], device))
    labels = batch["labels"].to(device)
    node_indices = batch["node_indices"].to(device)
    node_offsets = batch["node_offsets"].to(device)

    point_loss = (
        nodewise_pointwise_loss(
            logits,
            labels,
            node_indices,
            num_nodes=int(batch["num_nodes"]),
        )
        if mode in {"pointwise", "hybrid"}
        else None
    )
    pair_loss = (
        nodewise_pairwise_loss(logits, labels, node_offsets)
        if mode in {"pairwise", "hybrid"}
        else None
    )
    if mode == "pointwise" and point_loss is not None:
        total = point_loss
    elif mode == "pairwise" and pair_loss is not None:
        total = pair_loss
    elif mode == "hybrid" and point_loss is not None and pair_loss is not None:
        total = hybrid_loss(point_loss, pair_loss, lambda_pair)
    else:
        raise RuntimeError(f"Could not compute losses for mode={mode}")

    values = {"total": float(total.detach().float().cpu())}
    if point_loss is not None:
        values["pointwise"] = float(point_loss.detach().float().cpu())
    if pair_loss is not None:
        values["pairwise"] = float(pair_loss.detach().float().cpu())
    return total, values


def _empty_telemetry() -> dict[str, int]:
    return {
        "nodes": 0,
        "candidates": 0,
        "derived_pairs": 0,
        "tokens": 0,
        "problem_truncated": 0,
        "prefix_truncated": 0,
        "candidate_truncated": 0,
    }


def _add_batch_telemetry(target: dict[str, int], batch: dict) -> None:
    target["nodes"] += int(batch["num_nodes"])
    target["candidates"] += int(batch["num_candidates"])
    target["derived_pairs"] += int(batch["num_pairs"])
    packed = batch.get("telemetry", {})
    for key in (
        "tokens",
        "problem_truncated",
        "prefix_truncated",
        "candidate_truncated",
    ):
        target[key] += int(packed.get(key, 0))


@torch.no_grad()
def _validate(
    *,
    model: RewardModel,
    mode: str,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    lambda_pair: float,
    max_batches: int | None,
) -> tuple[dict[str, float], dict[str, int]]:
    model.eval()
    totals: dict[str, float] = {}
    total_nodes = 0
    telemetry = _empty_telemetry()
    for batch_index, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        with _autocast_context(device, precision):
            loss, values = _forward_node_losses(
                model=model,
                batch=batch,
                mode=mode,
                device=device,
                lambda_pair=lambda_pair,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite validation loss at batch {batch_index}"
            )
        batch_nodes = int(batch["num_nodes"])
        total_nodes += batch_nodes
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value * batch_nodes
        _add_batch_telemetry(telemetry, batch)
    if total_nodes == 0:
        raise RuntimeError("Validation data is empty")
    model.train()
    return (
        {key: value / total_nodes for key, value in totals.items()},
        telemetry,
    )


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ModuleNotFoundError:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "numpy" in state:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ModuleNotFoundError:
            pass
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _git_revision(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _environment_metadata(
    *,
    project_root: Path,
    device: torch.device,
    model: RewardModel,
) -> dict[str, Any]:
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "git_revision": _git_revision(project_root),
        "model_parameters": {
            "total": total,
            "trainable": trainable,
            "trainable_fraction": trainable / total if total else 0.0,
        },
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_JOB_PARTITION",
                "SLURM_JOB_NODELIST",
            )
            if os.environ.get(key)
        },
    }
    if torch.cuda.is_available():
        metadata["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "count_visible": torch.cuda.device_count(),
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    try:
        import transformers

        metadata["transformers"] = transformers.__version__
    except ModuleNotFoundError:
        pass
    try:
        import peft

        metadata["peft"] = peft.__version__
    except ModuleNotFoundError:
        pass
    return metadata


def _save_last_checkpoint(
    path: Path,
    *,
    model: RewardModel,
    model_config: dict[str, Any],
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: Any,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    history: list[dict],
    signature: str,
) -> None:
    _atomic_torch_save(
        {
            "format_version": 2,
            "checkpoint_type": "training_state",
            "model_state": model.checkpoint_state(),
            "model_config": model_config,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_validation_loss": best_validation_loss,
            "history": history,
            "signature": signature,
            "rng_state": _capture_rng_state(),
        },
        path,
    )


def run_training(
    *,
    project_root: Path,
    config: dict[str, Any],
    run_name: str | None = None,
    lambda_pair_override: float | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> Path:
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")

    effective_config = copy.deepcopy(config)
    run_cfg = effective_config.setdefault("run", {})
    mode = str(run_cfg.get("mode", "pointwise"))
    name = run_name or str(run_cfg.get("name", f"{mode}_v0"))
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("run name must be one non-empty path component")
    run_cfg["name"] = name
    train_cfg = effective_config.setdefault("training", {})
    lambda_pair = (
        float(lambda_pair_override)
        if lambda_pair_override is not None
        else float(train_cfg.get("lambda_pair", 0.3))
    )
    train_cfg["lambda_pair"] = lambda_pair
    validate_training_config(effective_config)

    model_cfg = dict(effective_config.get("model", {}))
    data_cfg = dict(effective_config.get("data", {}))
    seed = int(train_cfg.get("seed", 42))
    device = _device_from_config(str(train_cfg.get("device", "auto")))
    precision = str(train_cfg.get("mixed_precision", "fp16"))
    output_root = _path(
        project_root,
        effective_config.get("output_dir", "outputs/runs"),
    )
    run_dir = output_root / name
    last_checkpoint_path = run_dir / "last.pt"
    existing_files = list(run_dir.iterdir()) if run_dir.exists() else []
    if existing_files and not resume and not overwrite:
        raise FileExistsError(
            f"Run directory is not empty: {run_dir}. "
            "Use --resume or choose a new run name; use --overwrite only intentionally."
        )
    if resume and not last_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Cannot resume without an epoch checkpoint: {last_checkpoint_path}"
        )
    if existing_files and overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    resume_payload = (
        load_checkpoint_payload(last_checkpoint_path) if resume else None
    )
    if resume_payload is not None:
        saved_revision = resume_payload.get("model_config", {}).get(
            "revision"
        )
        if saved_revision:
            model_cfg["revision"] = saved_revision
            effective_config.setdefault("model", {})[
                "revision"
            ] = saved_revision

    seed_everything(seed)
    model_name = str(model_cfg.get("name_or_path", "roberta-base"))
    revision = model_cfg.get("revision")
    tokenizer_source = run_dir / "tokenizer" if resume else model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=True,
        revision=None if resume else revision,
    )
    tokenizer_revision = getattr(tokenizer, "init_kwargs", {}).get(
        "_commit_hash"
    )
    if not resume and not revision and tokenizer_revision:
        model_cfg["revision"] = tokenizer_revision
        effective_config.setdefault("model", {})[
            "revision"
        ] = tokenizer_revision
    if not resume:
        tokenizer.save_pretrained(run_dir / "tokenizer")

    model = build_reward_model(model_cfg).to(device)
    resolved_revision = getattr(model, "resolved_revision", None)
    if resolved_revision:
        model_cfg["revision"] = resolved_revision
        effective_config.setdefault("model", {})[
            "revision"
        ] = resolved_revision
    signature = training_signature(
        effective_config,
        lambda_pair=lambda_pair,
    )
    if resume_payload is not None and resume_payload.get("signature") != signature:
        raise ValueError(
            "Resume configuration differs from the saved training signature"
        )
    if resume_payload is not None:
        model.load_checkpoint_state(resume_payload["model_state"])
    packer = build_input_packer(tokenizer, model_cfg)

    staged_data_root = os.environ.get("PRM_DATA_ROOT")
    nodes_dir = (
        Path(staged_data_root) / "nodes_v0"
        if staged_data_root
        else _path(
            project_root,
            data_cfg.get("nodes_dir", "data/processed/nodes_v0"),
        )
    )
    train_data = IndexedJsonlDataset(
        nodes_dir / "train.jsonl",
        max_examples=_optional_int(data_cfg.get("max_train_nodes")),
        seed=seed,
    )
    val_data = IndexedJsonlDataset(
        nodes_dir / "val.jsonl",
        max_examples=_optional_int(data_cfg.get("max_val_nodes")),
        seed=seed + 1,
    )
    batch_size = int(train_cfg.get("batch_size", 4))
    eval_batch_size = int(train_cfg.get("eval_batch_size", batch_size))
    num_workers = int(train_cfg.get("num_workers", 0))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
        "collate_fn": ExactPrefixNodeCollator(packer),
    }
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    natural_steps = len(train_loader)
    configured_steps = _optional_int(train_cfg.get("max_steps_per_epoch"))
    steps_per_epoch = (
        natural_steps
        if configured_steps is None
        else min(natural_steps, configured_steps)
    )
    if steps_per_epoch == 0:
        raise RuntimeError("Training node cohort is empty")

    epochs = int(train_cfg.get("epochs", 3))
    accumulation = int(train_cfg.get("gradient_accumulation_steps", 1))
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("Model exposes no trainable parameters")
    optimizer = AdamW(
        trainable_parameters,
        lr=float(train_cfg.get("learning_rate", 2e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    total_updates = math.ceil(steps_per_epoch / accumulation) * epochs
    scheduler = _build_scheduler(
        optimizer,
        total_steps=total_updates,
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.06)),
    )
    scaler = _build_grad_scaler(device, precision)

    history: list[dict] = []
    best_validation_loss = math.inf
    global_step = 0
    start_epoch = 1
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        scaler.load_state_dict(resume_payload.get("scaler_state_dict", {}))
        history = list(resume_payload.get("history", []))
        best_validation_loss = float(
            resume_payload.get("best_validation_loss", math.inf)
        )
        global_step = int(resume_payload.get("global_step", 0))
        start_epoch = int(resume_payload["epoch"]) + 1
        _restore_rng_state(resume_payload["rng_state"])

    environment = _environment_metadata(
        project_root=project_root,
        device=device,
        model=model,
    )
    stats_path = nodes_dir / "stats.json"
    environment["data"] = {
        "nodes_dir": str(nodes_dir),
        "stats": (
            json.loads(stats_path.read_text(encoding="utf-8"))
            if stats_path.exists()
            else None
        ),
    }
    (run_dir / "resolved_config.json").write_text(
        json.dumps(effective_config, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "run": name,
                "mode": mode,
                "backend": model_cfg.get("backend", "encoder"),
                "device": str(device),
                "model": model_name,
                "lambda_pair": lambda_pair if mode == "hybrid" else None,
                "train_nodes": len(train_data),
                "val_nodes": len(val_data),
                "steps_per_epoch": steps_per_epoch,
                "epochs": epochs,
                "start_epoch": start_epoch,
                "trainable_parameters": environment["model_parameters"][
                    "trainable"
                ],
            },
            indent=2,
        )
    )

    if start_epoch > epochs:
        print(f"Run already completed through epoch {epochs}: {run_dir}")
        return run_dir

    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    log_every = int(train_cfg.get("log_every", 100))
    max_validation_batches = _optional_int(
        train_cfg.get("max_validation_batches")
    )
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_started = time.time()
        running = 0.0
        telemetry = _empty_telemetry()
        train_iterator = iter(train_loader)
        for local_step in range(1, steps_per_epoch + 1):
            batch = next(train_iterator)
            window_start = ((local_step - 1) // accumulation) * accumulation + 1
            window_end = min(window_start + accumulation - 1, steps_per_epoch)
            window_nodes = sum(
                (
                    len(train_data)
                    - batch_size * (natural_steps - 1)
                    if step == natural_steps
                    else batch_size
                )
                for step in range(window_start, window_end + 1)
            )
            with _autocast_context(device, precision):
                loss, _ = _forward_node_losses(
                    model=model,
                    batch=batch,
                    mode=mode,
                    device=device,
                    lambda_pair=lambda_pair,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch={epoch}, "
                    f"step={local_step}"
                )
            batch_weight = int(batch["num_nodes"]) / window_nodes
            scaler.scale(loss * batch_weight).backward()
            loss_value = float(loss.detach().float().cpu())
            running += loss_value * int(batch["num_nodes"])
            _add_batch_telemetry(telemetry, batch)

            should_update = local_step == window_end
            if should_update:
                scaler.unscale_(optimizer)
                clip_grad_norm_(trainable_parameters, max_grad_norm)
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_stepped = (
                    not scaler.is_enabled()
                    or scaler.get_scale() >= scale_before_step
                )
                if optimizer_stepped:
                    scheduler.step()
                    global_step += 1
                optimizer.zero_grad(set_to_none=True)

            if log_every and local_step % log_every == 0:
                elapsed = max(time.time() - epoch_started, 1e-9)
                print(
                    f"epoch={epoch} step={local_step}/{steps_per_epoch} "
                    f"loss={running / telemetry['nodes']:.5f} "
                    f"nodes/s={telemetry['nodes'] / elapsed:.2f} "
                    f"candidates/s={telemetry['candidates'] / elapsed:.2f}"
                )

        validation, validation_telemetry = _validate(
            model=model,
            mode=mode,
            loader=val_loader,
            device=device,
            precision=precision,
            lambda_pair=lambda_pair,
            max_batches=max_validation_batches,
        )
        elapsed_seconds = time.time() - epoch_started
        row = {
            "epoch": epoch,
            "train_loss": running / telemetry["nodes"],
            "validation": validation,
            "elapsed_seconds": elapsed_seconds,
            "optimizer_steps": global_step,
            "train_telemetry": telemetry,
            "validation_telemetry": validation_telemetry,
            "throughput": {
                "nodes_per_second": telemetry["nodes"] / elapsed_seconds,
                "candidates_per_second": (
                    telemetry["candidates"] / elapsed_seconds
                ),
                "tokens_per_second": telemetry["tokens"] / elapsed_seconds,
            },
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        if validation["total"] < best_validation_loss:
            best_validation_loss = validation["total"]
            save_reward_checkpoint(
                run_dir / "best.pt",
                model=model,
                model_config=model_cfg,
                training_config={
                    "run_name": name,
                    "mode": mode,
                    "lambda_pair": lambda_pair if mode == "hybrid" else None,
                    "seed": seed,
                    "train_nodes": len(train_data),
                    "val_nodes": len(val_data),
                    "configured_max_train_nodes": data_cfg.get(
                        "max_train_nodes"
                    ),
                    "signature": signature,
                },
                epoch=epoch,
                validation_loss=best_validation_loss,
            )

        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )
        _save_last_checkpoint(
            last_checkpoint_path,
            model=model,
            model_config=model_cfg,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_validation_loss=best_validation_loss,
            history=history,
            signature=signature,
        )

    metadata = {
        "status": "completed",
        "run_name": name,
        "mode": mode,
        "backend": model_cfg.get("backend", "encoder"),
        "lambda_pair": lambda_pair if mode == "hybrid" else None,
        "seed": seed,
        "train_nodes": len(train_data),
        "val_nodes": len(val_data),
        "best_validation_loss": best_validation_loss,
        "best_checkpoint": str(run_dir / "best.pt"),
        "resume_checkpoint": str(last_checkpoint_path),
        "threshold_status": "not_calibrated_during_training",
        "threshold_rule": (
            "calibrate on validation trajectories in scripts/07_eval_all.py"
        ),
    }
    (run_dir / "run.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir
