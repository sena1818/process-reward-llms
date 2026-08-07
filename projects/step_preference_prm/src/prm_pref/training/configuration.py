from __future__ import annotations

import copy
import hashlib
import json
from typing import Any


VALID_MODES = {"pointwise", "pairwise", "hybrid"}
VALID_BACKENDS = {"encoder", "causal_lora"}
VALID_PRECISIONS = {"fp32", "fp16", "bf16"}
VALID_DEVICES = {"auto", "cpu", "cuda", "mps"}


def _cuda_is_target(device: str) -> bool:
    """Whether the configured device will actually resolve to CUDA."""

    if device.startswith("cuda"):
        return True
    if device != "auto":
        return False
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return bool(torch.cuda.is_available())


def with_smoke_profile(config: dict[str, Any]) -> dict[str, Any]:
    """Return a tiny config that exercises real tokenization and backprop.

    Off CUDA the profile also drops to fp32.  A smoke run checks that the code
    path is correct, not that reduced precision is fast, and bf16 matmuls on
    CPU fall back to a slow unaccelerated kernel that can turn five steps into
    many minutes.  On CUDA the configured precision is kept so the cluster
    smoke still exercises the production autocast path.
    """

    result = copy.deepcopy(config)
    model = result.setdefault("model", {})
    data = result.setdefault("data", {})
    training = result.setdefault("training", {})
    if not _cuda_is_target(str(training.get("device", "auto"))):
        model["load_dtype"] = "fp32"
        training["mixed_precision"] = "fp32"
    model["max_length"] = 128
    # Checkpointing trades compute for memory, and a smoke batch needs no
    # memory relief.  Production runs keep it enabled.
    model["gradient_checkpointing"] = False
    data["max_train_nodes"] = 64
    data["max_val_nodes"] = 32
    training["epochs"] = 1
    training["batch_size"] = 2
    training["eval_batch_size"] = 4
    training["gradient_accumulation_steps"] = 1
    training["max_steps_per_epoch"] = 5
    training["max_validation_batches"] = 2
    training["log_every"] = 1
    result["output_dir"] = f"{result.get('output_dir', 'outputs/runs')}_smoke"
    return result


def with_pilot_profile(config: dict[str, Any]) -> dict[str, Any]:
    """Preserve production shapes but cap work for throughput measurement."""

    result = copy.deepcopy(config)
    data = result.setdefault("data", {})
    training = result.setdefault("training", {})
    data["max_train_nodes"] = min(
        int(data.get("max_train_nodes") or 4096),
        4096,
    )
    data["max_val_nodes"] = min(
        int(data.get("max_val_nodes") or 512),
        512,
    )
    training["epochs"] = 1
    training["max_steps_per_epoch"] = 200
    training["max_validation_batches"] = 25
    training["log_every"] = 10
    result["output_dir"] = f"{result.get('output_dir', 'outputs/runs')}_pilot"
    return result


def with_experiment_overrides(
    config: dict[str, Any],
    *,
    seed: int | None = None,
    max_train_nodes: int | None = None,
    model_name: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Apply explicit, reproducible command-line experiment overrides.

    ``model_name`` and ``device`` exist so a laptop can exercise the real
    causal-LoRA code path with a small stand-in backbone.  They must never be
    used for a reported run: both are recorded in the resolved config and in
    the resume signature, so a substituted backbone can always be detected.
    """

    result = copy.deepcopy(config)
    if seed is not None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        result.setdefault("training", {})["seed"] = seed
    if max_train_nodes is not None:
        if max_train_nodes <= 0:
            raise ValueError("max_train_nodes must be positive")
        result.setdefault("data", {})["max_train_nodes"] = max_train_nodes
    if model_name is not None:
        if not model_name.strip():
            raise ValueError("model_name must be a non-empty identifier")
        model = result.setdefault("model", {})
        model["name_or_path"] = model_name
        # A substituted backbone invalidates any pinned upstream revision.
        model.pop("revision", None)
    if device is not None:
        if device not in VALID_DEVICES and not device.startswith("cuda:"):
            raise ValueError(
                f"device must be cuda[:n] or one of {sorted(VALID_DEVICES)}"
            )
        result.setdefault("training", {})["device"] = device
    return result


def validate_training_config(config: dict[str, Any]) -> None:
    run = dict(config.get("run", {}))
    model = dict(config.get("model", {}))
    data = dict(config.get("data", {}))
    training = dict(config.get("training", {}))
    mode = str(run.get("mode", ""))
    backend = str(model.get("backend", "encoder"))
    precision = str(training.get("mixed_precision", "fp32"))

    if mode not in VALID_MODES:
        raise ValueError(f"run.mode must be one of {sorted(VALID_MODES)}")
    if backend not in VALID_BACKENDS:
        raise ValueError(f"model.backend must be one of {sorted(VALID_BACKENDS)}")
    if precision not in VALID_PRECISIONS:
        raise ValueError(
            f"training.mixed_precision must be one of {sorted(VALID_PRECISIONS)}"
        )
    if not str(model.get("name_or_path", "")).strip():
        raise ValueError("model.name_or_path is required")
    if int(model.get("max_length", 0)) <= 0:
        raise ValueError("model.max_length must be positive")
    if not str(data.get("nodes_dir", "")).strip():
        raise ValueError("data.nodes_dir is required for strict V0 training")

    positive_ints = {
        "batch_size": training.get("batch_size", 0),
        "eval_batch_size": training.get("eval_batch_size", 0),
        "gradient_accumulation_steps": training.get(
            "gradient_accumulation_steps", 0
        ),
        "epochs": training.get("epochs", 0),
    }
    for key, value in positive_ints.items():
        if int(value) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if int(training.get("num_workers", 0)) < 0:
        raise ValueError("training.num_workers must be non-negative")
    for key in ("max_steps_per_epoch", "max_validation_batches"):
        value = training.get(key)
        if value is not None and int(value) <= 0:
            raise ValueError(f"training.{key} must be positive when set")
    if float(training.get("learning_rate", 0.0)) <= 0:
        raise ValueError("training.learning_rate must be positive")
    if not 0.0 <= float(training.get("warmup_ratio", 0.0)) < 1.0:
        raise ValueError("training.warmup_ratio must be in [0, 1)")
    if mode == "hybrid" and float(training.get("lambda_pair", 0.0)) < 0:
        raise ValueError("training.lambda_pair must be non-negative")
    for key in ("max_train_nodes", "max_val_nodes"):
        value = data.get(key)
        if value is not None and int(value) <= 0:
            raise ValueError(f"data.{key} must be positive when set")
    if backend == "causal_lora":
        lora = dict(model.get("lora", {}))
        if int(lora.get("rank", 0)) <= 0:
            raise ValueError("model.lora.rank must be positive")
        if int(lora.get("alpha", 0)) <= 0:
            raise ValueError("model.lora.alpha must be positive")
        if not 0.0 <= float(lora.get("dropout", 0.0)) < 1.0:
            raise ValueError("model.lora.dropout must be in [0, 1)")
        if str(lora.get("bias", "none")) not in {
            "none",
            "all",
            "lora_only",
        }:
            raise ValueError(
                "model.lora.bias must be none, all, or lora_only"
            )
        if str(model.get("load_dtype", "bf16")) not in VALID_PRECISIONS:
            raise ValueError(
                f"model.load_dtype must be one of {sorted(VALID_PRECISIONS)}"
            )
        packing = dict(model.get("packing", {}))
        for key in ("max_problem_tokens", "max_candidate_tokens"):
            if int(packing.get(key, 0)) <= 0:
                raise ValueError(f"model.packing.{key} must be positive")


def training_signature(
    config: dict[str, Any],
    *,
    lambda_pair: float,
) -> str:
    """Fingerprint all settings that must stay fixed across resume."""

    payload = {
        "run": config.get("run", {}),
        "model": config.get("model", {}),
        "data": config.get("data", {}),
        "training": config.get("training", {}),
        "lambda_pair": lambda_pair,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()
