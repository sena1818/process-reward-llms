from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class EncoderRewardModel(nn.Module):
    """A bidirectional encoder with one scalar reward logit per step."""

    backend = "encoder"

    def __init__(
        self,
        model_name_or_path: str,
        *,
        dropout: float | None = None,
        gradient_checkpointing: bool = False,
        revision: str | None = None,
        encoder_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        if encoder_config is None:
            self.encoder = AutoModel.from_pretrained(
                model_name_or_path,
                revision=revision,
            )
        else:
            config_payload = dict(encoder_config)
            model_type = config_payload.pop("model_type")
            self.encoder = AutoModel.from_config(
                AutoConfig.for_model(model_type, **config_payload)
            )
        hidden_size = int(self.encoder.config.hidden_size)
        self.resolved_revision = getattr(
            self.encoder.config,
            "_commit_hash",
            revision,
        )
        default_dropout = float(
            getattr(self.encoder.config, "hidden_dropout_prob", 0.1)
        )
        self.dropout = nn.Dropout(
            default_dropout if dropout is None else float(dropout)
        )
        self.reward_head = nn.Linear(hidden_size, 1)
        if gradient_checkpointing and hasattr(
            self.encoder, "gradient_checkpointing_enable"
        ):
            self.encoder.gradient_checkpointing_enable()

    def forward(self, **tokens: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(**tokens)
        pooled = outputs.last_hidden_state[:, 0]
        return self.reward_head(self.dropout(pooled)).squeeze(-1)

    def checkpoint_state(self) -> dict[str, Any]:
        return {"kind": "full", "state_dict": self.state_dict()}

    def load_checkpoint_state(self, payload: dict[str, Any]) -> None:
        state_dict = payload.get("state_dict", payload)
        self.load_state_dict(state_dict)


class CausalLoRARewardModel(nn.Module):
    """A causal base model with LoRA adapters and a scalar reward head."""

    backend = "causal_lora"

    def __init__(
        self,
        model_name_or_path: str,
        *,
        dropout: float = 0.05,
        gradient_checkpointing: bool = True,
        load_dtype: str = "bf16",
        revision: str | None = None,
        lora_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The causal_lora backend requires PEFT; install requirements.txt"
            ) from exc

        dtype_by_name = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if load_dtype not in dtype_by_name:
            raise ValueError(f"Unsupported causal model load_dtype: {load_dtype}")
        base = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype_by_name[load_dtype],
            revision=revision,
            low_cpu_mem_usage=True,
        )
        self.resolved_revision = getattr(base.config, "_commit_hash", revision)
        base.config.use_cache = False
        if gradient_checkpointing:
            try:
                base.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                base.gradient_checkpointing_enable()
            if hasattr(base, "enable_input_require_grads"):
                base.enable_input_require_grads()

        lora = dict(lora_config or {})
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=int(lora.get("rank", 16)),
            lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            target_modules=lora.get("target_modules", "all-linear"),
            bias=str(lora.get("bias", "none")),
        )
        self.backbone = get_peft_model(base, peft_config)
        self.model_name_or_path = model_name_or_path
        self.dropout = nn.Dropout(float(dropout))
        self.reward_head = nn.Linear(int(base.config.hidden_size), 1)

    def forward(self, **tokens: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(**tokens)
        hidden = outputs.last_hidden_state
        attention_mask = tokens.get("attention_mask")
        if attention_mask is None:
            pooled = hidden[:, -1]
        else:
            last_indices = attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
            batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
            pooled = hidden[batch_indices, last_indices]
        return self.reward_head(self.dropout(pooled)).squeeze(-1)

    def checkpoint_state(self) -> dict[str, Any]:
        from peft import get_peft_model_state_dict

        return {
            "kind": "peft_reward",
            "adapter_state_dict": get_peft_model_state_dict(self.backbone),
            "reward_head_state_dict": self.reward_head.state_dict(),
        }

    def load_checkpoint_state(self, payload: dict[str, Any]) -> None:
        from peft import set_peft_model_state_dict

        if payload.get("kind") != "peft_reward":
            raise ValueError("Expected a PEFT reward-model checkpoint")
        set_peft_model_state_dict(
            self.backbone,
            payload["adapter_state_dict"],
        )
        self.reward_head.load_state_dict(payload["reward_head_state_dict"])


RewardModel = EncoderRewardModel | CausalLoRARewardModel


def build_reward_model(
    model_config: dict[str, Any],
    *,
    encoder_config: dict[str, Any] | None = None,
) -> RewardModel:
    backend = str(model_config.get("backend", "encoder"))
    model_name = str(model_config.get("name_or_path", "roberta-base"))
    if backend == "encoder":
        return EncoderRewardModel(
            model_name,
            dropout=model_config.get("dropout"),
            gradient_checkpointing=bool(
                model_config.get("gradient_checkpointing", True)
            ),
            revision=model_config.get("revision"),
            encoder_config=encoder_config,
        )
    if backend == "causal_lora":
        if encoder_config is not None:
            raise ValueError("Causal LoRA checkpoints do not use encoder_config")
        return CausalLoRARewardModel(
            model_name,
            dropout=float(model_config.get("dropout", 0.05)),
            gradient_checkpointing=bool(
                model_config.get("gradient_checkpointing", True)
            ),
            load_dtype=str(model_config.get("load_dtype", "bf16")),
            revision=model_config.get("revision"),
            lora_config=dict(model_config.get("lora", {})),
        )
    raise ValueError(f"Unsupported reward-model backend: {backend}")


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint_payload(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_reward_checkpoint(
    path: Path,
    *,
    model: RewardModel,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    epoch: int,
    validation_loss: float,
) -> None:
    payload: dict[str, Any] = {
        "format_version": 2,
        "model_state": model.checkpoint_state(),
        "model_config": model_config,
        "training_config": training_config,
        "epoch": epoch,
        "validation_loss": validation_loss,
    }
    if isinstance(model, EncoderRewardModel):
        payload["encoder_config"] = model.encoder.config.to_dict()
    _atomic_torch_save(payload, path)


def load_reward_checkpoint(
    path: Path,
    *,
    device: torch.device,
) -> tuple[RewardModel, dict]:
    payload = load_checkpoint_payload(path)
    model_config = dict(payload["model_config"])
    if int(payload.get("format_version", 1)) == 1:
        model = EncoderRewardModel(
            model_config["name_or_path"],
            dropout=model_config.get("dropout"),
            gradient_checkpointing=False,
            revision=model_config.get("revision"),
            encoder_config=payload.get("encoder_config"),
        )
        model.load_state_dict(payload["model_state_dict"])
    else:
        evaluation_config = dict(model_config)
        evaluation_config["gradient_checkpointing"] = False
        model = build_reward_model(
            evaluation_config,
            encoder_config=payload.get("encoder_config"),
        )
        model.load_checkpoint_state(payload["model_state"])
    model.to(device)
    model.eval()
    return model, payload
