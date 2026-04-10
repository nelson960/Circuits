from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from circut.config import ModelSpec
from circut.model.decoder import DecoderOnlyTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def require_device(device_name: str) -> torch.device:
    if device_name == "mps":
        if not torch.backends.mps.is_built():
            raise RuntimeError("Requested device 'mps', but the installed PyTorch build does not include MPS support.")
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested device 'mps', but MPS is not available in the current environment.")
        return torch.device("mps")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device 'cuda', but CUDA is not available in the current environment.")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {device_name}")


def build_model(spec: ModelSpec, vocab_size: int, device: torch.device) -> DecoderOnlyTransformer:
    model = DecoderOnlyTransformer(spec, vocab_size)
    return model.to(device)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = {inner_key: inner_value.to(device) for inner_key, inner_value in value.items()}
        else:
            moved[key] = value
    return moved


def compute_lm_loss(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_targets = input_ids[:, 1:].contiguous()
    shifted_mask = attention_mask[:, 1:].contiguous()
    targets = shifted_targets.masked_fill(~shifted_mask, -100)
    loss = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        targets.view(-1),
        ignore_index=-100,
    )
    predictions = shifted_logits.argmax(dim=-1)
    correct = (predictions == shifted_targets) & shifted_mask
    token_accuracy = correct.sum() / shifted_mask.sum()
    if pad_token_id < 0:
        raise ValueError("pad_token_id must be non-negative.")
    return loss, token_accuracy


def save_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device)


def _migrate_legacy_feedforward_state_dict(model_state: dict[str, Any]) -> dict[str, Any]:
    legacy_prefix_map = {
        ".ff.net.0.": ".ff.fc_in.",
        ".ff.net.2.": ".ff.fc_out.",
    }
    found_legacy_keys = [key for key in model_state if any(prefix in key for prefix in legacy_prefix_map)]
    if not found_legacy_keys:
        return model_state

    found_current_keys = [
        key
        for key in model_state
        if ".ff.fc_in." in key or ".ff.fc_out." in key
    ]
    if found_current_keys:
        raise RuntimeError(
            "Checkpoint model_state mixes legacy feedforward parameter names with current names. "
            "Refusing to guess which schema is correct."
        )

    migrated_state: dict[str, Any] = {}
    for key, value in model_state.items():
        migrated_key = key
        for legacy_prefix, current_prefix in legacy_prefix_map.items():
            if legacy_prefix in key:
                migrated_key = key.replace(legacy_prefix, current_prefix)
                break
        if migrated_key in migrated_state:
            raise RuntimeError(
                f"Legacy feedforward checkpoint migration produced a duplicate parameter key: {migrated_key}"
            )
        migrated_state[migrated_key] = value
    return migrated_state


def load_model_state(model: torch.nn.Module, model_state: dict[str, Any]) -> None:
    migrated_state = _migrate_legacy_feedforward_state_dict(model_state)
    model.load_state_dict(migrated_state)
