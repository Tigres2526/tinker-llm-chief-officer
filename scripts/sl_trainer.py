#!/usr/bin/env python
"""
Minimal supervised LoRA trainer on Tinker.

Expects a JSONL dataset of objects with either:
  - {"input": "...", "output": "..."}  OR
  - {"prompt": "...", "completion": "..."}

Config is provided via a YAML file (see templates/training-config.yaml).
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import tinker
from tinker import types
import yaml

try:
    # Optional, but strongly recommended; we use it if installed.
    from tinker_cookbook.hyperparam_utils import get_lr, get_lora_param_count
except ImportError:
    get_lr = None
    get_lora_param_count = None


def load_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_datum(example: Dict, tokenizer, max_seq_len: int) -> types.Datum:
    # Support both input/output and prompt/completion field names
    prompt = example.get("prompt") or example.get("input")
    completion = example.get("completion") or example.get("output")

    if prompt is None or completion is None:
        raise ValueError(
            "Each example must have ('prompt' or 'input') and "
            "('completion' or 'output') fields."
        )

    prompt = prompt.strip()
    completion = completion.strip()

    # Simple input → output format, similar to Tinker docs demo
    prompt_text = prompt
    completion_text = " " + completion + "\n\n"

    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion_text, add_special_tokens=False)

    tokens = prompt_tokens + completion_tokens
    if len(tokens) > max_seq_len:
        # Keep the last max_seq_len tokens (simple truncation policy)
        tokens = tokens[-max_seq_len:]

    # Weights: 0 for prompt tokens, 1 for completion tokens
    weights = [0] * len(prompt_tokens) + [1] * len(completion_tokens)
    if len(weights) > max_seq_len:
        weights = weights[-max_seq_len:]

    # Shift for next-token prediction
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(
            weights=weights,
            target_tokens=target_tokens,
        ),
    )


def batchify(data: List[types.Datum], batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


def evaluate(
    training_client: tinker.TrainingClient,
    val_data: List[types.Datum],
    batch_size: int,
) -> float:
    """Compute mean validation loss using forward() (no gradient)."""
    losses = []
    for batch in batchify(val_data, batch_size):
        fut = training_client.forward(batch, "cross_entropy")
        result = fut.result()
        losses.append(result.loss)
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    print(f"[eval] val_loss={mean_loss:.4f}")
    return mean_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    base_model = cfg["base_model"]
    train_file = Path(cfg["train_file"])
    val_file = Path(cfg.get("val_file", cfg["train_file"]))

    lora_rank = int(cfg.get("lora_rank", 32))
    max_seq_len = int(cfg.get("max_seq_len", 2048))
    train_batch_size = int(cfg.get("train_batch_size", 8))
    num_steps = int(cfg.get("num_steps", 1000))
    eval_every = int(cfg.get("eval_every", 100))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    num_epochs = int(cfg.get("num_epochs", 1))

    train_mlp = bool(cfg.get("train_mlp", True))
    train_attn = bool(cfg.get("train_attn", True))
    train_unembed = bool(cfg.get("train_unembed", True))

    # Learning rate: prefer hyperparam_utils if present
    lr = cfg.get("learning_rate")
    if lr is None and get_lr is not None:
        lr = float(get_lr(base_model, is_lora=True))
        print(f"[config] Using recommended LoRA LR from hyperparam_utils: {lr:.2e}")
    elif lr is None:
        lr = 2e-4
        print(
            "[config] WARNING: get_lr not available; defaulting to lr=2e-4. "
            "Consider setting learning_rate explicitly."
        )
    else:
        lr = float(lr)
        print(f"[config] Using user-specified learning_rate={lr:.2e}")

    if get_lora_param_count is not None:
        try:
            lora_params = get_lora_param_count(base_model, lora_rank)
            print(f"[info] Estimated LoRA params @ rank={lora_rank}: {lora_params:,}")
        except Exception as e:
            print(f"[warn] Could not compute LoRA param count: {e}")

    print(f"[config] base_model={base_model}")
    print(f"[config] train_file={train_file}")
    print(f"[config] val_file={val_file}")
    print(f"[config] lora_rank={lora_rank}, max_seq_len={max_seq_len}")
    print(f"[config] train_batch_size={train_batch_size}, num_steps={num_steps}")
    print(f"[config] train_mlp={train_mlp}, train_attn={train_attn}, train_unembed={train_unembed}")

    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=base_model,
        rank=lora_rank,
        train_mlp=train_mlp,
        train_attn=train_attn,
        train_unembed=train_unembed,
    )

    info = training_client.get_info()
    print(
        f"[info] Tinker model: base={info.model_data.model_name}, "
        f"lora_rank={info.model_data.lora_rank}"
    )

    tokenizer = training_client.get_tokenizer()

    train_examples = list(load_jsonl(train_file))
    val_examples = list(load_jsonl(val_file))

    print(f"[data] {len(train_examples)} train examples, {len(val_examples)} val examples")

    train_data = [build_datum(ex, tokenizer, max_seq_len) for ex in train_examples]
    val_data = [build_datum(ex, tokenizer, max_seq_len) for ex in val_examples]

    global_step = 0
    best_val_loss = float("inf")
    best_checkpoint_path = None
    run_name = cfg.get("run_name", base_model.replace("/", "-") + f"-lora-r{lora_rank}")

    for epoch in range(num_epochs):
        print(f"[train] Starting epoch {epoch + 1}/{num_epochs}")
        for batch in batchify(train_data, train_batch_size):
            global_step += 1

            fwdbwd_future = training_client.forward_backward(batch, "cross_entropy")
            optim_future = training_client.optim_step(
                types.AdamParams(learning_rate=lr, weight_decay=weight_decay)
            )

            fwdbwd_result = fwdbwd_future.result()
            _ = optim_future.result()

            loss = float(fwdbwd_result.loss)
            print(f"[train] step={global_step} loss={loss:.4f}")

            if global_step % eval_every == 0:
                val_loss = evaluate(training_client, val_data, train_batch_size)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_name = f"{run_name}-best-step{global_step}"
                    save_future = training_client.save_state(save_name)
                    save_result = save_future.result()
                    best_checkpoint_path = save_result.path
                    print(
                        f"[checkpoint] New best val_loss={val_loss:.4f} "
                        f"-> saved state at {best_checkpoint_path}"
                    )

            if global_step >= num_steps:
                print("[train] Reached max steps, stopping.")
                break
        if global_step >= num_steps:
            break

    print(f"[train] Best val_loss={best_val_loss:.4f}")
    if best_checkpoint_path is not None:
        print(f"[train] Best checkpoint path: {best_checkpoint_path}")

    # Also save sampler weights and report the model_path for downstream usage
    sampler_name = f"{run_name}-sampler"
    sampler_future = training_client.save_weights_and_get_sampling_client(sampler_name)
    sampling_client = sampler_future
    if hasattr(sampling_client, "result"):
        sampling_client = sampling_client.result()

    if hasattr(sampling_client, "model_path"):
        print(f"[train] Sampling client model_path: {sampling_client.model_path}")
    else:
        print("[train] WARNING: sampling_client has no model_path attribute; check tinker version.")


if __name__ == "__main__":
    main()
