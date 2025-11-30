# Tinker LLM Chief Officer – Examples

This file is just a lightweight place to drop concrete run configs and notes.
You can extend it as you start using the Skill.

---

## Example 1 – Chat SFT + public benchmarks + custom eval

- Base model: `meta-llama/Llama-3.1-8B-Instruct`
- Use-case: English chat assistant for customer support.
- LoRA config:
  - rank: 32
  - train_mlp: true
  - train_attn: true
  - train_unembed: true
- Training:
  - `templates/training-config.yaml` with `train_file=data/support_train.jsonl`, `val_file=data/support_val.jsonl`
  - `python scripts/sl_trainer.py --config templates/training-config.yaml`
- Eval:
  - Layer 1: validation loss from `sl_trainer.py`.
  - Layer 2: run `scripts/run_inspect_evals.sh` with tasks:
    - `inspect_evals/ifeval`
    - `inspect_evals/mmlu_0_shot`
  - Layer 3: `python scripts/eval_offline.py --model-path tinker://YOUR/MODEL/PATH --eval-file data/support_eval.jsonl --model-name meta-llama/Llama-3.1-8B-Instruct --renderer-name llama3`

Keep adding more examples that reflect your real runs.
