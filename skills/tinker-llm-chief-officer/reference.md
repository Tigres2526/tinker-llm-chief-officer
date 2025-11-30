# Tinker LLM Chief Officer – Reference

This file encodes the training and evaluation doctrine this Skill should follow.

---

## 1. Tinker and Cookbook quick refresher

- `tinker` is the low-level training SDK (ServiceClient, TrainingClient, SamplingClient, RestClient).
- `tinker-cookbook` is the higher-level set of recipes and utilities (renderers, hyperparam_utils, evaluation, RL/SL loops, etc.).

Core workflow pattern:

1. Create a `ServiceClient`.
2. Create a `TrainingClient` with `create_lora_training_client(...)`.
3. Call `forward_backward` and `optim_step` in a loop.
4. Save checkpoints with `save_state` and export weights for inference with `save_weights_and_get_sampling_client`.
5. For later inference on any checkpoint, use `ServiceClient().create_sampling_client(model_path=...)`.

The Cookbook provides:
- Supervised training recipes (chat SFT, prompt distillation, etc.).
- RL recipes (math reasoning, tool use, RLHF pipelines).
- Preference learning recipes (DPO, reward models).
- Evaluation utilities including Inspect AI integration.

---

## 2. LoRA Without Regret – mental model

Key ideas from the Tinker LoRA Primer and related research:

1. **Capacity vs dataset size**

   - LoRA can match full fine-tuning when the LoRA parameter count is large enough relative to the dataset’s completion tokens.
   - Simple heuristic: for supervised learning, keep `LoRA_params >= completion_tokens` for a “low-regret” regime.

2. **Learning rate scaling**

   - LoRA typically wants a significantly higher LR than full FT (roughly 10× is a good starting point).
   - Tinker’s `hyperparam_utils` encodes model-size-aware LR recommendations via functions like `get_lr(model_name, is_lora=True)` and `get_lora_lr_over_full_finetune_lr(...)`.

3. **Where to place LoRA**

   - Apply LoRA to:
     - MLP layers (and MoE experts, if present).
     - Attention layers.
     - Optional unembedding layer.
   - Avoid attention-only LoRA if you can; it tends to underperform for a fixed parameter budget.

4. **Batch size sensitivity**

   - LoRA can be more brittle with very large batch sizes.
   - Prefer moderate batch sizes and increase them only after checking that metrics do not degrade.

5. **RL and distillation**

   - In RL and on-policy distillation settings, LoRA can work even at very low ranks (e.g., rank 1–8) because the objective is more targeted.
   - You can often trade lower rank for slightly more steps or more rollout data.

---

## 3. Hyperparameter utilities

Prefer these `tinker_cookbook.hyperparam_utils` helpers when available:

- `get_lr(model_name: str, is_lora: bool = True) -> float`
  - Returns a recommended LR considering model size and whether LoRA is used.

- `get_lora_param_count(model_name: str, lora_rank: int) -> int`
  - Returns LoRA parameter count. Compare this against the supervised completion token count.

- `get_lora_lr_over_full_finetune_lr(model_name: str, lora_alpha: int = 32) -> float`
  - Returns a multiplier to convert a full FT LR into a LoRA-appropriate LR.

Use these to derive defaults, then refine based on learning curves and evaluation results.

---

## 4. Evaluation framework – how to think about SOTA-ish evals

The Cookbook’s evaluation framework provides two main evaluator interfaces:

- `TrainingClientEvaluator` – works with `TrainingClient` (forward-only) for things like NLL on held-out data.
- `SamplingClientEvaluator` – works with `SamplingClient` for generative evaluations (task accuracy, rewards, etc.).

Key patterns to keep in mind:

- Inline evaluation is configured via `evaluator_builders` / `infrequent_evaluator_builders` for SL and RL training recipes.
- Offline evaluation is done using:
  - Inspect AI + Inspect Evals (for public benchmarks).
  - Custom `SamplingClientEvaluator` or Inspect tasks (for product-specific datasets).

When designing evals, aim for **three layers**:

1. Training-centric metrics – track whether training is even behaving.
2. Public benchmarks – compare against broader ecosystem baselines.
3. Product / domain metrics – measure the thing the user actually cares about.

---

## 5. Standard benchmarks via Inspect AI + Inspect Evals

Inspect AI is an evaluation framework, and Inspect Evals is a large collection of community-maintained benchmark tasks that plug into Inspect AI.

Common SOTA-style benchmarks you can reference when building eval suites:

- **General reasoning and knowledge**
  - MMLU (`inspect_evals/mmlu_*`): multitask exam-style questions across many domains.
  - GSM8K (`inspect_evals/gsm8k`): grade-school math word problems.
- **Instruction-following**
  - IFEval (`inspect_evals/ifeval`): tests strict adherence to automatically verifiable natural-language instructions.
- **Code generation**
  - HumanEval (`inspect_evals/humaneval`): Python function generation from docstrings.
  - MBPP (`inspect_evals/mbpp`): short Python coding challenges.
  - APPS (`inspect_evals/apps`): more complex programming tasks.
- **Agents / tools / assistants**
  - BFCL (`inspect_evals/bfcl`): function-calling ability.
  - GAIA (`inspect_evals/gaia` and sub-tasks): general assistant tasks involving tools and web.
  - AssistantBench (`inspect_evals/assistant_bench_*`): web and non-web agents.

As Chief Officer, you should:

- Propose a small set of tasks that actually test the capabilities the user wants (don’t blindly run everything).
- Document which versions of tasks you used and with what configuration (few-shot vs zero-shot, chain-of-thought, etc.).
- Keep Inspect Evals updated via `pip install -U inspect-evals` when appropriate.

---

## 6. Running Inspect benchmarks against Tinker models

Use Tinker’s provided script `tinker_cookbook.eval.run_inspect_evals` whenever possible.

Recommended approach (mirrored in `scripts/run_inspect_evals.sh`):

1. Save a sampler checkpoint from a training run and note its `model_path` (e.g., `tinker://...`).
2. Choose a renderer name that matches the base model family (e.g., `llama3`, `qwen3`).
3. Run something like:

   ```bash
   MODEL_PATH="tinker://YOUR/MODEL/PATH"
   MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
   RENDERER_NAME="llama3"

   python -m tinker_cookbook.eval.run_inspect_evals      model_path="$MODEL_PATH"      model_name="$MODEL_NAME"      renderer_name="$RENDERER_NAME"      tasks="inspect_evals/ifeval,inspect_evals/mmlu_0_shot,inspect_evals/gsm8k"
   ```

The Skill should generate or modify `scripts/run_inspect_evals.sh` to reflect project-specific tasks (e.g., code vs math vs tool-use) and explain to the user how to run it.

---

## 7. Custom application-specific evals

Two main patterns are useful here.

### 7.1 Inspect tasks using Tinker sampling

For complex tasks where correctness is not trivially machine-checkable, you can:

1. Wrap a Tinker `SamplingClient` into an Inspect API via `InspectAPIFromTinkerSampling`.
2. Define a custom Inspect `Task` with your dataset (could be internal tickets, product prompts, agent traces).
3. Use `model_graded_qa` or similar scorers to get LLM-as-a-judge labels.
4. Run `inspect eval` or `tinker_cookbook.eval.run_inspect_evals` targeting that task.

This gives you:
- Reusable Inspect logs.
- Rich scoring (partial credit, rubric-based grading).
- Compatibility with the rest of the Inspect ecosystem.

### 7.2 Custom SamplingClientEvaluator

For simpler yes/no style correctness, `SamplingClientEvaluator` is a good fit.

Basic pattern:

- Build a dataset like:

  ```python
  qa_dataset = [
      {"input": "What is the capital of France?", "output": "Paris"},
      ...
  ]
  ```

- Define a `grader_fn(response: str, target: str) -> bool`.
- Implement a `SamplingClientEvaluator` subclass that:
  - Builds prompts using `tinker_cookbook.renderers`.
  - Samples from the `SamplingClient`.
  - Uses `grader_fn` to compute per-example correctness and aggregates metrics (e.g., accuracy).

The included `scripts/eval_offline.py` shows how to:

- Load a JSONL dataset of `{input, output}` pairs.
- Evaluate any Tinker `model_path` on that dataset.
- Print a compact metrics dict (e.g., accuracy, number of examples).

---

## 8. Inline vs offline eval strategy

- **Inline evals** (during training):
  - Use them sparingly to get early signal.
  - Good for NLL on validation sets or small “smoke test” generative tasks.
  - Configure via `evaluator_builders` / `infrequent_evaluator_builders` in Cookbook configs.

- **Offline evals** (after training / between runs):
  - Use them more heavily to compare checkpoints and hyperparameter sweeps.
  - Run them on:
    - Standardized benchmarks (Inspect Evals).
    - Application datasets (custom Inspect tasks or `SamplingClientEvaluator`s).
  - Store results along with checkpoint metadata so you can trace which model is best for which metric.

---

## 9. Recommended evaluation templates by use-case

When the user describes a project, you can quickly propose templates like:

- **General chat assistant:**
  - Layer 1: NLL on validation chat set.
  - Layer 2: IFEval, MMLU, GSM8K.
  - Layer 3: Custom QA/helpdesk dataset + safety prompts scored by LLM-as-a-judge.

- **Code assistant:**
  - Layer 1: NLL on code-instruction pairs.
  - Layer 2: HumanEval, MBPP, APPS (if feasible).
  - Layer 3: Internal bug-fix / repo-specific eval tasks.

- **Math / reasoning model:**
  - Layer 1: NLL on math SFT data.
  - Layer 2: GSM8K, MATH, maybe other math benchmarks available in Inspect Evals.
  - Layer 3: Domain-specific math tasks or internal analytics workloads.

- **Tool-using / agentic assistant:**
  - Layer 1: NLL on tool-use traces.
  - Layer 2: BFCL, GAIA, AssistantBench variants.
  - Layer 3: End-to-end user journeys (e.g., retrieval + answer correctness).

These templates are starting points. The Skill should treat them as defaults and then adapt to the user’s constraints and data.

---

## 10. Governance and safety considerations

For serious deployments:

- Encourage the user to include:
  - Safety evals (e.g., jailbreak resilience, harmful content filters) if available in Inspect Evals or internal datasets.
  - Calibration metrics where appropriate (over/under-confidence).
- Make sure evals are re-run after significant training changes, not just once.
- Keep a simple “model registry” document (even in markdown) that tracks:
  - Model/version identifier.
  - Base model + LoRA config.
  - Training data summary.
  - Evaluation results on the agreed suite.
  - Deployment status (staging, prod, retired).

This keeps the fine-tuning/eval story auditable and reduces the chance of silently shipping a worse checkpoint.
