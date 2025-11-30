# Tinker LLM Chief Officer

A Claude Code skill for designing and running state-of-the-art LoRA/QLoRA fine-tuning, RL, RLHF, and evaluation pipelines for open-weight LLMs using the [Tinker API](https://github.com/thinking-machines-lab/tinker).

## Installation

### Via Claude Code Plugin System

```
/plugin install Tigres2526/tinker-llm-chief-officer
```

### Manual Installation

Clone this repository and copy the skill to your Claude Code skills directory:

```bash
git clone https://github.com/Tigres2526/tinker-llm-chief-officer.git
cp -r tinker-llm-chief-officer/skills/tinker-llm-chief-officer ~/.claude/skills/
```

## Prerequisites

- **Python 3.10+** (3.11 recommended for Inspect Evals)
- **Tinker API Key** - Set as environment variable: `export TINKER_API_KEY=your_key`
- **Dependencies**:
  ```bash
  pip install tinker "tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git"
  pip install inspect-ai inspect-evals
  pip install pyyaml numpy
  ```

## What This Skill Does

When activated, Claude acts as a senior LLM post-training lead, helping you:

- **Choose base models** - Llama, Qwen, and other open-weight families with size recommendations
- **Configure LoRA/QLoRA** - Optimal rank, learning rates, and training layers
- **Design training pipelines** - Supervised fine-tuning, DPO/RLHF, RL with rewards, distillation
- **Set up evaluations** - Three-layer eval strategy with training metrics, public benchmarks (Inspect AI), and custom application evals
- **Manage checkpoints** - Save, compare, and select the best models

## Included Files

| File | Description |
|------|-------------|
| `skills/tinker-llm-chief-officer/SKILL.md` | Main skill definition and operating procedures |
| `skills/tinker-llm-chief-officer/reference.md` | Hyperparameter policy and evaluation framework |
| `skills/tinker-llm-chief-officer/examples.md` | Example configurations and run templates |
| `scripts/sl_trainer.py` | Supervised learning training script |
| `scripts/eval_offline.py` | Offline evaluation with custom metrics |
| `scripts/run_inspect_evals.sh` | Inspect AI benchmark runner |
| `templates/training-config.yaml` | Training configuration template |

## Quick Start

1. Install the plugin
2. Ask Claude: *"Help me fine-tune Llama 3.1 8B for customer support"*
3. Claude will guide you through:
   - Clarifying your use-case and constraints
   - Choosing optimal LoRA configuration
   - Creating training configs and scripts
   - Running training and evaluation
   - Selecting the best checkpoint

## License

MIT License - see [LICENSE](LICENSE) for details.
