#!/usr/bin/env python
"""
Offline evaluation script for Tinker models using a simple QA-style dataset.

- Loads a JSONL file with {"input": "...", "output": "..."} examples.
- Wraps a Tinker SamplingClient in a simple evaluator.
- Prints accuracy and basic stats.

This is meant as a template. You can customize the grading logic or metrics.
"""
import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import tinker
from tinker import types

from tinker_cookbook import renderers
from tinker_cookbook.evaluators import SamplingClientEvaluator
from tinker_cookbook.tokenizer_utils import get_tokenizer


def load_qa_jsonl(path: Path) -> List[Dict[str, str]]:
    data: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            inp = obj.get("input") or obj.get("prompt")
            out = obj.get("output") or obj.get("completion")
            if inp is None or out is None:
                raise ValueError(
                    "Each eval example must have ('input' or 'prompt') and "
                    "('output' or 'completion') fields."
                )
            data.append({"input": inp, "output": out})
    return data


class QAEvaluator(SamplingClientEvaluator):
    """
    Simple accuracy evaluator for QA-style tasks.

    Dataset should be a list of {"input": str, "output": str} items.
    Grader function returns True if the model response is counted as correct.
    """

    def __init__(
        self,
        dataset: List[Dict[str, str]],
        grader_fn: Callable[[str, str], bool],
        model_name: str,
        renderer_name: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        self.dataset = dataset
        self.grader_fn = grader_fn

        tokenizer = get_tokenizer(model_name)
        self.renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)

        self.sampling_params = types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=self.renderer.get_stop_sequences(),
        )

    async def __call__(self, sampling_client: tinker.SamplingClient) -> Dict[str, float]:
        num_examples = len(self.dataset)
        num_correct = 0

        for datum in self.dataset:
            model_input: types.ModelInput = self.renderer.build_generation_prompt(
                [renderers.Message(role="user", content=datum["input"])]
            )
            resp: types.SampleResponse = await sampling_client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=self.sampling_params,
            )
            tokens: List[int] = resp.sequences[0].tokens
            messages = self.renderer.parse_response(tokens)
            if not messages:
                continue
            answer = messages[0].content
            if self.grader_fn(answer, datum["output"]):
                num_correct += 1

        accuracy = num_correct / num_examples if num_examples > 0 else 0.0
        return {
            "num_examples": float(num_examples),
            "num_correct": float(num_correct),
            "accuracy": float(accuracy),
        }


def simple_contains_grader(response: str, target: str) -> bool:
    """
    Very simple grader: checks if the normalized target appears in the response.
    Replace with something more sophisticated (e.g., regex, numeric comparison,
    or LLM-as-a-judge) for real projects.
    """
    return target.strip().lower() in response.strip().lower()


async def run_eval(
    model_path: str,
    eval_file: Path,
    model_name: str,
    renderer_name: str,
) -> None:
    dataset = load_qa_jsonl(eval_file)
    print(f"[eval] Loaded {len(dataset)} examples from {eval_file}")

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=model_path)

    evaluator = QAEvaluator(
        dataset=dataset,
        grader_fn=simple_contains_grader,
        model_name=model_name,
        renderer_name=renderer_name,
    )

    metrics = await evaluator(sampling_client)
    print("[eval] Metrics:", metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Tinker model path for the sampler (e.g., tinker://run-id/weights/checkpoint-001)",
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        required=True,
        help="Path to JSONL eval file with {input, output} pairs.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Base model name (e.g., meta-llama/Llama-3.1-8B-Instruct).",
    )
    parser.add_argument(
        "--renderer-name",
        type=str,
        required=True,
        help="Renderer name for tinker_cookbook (e.g., llama3, qwen3).",
    )

    args = parser.parse_args()
    asyncio.run(
        run_eval(
            model_path=args.model_path,
            eval_file=Path(args.eval_file),
            model_name=args.model_name,
            renderer_name=args.renderer_name,
        )
    )


if __name__ == "__main__":
    main()
