from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch
from model_assets import load_manifest, sha256_file, verify_file
from numpy.typing import NDArray
from tlie_format import write_tensor_file
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "The capital of France is"
GOLDEN_TOKENS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export TinyLlama FP32 traces and greedy goldens")
    parser.add_argument("--manifest", type=Path, default=Path("config/model_manifest.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/tinyllama-chat-v1.0"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/generated/tinyllama-chat-v1.0")
    )
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def tensor_value(output: Any) -> torch.Tensor:
    value = output[0] if isinstance(output, tuple) else output
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"hook output is not a tensor: {type(value)!r}")
    return value


def hook_for(
    name: str, captured: dict[str, torch.Tensor]
) -> Callable[[Any, tuple[Any, ...], Any], None]:
    def capture(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        value = tensor_value(output)
        captured[name] = value.detach()[0, -1].float().cpu().contiguous()

    return capture


def numpy_tensors(captured: dict[str, torch.Tensor]) -> Iterator[tuple[str, NDArray[Any]]]:
    for name in sorted(captured):
        yield name, captured[name].numpy()


def main() -> None:
    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    manifest = load_manifest(args.manifest)
    for filename, record in manifest["files"].items():
        verify_file(args.model_dir / filename, record)

    tokenizer: Any = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        args.model_dir, local_files_only=True, use_fast=False, legacy=False
    )
    model: Any = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()

    captured: dict[str, torch.Tensor] = {}
    handles = []
    required_traces = {"embedding", "final_norm", "logits"}
    for index, layer in enumerate(model.model.layers):
        prefix = f"layer{index}."
        layer_names = {
            "input_norm",
            "attention_output",
            "post_attention_norm",
            "mlp_output",
            "output",
        }
        required_traces.update(prefix + name for name in layer_names)
        handles.extend(
            [
                layer.input_layernorm.register_forward_hook(
                    hook_for(prefix + "input_norm", captured)
                ),
                layer.self_attn.register_forward_hook(
                    hook_for(prefix + "attention_output", captured)
                ),
                layer.post_attention_layernorm.register_forward_hook(
                    hook_for(prefix + "post_attention_norm", captured)
                ),
                layer.mlp.register_forward_hook(hook_for(prefix + "mlp_output", captured)),
                layer.register_forward_hook(hook_for(prefix + "output", captured)),
            ]
        )
    handles.extend(
        [
            model.model.embed_tokens.register_forward_hook(hook_for("embedding", captured)),
            model.model.norm.register_forward_hook(hook_for("final_norm", captured)),
            model.lm_head.register_forward_hook(hook_for("logits", captured)),
        ]
    )

    encoded: Any = tokenizer(PROMPT, return_tensors="pt", add_special_tokens=True)
    input_ids: torch.Tensor = encoded["input_ids"]
    with torch.inference_mode():
        outputs: Any = model(input_ids=input_ids, use_cache=True)
    for handle in handles:
        handle.remove()

    if captured.keys() != required_traces:
        raise RuntimeError(f"missing trace tensors: {sorted(required_traces - captured.keys())}")

    generated: list[int] = []
    logits: torch.Tensor = outputs.logits[:, -1, :]
    past_key_values: Any = outputs.past_key_values
    with torch.inference_mode():
        for step in range(GOLDEN_TOKENS):
            next_token = torch.argmax(logits, dim=-1)
            generated.append(int(next_token.item()))
            if step + 1 < GOLDEN_TOKENS:
                outputs = model(
                    input_ids=next_token[:, None], past_key_values=past_key_values, use_cache=True
                )
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "reference_trace.tliewgt"
    source_sha = manifest["files"]["model.safetensors"]["sha256"]
    write_tensor_file(trace_path, source_sha, numpy_tensors(captured), len(captured))
    metadata = {
        "schema_version": 1,
        "model_id": manifest["model_id"],
        "revision": manifest["revision"],
        "source_model_sha256": source_sha,
        "config_sha256": manifest["files"]["config.json"]["sha256"],
        "tokenizer_sha256": manifest["files"]["tokenizer.model"]["sha256"],
        "reference": f"Transformers 4.53.2 / PyTorch {torch.__version__} eager FP32",
        "prompt": PROMPT,
        "prompt_ids": [int(value) for value in input_ids[0].tolist()],
        "greedy_tokens": generated,
        "trace_file": trace_path.name,
        "trace_sha256": sha256_file(trace_path),
        "trace_tensors": sorted(captured),
        "atol": 1.0e-4,
        "rtol": 1.0e-4,
    }
    metadata_path = args.output_dir / "reference.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {len(captured)} FP32 traces and {len(generated)} greedy tokens to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
