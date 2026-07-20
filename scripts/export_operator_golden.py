from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as functional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export deterministic FP32 operator goldens")
    parser.add_argument("--output", type=Path, default=Path("data/golden/operators.json"))
    return parser.parse_args()


def as_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.reshape(-1).tolist()]


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.float32)

    rms_input = torch.tensor([1.0, -2.0, 3.0, -4.0])
    rms_weight = torch.tensor([0.5, 1.0, 1.5, 2.0])
    rms_output = rms_input * torch.rsqrt(rms_input.square().mean() + 1.0e-5) * rms_weight

    query = torch.tensor([[0.1, 0.2, 0.3, 0.4], [-0.5, 0.6, -0.7, 0.8]])
    key = torch.tensor([[0.9, -1.0, 1.1, -1.2]])
    position = 7
    theta = 10000.0
    inverse_frequency = 1.0 / (theta ** (torch.arange(0, 4, 2, dtype=torch.float32) / 4.0))
    angles = position * inverse_frequency
    cosine = torch.cos(angles)
    sine = torch.sin(angles)

    def rotate(tensor: torch.Tensor) -> torch.Tensor:
        first, second = tensor[..., :2], tensor[..., 2:]
        return torch.cat([first * cosine - second * sine, second * cosine + first * sine], dim=-1)

    rope_query = rotate(query)
    rope_key = rotate(key)

    softmax_input = torch.tensor([1000.0, 1001.0, 999.0, -1000.0])
    softmax_output = torch.softmax(softmax_input, dim=-1)

    attention_query = torch.tensor([[0.2, -0.1, 0.4, 0.3], [-0.5, 0.7, 0.1, -0.2]])
    attention_keys = torch.tensor(
        [[[0.1, 0.2, 0.3, 0.4]], [[-0.4, 0.3, -0.2, 0.1]], [[0.5, -0.6, 0.7, -0.8]]]
    )
    attention_values = torch.tensor(
        [[[1.0, 0.0, -1.0, 0.5]], [[0.2, 0.4, 0.6, 0.8]], [[-0.3, 0.7, 0.9, -0.1]]]
    )
    repeated_keys = attention_keys[:, 0, :]
    repeated_values = attention_values[:, 0, :]
    scores = attention_query @ repeated_keys.T / math.sqrt(4.0)
    attention_output = torch.softmax(scores, dim=-1) @ repeated_values

    gate = torch.tensor([-2.0, -0.5, 0.0, 1.5])
    up = torch.tensor([0.5, -1.0, 2.0, 3.0])
    silu_output = functional.silu(gate) * up

    document = {
        "reference": f"PyTorch {torch.__version__} FP32",
        "atol": 1.0e-6,
        "rms_norm": {
            "input": as_list(rms_input),
            "weight": as_list(rms_weight),
            "epsilon": 1.0e-5,
            "output": as_list(rms_output),
        },
        "rope": {
            "query": as_list(query),
            "key": as_list(key),
            "query_heads": 2,
            "key_value_heads": 1,
            "head_dim": 4,
            "position": position,
            "theta": theta,
            "output_query": as_list(rope_query),
            "output_key": as_list(rope_key),
        },
        "softmax": {"input": as_list(softmax_input), "output": as_list(softmax_output)},
        "attention": {
            "query": as_list(attention_query),
            "keys": as_list(attention_keys),
            "values": as_list(attention_values),
            "sequence_length": 3,
            "query_heads": 2,
            "key_value_heads": 1,
            "head_dim": 4,
            "output": as_list(attention_output),
        },
        "silu_multiply": {
            "gate": as_list(gate),
            "up": as_list(up),
            "output": as_list(silu_output),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote operator golden data to {args.output}")


if __name__ == "__main__":
    main()
