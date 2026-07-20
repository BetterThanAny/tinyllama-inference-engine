from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, cast

REQUIRED_ENGINES = ("pytorch_fp16", "llama_cpp", "tlie_fp16", "tlie_int8")
PROMPT_SEED = "The capital of France is"
LLAMA_CPP_COMMIT = "571d0d540df04f25298d0e159e520d9fc62ed121"
SOURCE_MODEL_SHA256 = "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933"
FIELDS = (
    "engine",
    "source_tree_sha256",
    "status",
    "dtype",
    "context",
    "output_tokens",
    "warmup",
    "samples",
    "sampling",
    "seed",
    "ttft_ms_median",
    "tpot_ms_median",
    "output_tokens_per_second_median",
    "peak_device_bytes",
    "batch_4_total_tokens_per_second",
    "generated_tokens",
    "software_thermal_slowdown_states",
    "thermal_clean",
    "note",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the M5 cross-engine comparison")
    parser.add_argument("--tlie-report", type=Path, required=True)
    parser.add_argument("--tlie-int8-report", type=Path)
    parser.add_argument("--batch-report", type=Path, required=True)
    parser.add_argument("--pytorch-report", type=Path, required=True)
    parser.add_argument("--llama-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def external_row(path: Path, engine: str) -> dict[str, Any]:
    document = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if document.get("engine") != engine or document.get("status") != "available":
        raise ValueError(f"{engine} report is missing a successful real-engine result")
    source = cast(dict[str, Any], document.get("source"))
    before = cast(dict[str, Any], source.get("before")) if isinstance(source, dict) else {}
    after = cast(dict[str, Any], source.get("after")) if isinstance(source, dict) else {}
    tree_sha256 = before.get("tree_sha256")
    if before != after or not isinstance(tree_sha256, str) or len(tree_sha256) != 64:
        raise ValueError(f"{engine} report has no unchanged verified source snapshot")
    if document.get("prompt_seed") != PROMPT_SEED:
        raise ValueError(f"{engine} report used a non-comparable prompt seed")
    if document.get("model_source_sha256") != SOURCE_MODEL_SHA256:
        raise ValueError(f"{engine} report used an unpinned source model")
    monitor = document.get("gpu_monitor")
    if not isinstance(monitor, dict):
        raise ValueError(f"{engine} report has no in-workload GPU monitor")
    states = cast(list[str], monitor.get("software_thermal_slowdown_states"))
    if engine == "pytorch_fp16" and document.get("tokens_deterministic") is not True:
        raise ValueError("PyTorch report did not prove deterministic tokens")
    if engine == "llama_cpp":
        if document.get("tokens_match_reference") is not True:
            raise ValueError("llama.cpp report did not match the PyTorch token reference")
        if document.get("llama_cpp_commit") != LLAMA_CPP_COMMIT:
            raise ValueError("llama.cpp report used an unpinned source commit")
        gguf_sha256 = document.get("model_gguf_sha256")
        if not isinstance(gguf_sha256, str) or len(gguf_sha256) != 64:
            raise ValueError("llama.cpp report has no pinned GGUF checksum")
    row = {field: document.get(field) for field in FIELDS}
    row["source_tree_sha256"] = tree_sha256
    row["software_thermal_slowdown_states"] = states
    row["thermal_clean"] = "Active" not in states
    return row


def tlie_rows(
    tlie_path: Path, batch_path: Path, int8_path: Path | None = None
) -> list[dict[str, Any]]:
    report = cast(dict[str, Any], json.loads(tlie_path.read_text(encoding="utf-8")))
    int8_report = (
        report
        if int8_path is None
        else cast(dict[str, Any], json.loads(int8_path.read_text(encoding="utf-8")))
    )
    batch_report = cast(dict[str, Any], json.loads(batch_path.read_text(encoding="utf-8")))
    batch = cast(dict[str, Any], batch_report["benchmark"])
    source_hashes: set[str] = set()
    for name, document in (("TLIE FP16", report), ("TLIE INT8", int8_report)):
        source = cast(dict[str, Any], document.get("source"))
        before = cast(dict[str, Any], source.get("before")) if isinstance(source, dict) else {}
        after = cast(dict[str, Any], source.get("after")) if isinstance(source, dict) else {}
        tree_sha256 = before.get("tree_sha256")
        if before != after or not isinstance(tree_sha256, str) or len(tree_sha256) != 64:
            raise ValueError(f"{name} report has no unchanged verified source snapshot")
        source_hashes.add(tree_sha256)
        if cast(dict[str, Any], document.get("acceptance")).get("passed") is not True:
            raise ValueError(f"{name} report did not pass its engine acceptance gates")
    batch_source = cast(dict[str, Any], batch_report.get("source"))
    batch_before = (
        cast(dict[str, Any], batch_source.get("before", {}))
        if isinstance(batch_source, dict)
        else {}
    )
    batch_tree_sha256 = batch_before.get("tree_sha256")
    if (
        not isinstance(batch_source, dict)
        or batch_source.get("before") != batch_source.get("after")
        or not isinstance(batch_tree_sha256, str)
        or len(batch_tree_sha256) != 64
        or cast(dict[str, Any], batch_report.get("acceptance")).get("passed") is not True
        or batch.get("context") != 128
        or batch.get("output_tokens") != 32
        or batch.get("warmup") != 3
        or batch.get("samples") != 10
        or batch.get("tokens_match") is not True
    ):
        raise ValueError("TLIE Batch 4 report is not a comparable accepted workload")
    source_hashes.add(batch_tree_sha256)
    if len(source_hashes) != 1:
        raise ValueError("TLIE reports contain mixed source snapshots")
    rows: list[dict[str, Any]] = []
    names = {"float16": "tlie_fp16", "int8_weight_only": "tlie_int8"}
    selected_reports = {"float16": report, "int8_weight_only": int8_report}
    for mode, selected_report in selected_reports.items():
        workload = cast(dict[str, Any], selected_report["workload"])
        if (
            workload.get("contexts") != [128]
            or workload.get("comparison_only") is not True
            or workload.get("output_tokens") != 32
            or workload.get("warmup") != 3
            or workload.get("samples") != 10
            or workload.get("sampling") != "greedy"
            or workload.get("prompt_seed_text") != PROMPT_SEED
        ):
            raise ValueError(f"{names[mode]} report used a non-comparable workload")
        summary = next(
            row
            for row in cast(list[dict[str, Any]], selected_report["summary"])
            if int(row["context"]) == 128 and str(row["mode"]) == mode
        )
        generated_tokens = next(
            row["generated_tokens"]
            for row in cast(list[dict[str, Any]], selected_report["raw"])
            if int(row["context"]) == 128 and str(row["mode"]) == mode
        )
        monitor = next(
            row
            for row in cast(list[dict[str, Any]], selected_report["gpu_workload_monitors"])
            if str(row["phase"]).startswith(mode)
        )
        thermal_states = cast(list[str], monitor["software_thermal_slowdown_states"])
        if int(summary["context"]) != 128:
            continue
        ttft_ms = float(summary["ttft_ms_median"])
        tpot_ms = float(summary["tpot_ms_median"])
        rows.append(
            {
                "engine": names[mode],
                "source_tree_sha256": next(iter(source_hashes)),
                "status": "available",
                "dtype": mode,
                "context": 128,
                "output_tokens": workload["output_tokens"],
                "warmup": workload["warmup"],
                "samples": workload["samples"],
                "sampling": "greedy",
                "seed": 0,
                "ttft_ms_median": ttft_ms,
                "tpot_ms_median": tpot_ms,
                "output_tokens_per_second_median": 32 * 1000.0 / (ttft_ms + 31 * tpot_ms),
                "peak_device_bytes": summary["engine_peak_device_bytes"],
                "batch_4_total_tokens_per_second": (
                    batch["batch_4_total_tokens_per_second"] if mode == "float16" else None
                ),
                "generated_tokens": generated_tokens,
                "software_thermal_slowdown_states": thermal_states,
                "thermal_clean": "Active" not in thermal_states,
                "note": (
                    "TLIE paired benchmark; total tok/s includes TTFT; "
                    "INT8 has no batched server path"
                ),
            }
        )
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> None:
    engines = [str(row.get("engine")) for row in rows]
    if sorted(engines) != sorted(REQUIRED_ENGINES):
        raise ValueError(f"comparison engines are incomplete: {engines}")
    source_hashes = {row.get("source_tree_sha256") for row in rows}
    source_hash = next(iter(source_hashes)) if len(source_hashes) == 1 else None
    if not isinstance(source_hash, str) or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None:
        raise ValueError("cross-engine report contains mixed source snapshots")
    for row in rows:
        if row.get("status") != "available":
            raise ValueError(f"engine is not available: {row.get('engine')}")
        if (
            row.get("context") != 128
            or row.get("output_tokens") != 32
            or row.get("warmup") != 3
            or row.get("samples") != 10
        ):
            raise ValueError(f"engine used a non-comparable workload: {row.get('engine')}")
        if row.get("sampling") != "greedy" or row.get("seed") != 0:
            raise ValueError(f"engine used non-comparable sampling: {row.get('engine')}")
        states = row.get("software_thermal_slowdown_states")
        if (
            not isinstance(states, list)
            or not states
            or not set(states) <= {"Active", "Not Active"}
        ):
            raise ValueError(f"engine has invalid thermal metadata: {row.get('engine')}")
        for field in (
            "ttft_ms_median",
            "tpot_ms_median",
            "output_tokens_per_second_median",
            "peak_device_bytes",
        ):
            value = float(row.get(field, 0.0))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"engine {row.get('engine')} has invalid {field}")
    reference_tokens = rows[0].get("generated_tokens")
    if not isinstance(reference_tokens, list) or len(reference_tokens) != 32:
        raise ValueError("PyTorch reference did not provide exactly 32 generated tokens")
    if any(row.get("generated_tokens") != reference_tokens for row in rows[1:]):
        raise ValueError("cross-engine greedy tokens differ for the fixed workload")


def main() -> None:
    args = parse_args()
    rows = [
        external_row(args.pytorch_report, "pytorch_fp16"),
        external_row(args.llama_report, "llama_cpp"),
        *tlie_rows(args.tlie_report, args.batch_report, args.tlie_int8_report),
    ]
    validate_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "workload": {
            "prompt_seed": PROMPT_SEED,
            "context": 128,
            "output_tokens": 32,
            "sampling": "greedy",
            "seed": 0,
        },
        "engines": rows,
        "source_tree_sha256": rows[0]["source_tree_sha256"],
        "complete": True,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# M5 cross-engine comparison",
        "",
        "Fixed workload: context 128, 32 output tokens, greedy sampling.",
        f"Source tree SHA-256: `{rows[0]['source_tree_sha256']}`.",
        "",
        "| Engine | dtype | TTFT ms | TPOT ms | tok/s | peak MiB | Batch 4 tok/s | Thermal clean |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        batch_value = row["batch_4_total_tokens_per_second"]
        lines.append(
            f"| {row['engine']} | {row['dtype']} | {float(row['ttft_ms_median']):.3f} | "
            f"{float(row['tpot_ms_median']):.3f} | "
            f"{float(row['output_tokens_per_second_median']):.3f} | "
            f"{float(row['peak_device_bytes']) / 1024**2:.3f} | "
            f"{float(batch_value):.3f} | {'yes' if row['thermal_clean'] else 'no'} |"
            if batch_value is not None
            else f"| {row['engine']} | {row['dtype']} | {float(row['ttft_ms_median']):.3f} | "
            f"{float(row['tpot_ms_median']):.3f} | "
            f"{float(row['output_tokens_per_second_median']):.3f} | "
            f"{float(row['peak_device_bytes']) / 1024**2:.3f} | n/a | "
            f"{'yes' if row['thermal_clean'] else 'no'} |"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "engines": REQUIRED_ENGINES}))


if __name__ == "__main__":
    main()
