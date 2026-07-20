from __future__ import annotations

import argparse
import http.client
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

try:
    import wsl_cuda_env as cuda_env
    from source_snapshot import verify_snapshot
except ModuleNotFoundError:
    from scripts import wsl_cuda_env as cuda_env
    from scripts.source_snapshot import verify_snapshot


def request_json(
    method: str, port: int, path: str, body: dict[str, Any] | None = None, timeout: float = 90.0
) -> tuple[int, dict[str, Any]]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    payload = None if body is None else json.dumps(body)
    headers = {} if payload is None else {"Content-Type": "application/json"}
    connection.request(method, path, body=payload, headers=headers)
    response = connection.getresponse()
    document = json.loads(response.read())
    status = response.status
    connection.close()
    return status, cast(dict[str, Any], document)


def completion_body(
    prompt: str,
    *,
    stream: bool = False,
    max_tokens: int = 2,
    timeout_ms: int = 60_000,
) -> dict[str, Any]:
    return {
        "model": "tinyllama-1.1b",
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "timeout_ms": timeout_ms,
    }


def streaming_completion(port: int, prompt: str, *, cancel_after_first: bool) -> str:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=90.0)
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=json.dumps(
            completion_body(prompt, stream=True, max_tokens=32 if cancel_after_first else 3)
        ),
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    if response.status != 200:
        raise RuntimeError(f"stream returned HTTP {response.status}: {response.read()!r}")
    chunks: list[str] = []
    while True:
        line = response.readline().decode("utf-8")
        if not line:
            break
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ").strip()
        chunks.append(data)
        if cancel_after_first and data != "[DONE]":
            connection.close()
            return "cancelled"
        if data == "[DONE]":
            connection.close()
            return "completed" if len(chunks) >= 2 else "invalid"
        document = json.loads(data)
        if document.get("object") != "chat.completion.chunk":
            raise RuntimeError("stream chunk has the wrong OpenAI object")
    connection.close()
    return "invalid"


def concurrent_request(port: int, index: int) -> str:
    if index < 12:
        status, response = request_json(
            "POST",
            port,
            "/v1/chat/completions",
            completion_body(f"Say the number {index}.", max_tokens=2),
        )
        if status != 200 or response.get("object") != "chat.completion":
            raise RuntimeError(f"normal request {index} failed: HTTP {status} {response}")
        return "completed"
    if index < 16:
        status, response = request_json(
            "POST",
            port,
            "/v1/chat/completions",
            completion_body(f"Timeout request {index}.", max_tokens=16, timeout_ms=1),
        )
        if status != 408 or response.get("error", {}).get("code") != "request_cancelled":
            raise RuntimeError(f"timeout request {index} was not cancelled: {status} {response}")
        return "timeout"
    return streaming_completion(port, f"Cancel stream {index}.", cancel_after_first=True)


def wait_ready(process: subprocess.Popen[str], port: int) -> None:
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, _ = process.communicate()
            raise RuntimeError(f"server exited before ready: {stdout}")
        try:
            status, response = request_json("GET", port, "/v1/models", timeout=1.0)
            if status == 200 and response.get("object") == "list":
                return
        except (ConnectionError, OSError, TimeoutError):
            time.sleep(0.1)
    raise TimeoutError("server did not become ready")


def wait_drained(port: int) -> dict[str, Any]:
    deadline = time.monotonic() + 60.0
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status, latest = request_json("GET", port, "/metrics")
        if status == 200 and all(
            int(latest[field]) == 0
            for field in ("queued_requests", "active_sequences", "kv_blocks_used")
        ):
            return latest
        time.sleep(0.1)
    raise TimeoutError(f"scheduler did not drain: {latest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exercise the M4 OpenAI-compatible server")
    parser.add_argument("--server", type=Path, default=Path("build/cuda-release/tinyllama_server"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/tinyllama-chat-v1.0"))
    parser.add_argument("--source-snapshot", type=Path, default=Path(".tlie-source-snapshot.json"))
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_before = verify_snapshot(args.source_snapshot, Path.cwd())
    environment_directory = Path("/tmp/tlie-m4-server-env")
    environment_directory.mkdir(parents=True, exist_ok=True)
    environment = cuda_env.prepare_cuda_tool_environment(environment_directory)
    with subprocess.Popen(
        environment.wrap_target(
            [
                str(args.server),
                "--port",
                str(args.port),
                "--model-dir",
                str(args.model_dir),
            ]
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment.process_env,
    ) as server:
        try:
            wait_ready(server, args.port)
            status, models = request_json("GET", args.port, "/v1/models")
            if status != 200 or models["data"][0]["id"] != "tinyllama-1.1b":
                raise RuntimeError("/v1/models response is invalid")
            invalid_status, invalid = request_json(
                "POST",
                args.port,
                "/v1/chat/completions",
                {"messages": [], "max_tokens": 0},
            )
            if invalid_status != 400 or "error" not in invalid:
                raise RuntimeError("invalid request did not return a stable error shape")
            status, nonstream = request_json(
                "POST",
                args.port,
                "/v1/chat/completions",
                completion_body("The capital of France is", max_tokens=3),
            )
            if (
                status != 200
                or nonstream.get("choices", [{}])[0].get("message", {}).get("content") is None
            ):
                raise RuntimeError("non-streaming smoke failed")
            metrics = nonstream.get("metrics", {})
            if int(nonstream.get("usage", {}).get("completion_tokens", 0)) > 1:
                tpot_ms = float(metrics.get("tpot_ms", 0.0))
                output_tps = float(metrics.get("output_tokens_per_second", 0.0))
                if tpot_ms <= 0.0 or abs(output_tps - 1000.0 / tpot_ms) > 0.02 * output_tps:
                    raise RuntimeError("non-streaming TPOT and output tok/s metrics disagree")
            late_timeout_status, late_timeout = request_json(
                "POST",
                args.port,
                "/v1/chat/completions",
                completion_body("Exercise an in-flight timeout.", max_tokens=1, timeout_ms=1),
            )
            if (
                late_timeout_status != 408
                or late_timeout.get("error", {}).get("code") != "request_cancelled"
            ):
                raise RuntimeError(
                    f"in-flight final-token timeout was not cancelled: "
                    f"{late_timeout_status} {late_timeout}"
                )
            sampled_status, sampled = request_json(
                "POST",
                args.port,
                "/v1/chat/completions",
                {
                    **completion_body("Name one color.", max_tokens=2),
                    "temperature": 0.7,
                    "top_p": 0.9,
                },
            )
            if sampled_status != 200 or sampled.get("object") != "chat.completion":
                raise RuntimeError("temperature/top-p sampling smoke failed")
            if (
                streaming_completion(
                    args.port, "The capital of France is", cancel_after_first=False
                )
                != "completed"
            ):
                raise RuntimeError("streaming smoke failed")
            outcomes: list[str] = []
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = [
                    executor.submit(concurrent_request, args.port, index) for index in range(20)
                ]
                for future in as_completed(futures, timeout=120.0):
                    outcomes.append(future.result())
            metrics = wait_drained(args.port)
            source_after = verify_snapshot(args.source_snapshot, Path.cwd())
            counts = {outcome: outcomes.count(outcome) for outcome in set(outcomes)}
            if (
                len(outcomes) != 20
                or counts.get("completed") != 12
                or counts.get("timeout") != 4
                or counts.get("cancelled") != 4
            ):
                raise RuntimeError(f"concurrency outcomes are incomplete: {counts}")
            if int(metrics["kv_blocks_used"]) != 0 or int(metrics["cancelled_total"]) < 8:
                raise RuntimeError(f"KV blocks were not reclaimed after cancellation: {metrics}")
            report = {
                "schema_version": 1,
                "models_smoke": "passed",
                "nonstream_smoke": "passed",
                "stream_smoke": "passed",
                "temperature_top_p_smoke": "passed",
                "invalid_error_shape": "passed",
                "in_flight_timeout": "passed",
                "concurrent_requests": 20,
                "outcomes": counts,
                "final_metrics": metrics,
                "source": {"before": source_before, "after": source_after},
                "passed": True,
            }
            serialized = json.dumps(report, indent=2) + "\n"
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(serialized, encoding="utf-8")
            print(serialized, end="")
        finally:
            server.terminate()
            try:
                server.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10.0)


if __name__ == "__main__":
    main()
