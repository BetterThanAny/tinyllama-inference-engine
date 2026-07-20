# TinyLlama inference engine: portfolio overview

## Project in one minute

This project implements a transparent TinyLlama-1.1B inference stack rather than wrapping an
existing engine. The readable CPU FP32 path is the numerical oracle; CUDA FP16 is the main execution
path; hybrid W8A16 trades speed for lower device memory. A bounded continuous-batching server adds a
small OpenAI-compatible HTTP/SSE surface with cancellation, timeout, request lifecycle metrics, and
reclaimable KV slots.

```text
pinned model + tokenizer
          |
          v
strict TLIEWGT loaders -----> CPU FP32 numerical oracle
          |
          +-----> CUDA FP16 model -----> Batch 4 scheduler -----> JSON / SSE API
          |
          +-----> hybrid W8A16 model
```

The design is intentionally limited to one TinyLlama revision, one NVIDIA GPU, bounded Batch 4, and
twenty request slots. It is an inference-systems portfolio project, not a replacement for vLLM or a
production deployment claim.

## What is implemented

- Strict model/config/tensor validation with external SHA-256 trust roots.
- CPU reference operators, tokenizer tests, explicit KV cache, greedy generation, and structured
  failure paths.
- CUDA RMSNorm, RoPE, Softmax, GEMV/GEMM, grouped-query attention, KV updates, SiLU, residual add,
  finite-value checks, and CUDA-event timing.
- Hybrid per-output-channel W8A16 matrices with FP16 vocabulary-facing matrices and activations.
- Shared FP16 weights, four execution rows, twenty KV slots, rotating decode-priority scheduling,
  timeout/cancellation, and resource reclamation.
- `/v1/models`, non-streaming and SSE `/v1/chat/completions`, stable error shapes, and scheduler/KV
  metrics.
- Reproducible benchmark, Compute Sanitizer, Nsight, stability, and cross-engine report contracts.

## Evidence that matters

All figures below come from the fixed RTX 3080 Laptop acceptance run documented in the
[tracked evidence summary](../benchmarks/evidence/20260720-rtx3080-m5-summary.md).

| Question | Evidence-backed answer |
|---|---|
| Are the custom CUDA paths exercised? | 249 kernel checks; CUDA CTest 8/8 |
| Are selected memory paths clean? | memcheck 0 leaked bytes/errors; racecheck 0 hazards |
| Does batching improve aggregate throughput? | Batch 4 / Batch 1 = `2.905x` |
| Does INT8 help? | Peak memory `-40.434%`; throughput `-20.111%` (speed non-finding) |
| Does the service drain under sustained load? | 1800 seconds, 3076 terminal requests, failed=0, final KV=0 |
| Are third-party comparisons exact-token? | PyTorch, llama.cpp, TLIE FP16/INT8 share the same 32 greedy tokens |

Thermal slowdown was observed in sustained and INT8 workloads. Those results remain correctness,
memory, and stability evidence, but are not promoted as clean performance baselines.

## How verification is split

GitHub Actions runs model-independent checks that can be honest on a hosted CPU runner:

- lockfile, Ruff, mypy, and 52 Python contract tests;
- CPU ASan/UBSan unit and operator tests, 93 explicit checks.

Full-model CPU tests require pinned local model assets. CUDA correctness, performance, sanitizer,
and profiler acceptance require the physical 16 GiB RTX 3080 Laptop. The workflow does not register
a GPU test and skip it; the hardware boundary is explicit.

## Resume-safe claims

Safe claims describe the concrete implementation, fixed workload, and measured qualification:

- C++20/CUDA TinyLlama 1.1B inference engine with CPU FP32, CUDA FP16, and hybrid W8A16 paths.
- Batch 4 aggregate throughput `2.905x` Batch 1 on the fixed context-128 workload.
- Hybrid W8A16 reduced peak device memory by `40.434%`; it did not improve throughput.
- Compute Sanitizer clean selected paths and a 30-minute concurrency-20 stability run with no
  failed or leaked requests.

Do not claim full OpenAI API compatibility, INT8 acceleration, multi-GPU/distributed serving,
production deployment, unthrottled long-run performance, or parity with vLLM.

## Known gaps

- Real CUDA OOM and `ResetSlot` fault injection remain unverified.
- The sanitizer targets cover kernels and the batched model, not the HTTP thread pool itself.
- INT8 Batch 4 is not implemented.
- The stability workload is one process with contexts bounded to 512 tokens.
- Raw model weights, full golden traces, Nsight reports, and large logs are not stored in Git.

For reproduction commands, start with the repository `README.md`. For the complete chronological
audit, including failed attempts and environment gates, use `PLAN.md`.
