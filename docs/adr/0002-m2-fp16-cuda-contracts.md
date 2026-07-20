# ADR 0002: M2 FP16 CUDA contracts and benchmark boundaries

- Status: Implemented; acceptance requires verification on the designated RTX 3080 Laptop
- Date: 2026-07-16

## Context

M2 adds the first GPU execution path. It must preserve the fixed TinyLlama architecture and M1
goldens, make KV ownership and bounds visible, and produce timing data that does not include model
loading or model-file IO. TinyLlama is trained for 2048 positions while the project plan also asks
for a 4096-token stress baseline.

## Decision

`scripts/convert_model_fp16.py` deterministically converts all 201 source tensors to contiguous
little-endian FP16 TLIEWGT records. The CUDA loader requires the externally pinned full-file and
source SHA-256 values, validates all record metadata and payload hashes, and uploads validated
tensors before model creation. The FP16 artifact remains ignored by Git.

The runtime uses one non-blocking CUDA stream, cuBLAS FP16 inputs/outputs with FP32 accumulation,
and these custom kernels:

- RMSNorm: one block per row with FP32 sum reduction.
- half-split Llama RoPE: one block per query or KV head.
- stable Softmax: FP32 max/sum reductions with FP16 storage.
- grouped-query decode Attention: one block per query head, FP32 scores and accumulation.
- KV update, SiLU-gated multiply, and residual add: bounds-checked launch wrappers.

All model weights, workspaces, and per-layer KV storage are allocated before the first token. The
per-token path performs no device allocation. Requests must provide contiguous positions, and a KV
write at or beyond the declared maximum returns a structured error. Recoverable CUDA and cuBLAS
errors are propagated as `cuda_error` or `out_of_memory` rather than aborting.

Kernel tests compare FP16 results against readable CPU references with fixed tolerances: RMSNorm
`2e-3`, RoPE and GEMM `3e-3`, Softmax `1e-3`, grouped-query Attention `6e-3`, and pointwise kernels
`2e-3`. The end-to-end comparison additionally checks all 32,000 prompt logits with
`atol=0.15, rtol=0.03` and requires the first 32 greedy token IDs to match M1 exactly. These
tolerances may only be changed after recording observed errors on the designated GPU.

CUDA events delimit compute and logits transfer for every token. The host monotonic clock records
wall latency, TTFT, TPOT, and sampling separately. NVTX ranges name `warmup`, `prefill`, `decode`,
and `sampling` so Nsight Systems can distinguish phases and synchronization. Every warmup executes
the same full context and output-token workload as a measured sample. The microbenchmark records
median and p95 CUDA-event time for every M2 kernel family. Nsight Compute uses its full metric set
on one launch per family to capture memory bandwidth, occupancy, and launch behavior without
profiling all 50 timed microbenchmark samples.

The formal M2 baseline is batch 1 with input contexts 128, 512, 2048, and 4096, 32 output tokens,
three warmups, and ten samples. The 4096 row explicitly enables ordinary RoPE extrapolation and is
only a memory-safety and performance stress case, not a generation-quality claim. The KV capacity
allows those 4096 input tokens plus at most 256 measured decode tokens. The performance exit check
is the context-128 median end-to-end decode rate between the first and last output token, including
GPU work, logits transfer, host conversion, synchronization, and greedy sampling. It must be at
least 90 tok/s on the designated 16GB RTX 3080 Laptop; the CUDA-compute-only rate is reported
separately, and results on other hardware are separate baselines.

Every context is constructed deterministically from the tokenizer output for
`The capital of France is`: retain the first seed token once, then cyclically repeat the remaining
seed token IDs until the requested context length. Sampling is greedy. The JSON report stores the
seed text, exact seed token IDs, construction rule, sampling mode, and timing-boundary descriptions;
the report generator rejects drift between contexts.

## Consequences

- Prefill currently reuses the transparent token-at-a-time forward path. It is a correct baseline,
  not a claim of optimized parallel prefill.
- The full FP16 model occupies about 2.2 GB on disk, while weights plus the maximum KV allocation
  and workspaces must fit the 16GB laptop GPU.
- `scripts/benchmark.py` refuses a different GPU name, zero samples, non-finite measurements, or a
  context matrix other than the four M2 contexts. It also verifies the returned workload metadata,
  kernel-family matrix, SM version, memory accounting, and exact equality between the child-reported
  source/FP16 hashes and the pins read from `config/model_manifest.json`; records the manifest path,
  CUDA toolkit, driver/runtime and in-engine peak allocation. While each kernel/context child process is active,
  it polls `nvidia-smi` once per second and records temperature, SM/memory clock, P-state,
  utilization, used-memory ranges, and the software thermal-slowdown state so thermal/clock
  behavior is not inferred from idle-only post-run snapshots. It writes JSON, CSV, and Markdown
  from one run.
- The Mac-to-WSL sync creates a deterministic SHA-256 manifest for every mirrored source file and
  records the Mac Git commit/dirty metadata when available. The WSL benchmark verifies the complete
  source snapshot before and after the workload without reading or requiring a remote `.git`
  directory. Its exit code and `acceptance.passed` require both that unchanged snapshot and the
  90 tok/s threshold. The report explicitly keeps thermal/clock acceptance pending manual review of
  the in-workload ranges; automated acceptance alone is not an M2 pass.
- CUDA compilation, kernel tests, sanitizer checks, profiler captures, and performance acceptance
  remain unverified until their real commands run on the RTX 3080 Laptop.
