# ADR 0005: stability gate and cross-engine evidence

- Status: Implemented; acceptance requires RTX 3080 Laptop evidence
- Date: 2026-07-19

## Context

M5 must distinguish a short smoke from sustained stability and must not compare engines with
different prompts, token counts, or sampling rules. External engines are evidence only when the
actual executable and model run; an unavailable adapter is not a benchmark row.

## Decision

The service stability gate runs for 1800 seconds with twenty client workers and rotating short,
medium, and long prompts. Workers also exercise timeout and client-disconnect cancellation. The
runner polls scheduler/KV metrics and `nvidia-smi` once per second, rejects unexpected request
failures or non-finite response metrics, and requires queued requests, active sequences, and used KV
blocks to drain to zero. GPU-memory analysis discards up to the first 60 warmup samples, then
compares medians of settled beginning and ending windows; growth must not exceed 64 MiB. This is a
leak signal, not a claim that driver memory is constant at every instant.
The final scheduler counters must also satisfy submitted = completed + cancelled + failed with zero
failed requests; an empty queue alone cannot hide a lost or failed request.

A saturated service may miss a five-second `/metrics` probe while all HTTP workers are occupied.
The runner records every miss and the maximum consecutive streak, directly checks child-process
liveness, and still requires at least ten valid scheduler samples plus a successful final drain. It
does not relabel monitoring contention as a server crash.

The formal CLI accepts only duration 1800 and concurrency 20. A shorter diagnostic run cannot emit
a passing M5 report. The server also maintains one lock order: request encoding releases the
tokenizer mutex before acquiring the engine queue mutex. Generation may hold the engine mutex while
decoding under the tokenizer mutex, so retaining the reverse tokenizer-to-engine order in submission
would permit a sustained-load deadlock.

The cross-engine workload fixes the seed prompt `The capital of France is`, deterministic cyclic
construction to context 128, greedy generation of exactly 32 tokens, three warmups, and ten samples.
The report includes PyTorch FP16, pinned llama.cpp GGUF F16 CUDA, TLIE FP16, and TLIE hybrid W8A16.
PyTorch establishes the fixed-workload reference sequence, and every adapter must prove exact token
equality before reporting performance. The aggregator rejects missing engines, mismatched
context/output metadata, or token drift and writes JSON, CSV, and Markdown together.

TLIE FP16 and INT8 may be selected from separate cold runs so running one precision does not warm the
other. The aggregator records every row's sampled software thermal-slowdown states and marks a row
clean only when none is `Active`. Thermally active rows remain valid correctness and memory evidence,
but their throughput is qualified rather than promoted as a clean baseline. An absent INT8 Batch 4
implementation is reported as unavailable; it is never filled with the FP16 result.

The aggregator additionally requires three warmups, ten samples, the pinned prompt/source model,
unchanged source snapshots, real-engine success markers, an in-workload GPU monitor, and the pinned
llama.cpp commit. Every engine and Batch 4 input must also have the same source-tree SHA-256; checking
only that each individual report stayed unchanged would permit stale reports from different source
revisions to be mixed. The common hash is retained in every output format. PyTorch records GPU state
after every warmup and measured Batch 1/4 workload rather than inferring thermal behavior only from
two endpoint samples.
The llama.cpp adapter derives the source checkout from the server executable, requires the checkout
to be clean at the pinned commit, and measures Batch 4 with three warmups and ten samples rather than
promoting one request to a ten-sample result.

TTFT, TPOT, total output throughput (including TTFT), peak device bytes, and Batch 4 throughput are
recorded when the engine exposes a comparable path. TLIE peak bytes are allocator-local. PyTorch
uses its CUDA allocator peak. llama.cpp uses process-wide `nvidia-smi memory.used`; the report
retains that source and does not claim allocator-level comparability.

Compute Sanitizer memcheck covers the batched model/slot reset path, while racecheck covers the
custom-kernel test matrix. Memcheck requires both an explicit zero-byte leak summary and zero-error
summary; racecheck requires an explicit zero-error, zero-warning hazard summary. Process exit zero
without those selected-tool summaries is rejected.

## Consequences

- The report is a fixed-workload engineering comparison, not a broad model-quality leaderboard.
- The 30-minute gate covers one GPU, one server process, context at most 512, and bounded Batch 4.
- Driver-level memory sampling can detect sustained growth but cannot attribute every allocation.
- TensorRT-LLM remains optional and is not substituted for a failed required engine.
