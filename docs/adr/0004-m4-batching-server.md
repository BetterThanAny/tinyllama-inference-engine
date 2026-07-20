# ADR 0004: bounded continuous batching and loopback OpenAI server

- Status: Implemented; acceptance requires RTX 3080 Laptop evidence
- Date: 2026-07-19

## Context

M4 must serve concurrent requests without duplicating the 1.1B model weights, must reclaim KV state
after every terminal path, and must demonstrate a real Batch 4 throughput gain. A full vLLM-style
paged allocator or distributed server is outside scope; lifecycle correctness and measurable GPU
batching take priority.

## Decision

`TinyLlamaCuda` retains its batch-one API and adds a bounded batch API. One model owns one CUDA
stream, cuBLAS handle, shared weights, workspace for four rows, and twenty slot-major KV regions.
Each batch row supplies its token, contiguous position, and unique KV slot. FP16 linear layers use a
single cuBLAS GEMM across the active rows; attention, RoPE, and KV update dispatch per row because
sequence positions may differ. INT8 weights remain supported by per-row GEMV, while the M4 server
uses FP16 to exercise actual batched GEMM. Slot reset synchronizes and zeroes exactly one request's
key/value region before the allocator returns it.

The scheduler owns queued, running, completed, cancelled, and failed transitions. It admits up to
twenty active sequences, selects at most four per iteration with decode priority and a rotating
cursor, and may mix prefill and decode rows in one forward call. Client disconnect and deadline set
the same cancellation path; every completed, cancelled, timeout, and failed request resets and
releases its slot. No device allocation occurs in the scheduling loop.

The C++ HTTP server uses a pinned cpp-httplib revision, binds `127.0.0.1`, and exposes
`/v1/models`, `/v1/chat/completions`, and `/metrics`. Chat completions support JSON and SSE,
`max_tokens`, `temperature`, `top_p`, and the explicit `timeout_ms` extension. Recoverable request
errors use a stable OpenAI-shaped `error` object. Request responses expose queue time, TTFT, TPOT,
and output token rate; `/metrics` also exposes active sequences and KV capacity/utilization.

## Acceptance

The smoke runner starts the real CUDA server, checks models/error/non-stream/stream/sampling paths,
then launches twenty concurrent requests: twelve complete, four time out, and four close an SSE
connection after the first chunk. It waits for all futures and requires queued, active, and used KV
counts to return to zero. The separate paired benchmark fixes context 128, 32 output tokens, three
warmups, ten samples, and exact Batch 1/4 token equality; Batch 4 total throughput must be at least
1.5 times Batch 1. Compute Sanitizer memcheck covers the batched model and slot-reset workload.

## Consequences

- The design is deliberately bounded to one GPU, batch four, twenty active slots, and context 512.
- Prefill is token-wise rather than a fused multi-token kernel. Continuous batching improves total
  throughput but does not claim optimal prompt TTFT.
- HTTP is loopback-only and has no authentication or TLS. External deployment requires a trusted
  reverse proxy and is deferred.
- The smoke covers bounded concurrency and terminal cleanup, not the M5 thirty-minute stability or
  broader interoperability matrix.
