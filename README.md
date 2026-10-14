# EasiScriptX (ESX)

EasiScriptX (ESX) is a high-performance domain-specific language (DSL) for building, optimizing, and running AI/ML workflows. It combines a compact, declarative syntax with a C++ runtime that implements memory, inference, and training optimizations so researchers can express ideas quickly and measure production-grade performance.

Key takeaways (for a researcher who wants to evaluate quickly)

- Express models, datasets, and training flows in a few lines of readable DSL.
- Reproducible, C++-level performance and memory optimizations (memory broker, quantization, fused kernels).
- Measured improvements in representative workloads: memory reductions of 30–40% (Memory Broker), up to 4× model-size reduction with quantization, and ~2× speculative-decoding inference speedups for autoregressive LLMs.
- Interoperable with PyTorch/ONNX runtime for pretrained models and interop testing.

Table of contents

- Features & what they mean
- Quickstart (build & run a small example)
- Language / syntax highlights
- Research foundations (papers mapped to implementation)
- Performance & comparisons (measured, repository benchmarks)
- Installation & dependencies
- Testing & CI
- Contributing, roadmap, and license

---

Features (concrete, non-generic)

These are the implemented, testable capabilities you can rely on in this repository. I removed generic marketing language and flagged prototype features.

- Declarative DSL for ML workflows
  - Write model definitions, datasets, tokenization, and training loops with concise syntax (examples in docs/syntax.md and examples/).
  - Why this matters: reduces boilerplate compared to handwritten Python training scripts and keeps architecture + training configuration in one place.

- Integrated memory management (Memory Broker)
  - zeRO-style offload and dynamic memory strategies implemented in the runtime.
  - Measured: 30–40% reduction in peak GPU memory on benchmark workloads in tests/benchmarks/ when enabled.
  - Benefit vs standard stack: avoids hand-rolling ZeRO orchestration across separate libraries — useful for larger models or larger effective batch sizes on the same hardware.

- End-to-end quantization support
  - 8-bit and 4-bit quantize primitives (ptq/qat hooks) wired into the AST and runtime.
  - Measured: model size reduction up to ~4× and typical inference speedups of 2–3× depending on kernel availability.
  - Benefit vs standard stack: integrates quantization into the workflow and tests rather than requiring separate conversion tooling.

- Gradient checkpointing
  - Built-in checkpoint(...) statement with configurable segments.
  - Measured: ~50% memory reduction at a known recomputation cost.

- LLM-focused optimizations
  - Speculative decoding: runtime support for draft+main model decoding flow (speculative_decode statement).
  - Kernel fusion: fused matmul+activation paths for lower CPU overhead.
  - Sparse attention + FlashAttention hooks: memory-saving attention schedules and profiling support.
  - Measured: speculative decoding gave ~2× throughput improvement in repository benchmarks for applicable workloads; kernel fusion gives 20–30% CPU-side gains in microbenchmarks.

- Distributed training primitives
  - First-class distribute(gpus: N) statement; runtime integrations for NCCL (multi-GPU) and MPI (multi-node) with a CPU fallback.
  - Benefit: simpler switch from single-process to distributed runs compared to ad-hoc script-based orchestration.

- Dataset streaming and prefetching
  - Streaming datasets with a producer/consumer prefetch queue in dataset.hpp for lower host-side stalls in IO-heavy setups.

- Profiling and energy monitoring
  - Built-in timing (std::chrono) and energy-tracking hooks controlled by config flags for reproducible profiling during CI/benchmarks.

- Extensibility and research hooks
  - AST and interpreter structured so you can add new statements (e.g., custom ops, optimization passes) and test them with the existing harness.

- Pattern recognition (experimental prototype)
  - The repository contains a simple, rule-based pattern_recognize API (geometric, arithmetic) implemented as a prototype/example in src/dataset.hpp and examples/pattern_recognition.esx.
  - Status: prototype; example-only implementation with mock transformations. Treat it as experimental rather than a validated research component.

What this means practically — why ESX can be preferable to a "standard" stack

I'll be direct and factual here — no buzzwords, no blanket claims:

- Fewer moving pieces when you need both research flexibility and memory/throughput optimizations. Instead of gluing PyTorch + multiple conversion tools + custom CUDA kernels + orchestration scripts, ESX places DSL + runtime in one codebase so common optimizations are reusable across experiments.

- Lower friction to reproduce infrastructure-level optimizations. Memory broker, quantization, checkpointing, and speculative decoding are accessible via the same DSL and test harness used for correctness and benchmarks; you don't need to assemble separate toolchains to run the same experiments.

- Benchmarks live with the code. The repository contains microbenchmarks, tests, and CI configuration that exercise each optimization. The performance numbers in this README are taken from those benchmarks and from tests/benchmarks.

- Trade-offs are explicit. Integrating these features into one C++ runtime means you accept a C++ build/test cycle (longer iteration than Python REPLs) but gain lower-level control and measurable production-oriented performance without rewriting kernels from scratch.

---

Quickstart (minimal)

1) Build (linux/mac/windows via CMake)

```bash
# create build directory and build
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# run tests
ctest --output-on-failure
```

If you prefer containers, see docs/DOCKER.md. Full build notes are in docs/BUILD.md.

A smallest example (train loop in ESX DSL)

```esx
model mnist_net {
  layer dense(units=128, activation=relu);
  layer dropout(rate=0.5);
  layer dense(units=10, activation=softmax);
}

train(mnist_net, mnist, loss: ce, opt: adam(lr=0.001), epochs: 10, device: gpu);
```

Research foundations (concise mapping to code)

- Attention / Transformers — Vaswani et al., “Attention is All You Need” (2017)
  - Implementation: AttentionExpr in src/ast.hpp mapped to attention operations in the interpreter.

- Distributed training strategies — Li et al., “PyTorch Distributed: Experiences on Accelerating Data Parallel Training” (2020)
  - Implementation: distributed.hpp (init_multi_gpu: NCCL, init_multi_node: MPI) and DistributeStmt in the AST.

- Model compression — Han et al., “Deep Compression” (2015)
  - Implementation: QuantizeStmt / QuantizeExpr and PruneExpr with runtime hooks for ONNX/PyTorch paths.

- Autonomic computing — Kephart & Chess, “The Vision of Autonomic Computing” (2003)
  - Implementation: AutonomicStmt and AgentTuneStmt; currently a basic simulation harness (see src/ast.hpp and related tests).

- ONNX interoperability — ONNX project (Bai et al., 2019)
  - Implementation: ImportExpr and USE_ONNX flags in src/config.hpp; runtime hooks for ONNXRuntime where available.

Notes on pattern recognition / "ARC-AGI2"

- The codebase references "ARC-AGI2-inspired" pattern recognition in a few places (examples and docs). The implementation is an educational/prototype feature: a rule-based transformation and example harness in src/dataset.hpp and examples/pattern_recognition.esx.
- Recommendation: mark this as experimental in the docs (done here). If you want a validated research claim or a performance/generalization number, add an experiment and corresponding tests/benchmarks; otherwise avoid presenting prototype results as production-grade.

Performance & comparisons (measured claims present in repository benchmarks)

- Memory Broker: 30–40% reduction in peak GPU memory on benchmarked workloads when enabled.
- Quantization: up to ~4× model size reduction (fp32 -> 8/4-bit) with typical inference speedups of 2–3× where kernels support low-precision math.
- Gradient checkpointing: ~50% memory reduction during training (with recomputation trade-off).
- Speculative decoding (LLM inference): ~2× throughput improvement in repository benchmarks for applicable autoregressive workloads using a draft model.
- Kernel fusion: 20–30% CPU-side improvement on fused matmul+activation microbenchmarks.
- Sparse attention: up to ~40% memory reduction for very long sequences in sparsity-enabled attention schedules.

These are representative numbers from tests/benchmarks and change with hardware, model, and data. See tests/ and benchmarks/ for the exact commands and hardware used.

Installation & dependencies

Required:
- C++20-compatible compiler (GCC, Clang, MSVC)
- Boost (>=1.75)
- Eigen (>=3.4)
- ONNX Runtime (optional unless you use ONNX features)
- OpenMPI (for distributed runs)

Optional (recommended for GPU/LLM performance):
- CUDA toolkit (for GPU acceleration)

For full, reproducible build instructions: docs/BUILD.md

Testing & CI

- Build and run the test suite

```bash
cd build
cmake ..
make -j
ctest --output-on-failure
```

- CI: GitHub Actions runs multi-platform builds, dependency installation, feature tests, and performance validation.
- Tests include parser/interpreter correctness, edge cases, negative tests, feature-specific tests (memory broker, quantization, speculative decoding), and performance microbenchmarks.

Contribution & roadmap

We welcome contributions that include:
- A clear research or engineering motivation
- Tests that validate behavior and performance
- Reproducible benchmarks where relevant

Suggested areas where contributions are most useful:
- Improved LLM kernels (FlashAttention integration, CUDA kernels)
- More quantization-aware training recipes and accuracy-preserving QAT flows
- Additional distributed schedulers and heterogeneous resource managers

Contributing steps:
1. Fork the repository
2. Create a feature branch
3. Add tests and update docs
4. Ensure CI passes
5. Open a PR

Relevant docs and links

- docs/syntax.md — full DSL reference
- docs/tutorials.md — hands-on examples
- docs/api_reference.md — C++ runtime API
- docs/BUILD.md — build and dependency instructions
- docs/DOCKER.md — container-based builds

License

EasiScriptX is MIT-licensed. See LICENSE.txt for the full license text.

Contact

For questions, experimental results, or collaboration ideas, open an issue or a PR. For faster triage, include: the model/script, the command used, hardware details, and a short description of the unexpected behavior.
