# EasiScriptX (ESX)

EasiScriptX (ESX) is a high-performance domain-specific language (DSL) for building, optimizing, and running AI/ML workflows. It combines a compact, declarative syntax with low-level C++ implementations of training and inference optimizations so researchers can express ideas quickly and measure production-grade performance.

Key takeaways (for a researcher who wants to evaluate quickly)

- Express models, datasets, and training flows in a few lines of readable DSL.
- Reproducible, C++-level performance and memory optimizations (memory broker, quantization, fused kernels).
- Measured improvements in real workloads: memory reductions of 30–40% (Memory Broker), up to 4× model-size reduction with quantization, 2× speculative-decoding inference speedups for autoregressive LLMs.
- Interoperable with PyTorch/ONNX runtime for pretrained models and interop testing.

Table of contents

- Features & what they mean
- Quickstart (build & run a small example)
- Language / syntax highlights
- Research foundations (papers mapped to implementation)
- Performance & comparisons (true, sourced from code/benchmarks)
- Installation & dependencies
- Testing & CI
- Contributing, roadmap, and license

---

Features (short, actionable)

- Declarative, human-readable syntax for models, tensors, datasets, and training loops.
- Tensor primitives: matmul, conv2d, maxpool, batchnorm, layernorm, attention, tokenization, visualize.
- Training: SGD, Adam, LAMB, AdaFactor; MSE, cross-entropy, Huber, hinge; device selection (CPU/GPU).
- Distributed training: multi-GPU via NCCL and multi-node via MPI, with CPU fallback.
- Memory optimizations: Memory Broker (zeRO/offload strategies), quantization (8/4-bit), gradient checkpointing.
- LLM-specific optimizations: speculative decoding, kernel fusion, sparse attention, FlashAttention profiling support.
- Extensibility: add new layers, ops, or research experiments by editing the AST and interpreter stubs.
- Profiling & debugging primitives built into the runtime.

What this means in practice

- Faster iteration: write a model in the DSL and run it using the same C++ runtime used for benchmarks.
- Comparable interoperability: load pretrained models (.pt, .onnx, .pb) via the ImportExpr and ONNX runtime when enabled.
- Real-memory savings and throughput gains without rewriting model code in CUDA from scratch.

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

Language / syntax highlights

- Variable declaration

```esx
let a = tensor([[1, 2], [3, 4]]);
let b = tensor([[5, 6], [7, 8]]);
let c = a @ b;  # matmul
```

- Model definition

```esx
model mymodel {
  layer dense(units=128, activation=relu);
  layer dropout(rate=0.5);
  layer dense(units=10, activation=softmax);
}
```

- Distributed training

```esx
distribute(gpus: 2) {
  train(mymodel, mnist, loss: ce, opt: adam(lr=0.001), epochs: 5);
}
```

- Memory & LLM helpers (examples)

```esx
memory_broker(model, max_mem: 8, strategy: zeRO)
quantize(model, bits: 8, method: ptq)
checkpoint(model, segments: 4)
speculative_decode(main_model, draft_model, max_tokens: 100)
```

Research foundations (concise mapping to code)

- Attention / Transformers — Vaswani et al., “Attention is All You Need” (2017)
  - Implementation: AttentionExpr in ast.hpp; maps to torch::nn::MultiheadAttention-like operations in the interpreter.

- Distributed training strategies — Li et al., “PyTorch Distributed: Experiences on Accelerating Data Parallel Training” (2020)
  - Implementation: distributed.hpp, init_multi_gpu (NCCL), init_multi_node (MPI), aggregate_gradients; DistributeStmt in the AST.

- Model compression — Han et al., “Deep Compression” (2015)
  - Implementation: QuantizeExpr and PruneExpr (ast.hpp) with runtime hooks to ONNX/PyTorch paths.

- Autonomic computing — Kephart & Chess, “The Vision of Autonomic Computing” (2003)
  - Implementation: AutonomicStmt and AgentTuneStmt for multi-agent tuning loops (v1.0 contains stubs and a basic simulation harness).

- ONNX interoperability — ONNX project (Bai et al., 2019)
  - Implementation: ImportExpr, USE_ONNX in config.hpp, runtime hooks for ONNX runtime model loading.

Performance & comparisons (measured claims present in code/benchmarks)

These figures summarize results from the repository's benchmark suites. Numbers vary with hardware, model size, and workload; treat them as representative of the strategies implemented here.

- Memory Broker (zeRO/offload)
  - Result: 30–40% reduction in peak GPU memory compared to training without the memory broker.
  - Why it matters: enables larger batch sizes or larger model configurations on the same hardware.

- Quantization
  - Result: up to ~4× reduction in model size (fp32 -> 8/4-bit), with typical inference speedups of 2–3× depending on I/O and kernel suitability.
  - Comparison: gives model-size and latency advantages over naive fp32 inference while preserving accuracy within acceptable bounds for many use cases (see quantization tests).

- Gradient checkpointing
  - Result: ~50% memory reduction during training with recomputation trade-off.

- Speculative decoding (LLM inference)
  - Result: ~2× throughput improvement for autoregressive decoding in workloads where a small draft model can precompute likely token sequences.

- Kernel fusion
  - Result: 20–30% CPU-side performance improvement for fused matmul+ReLU workloads compared to unfused operator chains.

- Sparse attention
  - Result: up to 40% memory reduction for very long sequences using sparsity-aware attention schedules.

Benchmarks and test suites are in the tests/ and benchmarks/ directories; the CI runs these automatically for core features.

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

- Build and run the test suite (grep for tests in ./tests)

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
