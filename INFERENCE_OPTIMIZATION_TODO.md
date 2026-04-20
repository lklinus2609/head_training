# Inference optimization — remaining work

Full design in `~/.claude/plans/memoized-shimmying-tulip.md`. Target device: **Jetson Orin Nano Super** (Ampere GPU, 32 Tensor Cores). Target usage: real-time streaming (30 fps, <33 ms/frame) and offline batch eval.

## Done

- **A1** `scripts/export_inference_checkpoint.py` + `utils.checkpoint.save_inference_checkpoint` — strips optimizer / discriminator / RNG from a stage-2 or stage-3 checkpoint (67–74% size reduction).
- **A2** `viewer/generate_sequence.py --fp16` — wraps inference in `torch.autocast(cuda, float16)`.
- **B1** `models.layers.AdaLNTransformerDecoderLayerInference` — inference-only mirror layer with split Q/K/V + `scaled_dot_product_attention`, supports KV cache. Layer parity vs. training: 4e-7 max abs diff.
- **B2** `models/generator_inference.py` + `viewer/generate_sequence.py --kv_cache` — `GeneratorInference.from_training_generator(gen)` rebuilds the model with cached AR. Parity vs. training generator: 3e-6 at H=4, 7e-6 at H=32 (FP32 noise).

All of the above is additive — training code path is byte-identical (verified: no training module imports any of the new symbols).

## Not done

### 1. Golden parity reference — blocked on a trained checkpoint

Pick one fixed val-set clip + emotion. Run the current FP32 PyTorch path through `viewer/generate_sequence.py` (no `--fp16`, no `--kv_cache`) once for a stage-2 checkpoint and once for a stage-3 checkpoint. Save outputs as `parity_reference_stage2.npy` / `parity_reference_stage3.npy`.

Every subsequent optimization must reproduce these outputs within:

| Variant                             | L1 threshold (normalized space) |
|-------------------------------------|---------------------------------|
| Stripped checkpoint (A1)            | bitwise-equal                   |
| `--fp16` (A2)                       | < 0.01                          |
| `--kv_cache` FP32 (B2)              | < 1e-4                          |
| `--kv_cache --fp16`                 | < 0.01                          |
| ONNX via onnxruntime-gpu on x86     | < 0.02                          |
| TRT FP16 on Jetson                  | < 0.05                          |

Also record a latency number on the Jetson with the FP32 no-cache baseline — the headline metric is delta vs. that.

### 2. C1 — ONNX export (new `scripts/export_onnx.py`)

Export two graphs (the standard LLM-decoder split):
- **`generator_prefill.onnx`** — audio encoder + first decoder pass over the initial `prev_token`. Inputs: `audio`, `emotion`, `prev_expression`. Outputs: first prediction, initial per-layer `(K, V)` cache, cached `audio_enc` tensor.
- **`generator_step.onnx`** — one AR step. Inputs: `prev_pred_token`, `audio_enc`, per-layer `past_kv_k` / `past_kv_v`, `position`. Outputs: `next_pred`, updated per-layer `(K, V)`.

Use `torch.onnx.export` with `dynamic_axes` on `T_audio`, KV seq length, and batch dim. Opset 17+. Validate output parity on x86 via `onnxruntime-gpu` against the `GeneratorInference` output from B2 before moving the files to the Jetson. Thresholds: see above.

Implementation note: reuse `GeneratorInference` from `models/generator_inference.py`. The `forward` there already separates prefill (first step with `past_kv=None`) from cached steps, so splitting into two exportable modules is mostly plumbing. Avoid Python control flow in traced graphs — shape-polymorphic code only.

### 3. C2 — TensorRT engines (build on the Jetson itself)

TRT engines are device-specific — cannot build on x86 and run on ARM. On the Orin Nano:

```bash
trtexec --onnx=generator_prefill.onnx --fp16 --saveEngine=generator_prefill.engine
trtexec --onnx=generator_step.onnx    --fp16 --saveEngine=generator_step.engine
trtexec --loadEngine=generator_step.engine --dumpProfile
```

Record per-step latency. Target <5 ms per step at FP16 on Orin Nano.

### 4. C3 — TRT runtime wrapper (new `viewer/generate_sequence_trt.py`)

Mirror `viewer/generate_sequence.py` but call the two engines via the `tensorrt` Python bindings. Keep `generate_sequence.py` as the PyTorch reference / CI parity baseline — do not remove it.

Sliding-window outer loop stays in Python; each window calls `prefill` once + `step` (H-1) times. Allocate engine I/O buffers once outside the sliding loop and reuse.

### 5. Stage D — INT8 quantization (skip unless FP16 misses 30 fps)

TensorRT `Int8EntropyCalibrator2` against ~100 val audio windows. Model is small (~5–10 M params); accuracy risk is real. Compare against the FP16 output with the L1 thresholds above. **Skip unless latency measurement forces it.**

## Execution notes

- This machine (Windows) is not a valid runtime target. Training happens on an LS6 cluster node, inference deploys to the Jetson. Do not try to smoke-test the pipeline locally.
- Training pipeline is not touched by any of the above — existing `.pt` checkpoints load unchanged via the weight-remap path in `GeneratorInference.from_training_generator`.
- Keep `models/generator.py` and `AdaLNTransformerDecoderLayer` byte-identical. All inference-side changes go through the mirror classes.
