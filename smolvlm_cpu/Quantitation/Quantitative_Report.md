# SmolVLM CPU Quantization Analysis

## 1. Current SmolVLM CPU pipeline

The current `smolvlm_cpu` path is:

1. `prepare_image.py` converts the input image to normalized `float32` CHW data.
2. `export_smolvlm_cpu.py` loads Hugging Face SmolVLM on CPU in `torch.float32`, splits it into:
   - `vision_encoder`
   - `token_embedding`
   - `text_decoder`
3. Those three modules are exported and assembled into one `.pte`.
4. `build_android.sh` builds an Android runner.
5. `run_on_device.sh` pushes the `.pte`, tokenizer, image tensor, and runner binary to the phone and executes inference.

## 2. What quantization is active right now

### Conclusion

No quantization is active in the current `smolvlm_cpu` deployment.

This is not PTQ, not QAT, not static quantization, and not dynamic quantization. It is effectively an FP32 export:

- The model is loaded with `torch_dtype=torch.float32`.
- Example tensors for image and embeddings are `float32`.
- The export uses `partitioner=[]`, so no XNNPACK backend lowering is applied.
- The Android build explicitly disables XNNPACK with `-DEXECUTORCH_BUILD_XNNPACK=OFF`.

### Effective precision

- Weights: `W32`
- Activations: `A32`
- Overall mode: `W32A32`

### Why the runner still links quantized kernels

`smolvlm_cpu/CMakeLists.txt` links `quantized_kernels` and `quantized_ops_lib`, but the exported graph itself is not quantized, so those kernels are available in the binary without being used by this model.

## 3. Evidence in the codebase

- Current SmolVLM export is FP32 and not lowered to XNNPACK:
  - `smolvlm_cpu/export_smolvlm_cpu.py:359-364`
  - `smolvlm_cpu/export_smolvlm_cpu.py:381-385`
  - `smolvlm_cpu/export_smolvlm_cpu.py:419-426`
- Current Android build disables XNNPACK:
  - `smolvlm_cpu/build_android.sh:24-27`
  - `smolvlm_cpu/build_android.sh:53-56`
- Quantized kernels are linked, but only as runtime capability:
  - `smolvlm_cpu/CMakeLists.txt:64-65`
- Current on-device result shows CPU-only inference is functionally running but too slow:
  - `smolvlm_cpu/note.md:101-109`

## 4. Quantization modes supported by the broader ExecuTorch + XNNPACK stack

There are two relevant quantization paths in this repository:

1. XNNPACK PT2E graph-mode quantization
2. TorchAO source-transform quantization used by the LLM export stack

They overlap, but they are not identical.

### 4.1 XNNPACK PT2E quantization

The generic XNNPACK quantizer exposes:

- static INT8 quantization
- dynamic INT8 quantization
- per-tensor and per-channel variants
- QAT variants for the built-in symmetric INT8 flow

The core config generator uses `torch.int8` activations and `torch.int8` weights, with:

- dynamic activations when `is_dynamic=True`
- per-channel weights when `is_per_channel=True`
- QAT observers/fake-quant when `is_qat=True`

The repo also exposes official XNNPACK recipe types for:

- `xnnpack_fp32`
- `xnnpack_pt2e_int8_dynamic_per_channel`
- `xnnpack_pt2e_int8_static_per_channel`
- `xnnpack_pt2e_int8_static_per_tensor`
- `xnnpack_torchao_int8da_int4w_per_channel`
- `xnnpack_torchao_int8da_int4w_per_tensor`

### 4.2 LLM-specific PT2E helper for XNNPACK

The LLM export helper adds two practical XNNPACK dynamic quantization options for transformer-style models:

- `xnnpack_dynamic`
  - dynamic activations
  - per-channel INT8 weights
  - effectively `W8A8`
- `xnnpack_dynamic_qc4`
  - dynamic activations
  - per-channel 4-bit weights
  - effectively `W4A8`

The same helper can also enable embedding quantization via `embedding`, but that is separate from the linear-layer mode.

### 4.3 TorchAO source-transform quantization for LLMs

The LLM export path also supports source-transform quantization before `torch.export()`:

- Linear:
  - `4w`
  - `8w`
  - `8da4w`
  - `8da8w`
- Embedding:
  - `4w`
  - `8w`

Interpretation:

- `4w`: weight-only 4-bit, activations stay floating-point
- `8w`: weight-only 8-bit, activations stay floating-point
- `8da4w`: int8 dynamic activations + int4 weights
- `8da8w`: int8 dynamic activations + int8 weights

## 5. Support matrix

Status legend:

- `Active`: used by the current `smolvlm_cpu` pipeline
- `Supported`: available somewhere in this repository for ExecuTorch CPU/XNNPACK-style export
- `Disabled here`: available in the stack, but not wired into `smolvlm_cpu` today
- `Unsupported`: not exposed as a CPU/XNNPACK option in this tree

| Mode | Precision | In repo | In current `smolvlm_cpu` | Notes |
| --- | --- | --- | --- | --- |
| FP32 baseline | `W32A32` | Supported | Active | Current export path |
| PT2E static per-tensor INT8 | `W8A8` | Supported | Disabled here | XNNPACK quantizer/recipes |
| PT2E static per-channel INT8 | `W8A8` | Supported | Disabled here | XNNPACK quantizer/recipes |
| PT2E dynamic per-channel INT8 | `W8A8` | Supported | Disabled here | Recommended in repo for transformer-style models |
| PT2E dynamic per-channel QC4 | `W4A8` | Supported | Disabled here | Exposed by `xnnpack_dynamic_qc4` |
| TorchAO dynamic act + int4 weight | `W4A8` | Supported | Disabled here | `8da4w`; strongest low-bit mobile option in this tree |
| TorchAO dynamic act + int8 weight | `W8A8` | Supported | Disabled here | `8da8w` |
| TorchAO weight-only INT8 | `W8A32` | Supported | Disabled here | Activations remain float |
| TorchAO weight-only INT4 | `W4A32` | Supported | Disabled here | Activations remain float |
| Embedding-only INT8 | embedding `8w` | Supported | Disabled here | Separate from linear quantization |
| Embedding-only INT4 | embedding `4w` | Supported | Disabled here | Separate from linear quantization |
| FP16 XNNPACK CPU mode | `W16A16` | Unsupported | Unsupported | XNNPACK precision config here is FP32/static quant/dynamic quant only |
| INT8 weight + FP16 activation | `W8A16` | Unsupported | Unsupported | Not exposed by current CPU/XNNPACK stack |
| INT4 weight + FP16 activation | `W4A16` | Unsupported | Unsupported | Not exposed by current CPU/XNNPACK stack |

## 6. Practical backend constraints

### Current `smolvlm_cpu` constraints

- XNNPACK is fully disabled in `build_android.sh`.
- The export script does not call any quantizer.
- The export script does not call any XNNPACK partitioner.

So all quantized modes are currently disabled in practice for this SmolVLM pipeline.

### XNNPACK constraints in the LLM export stack

The LLM partitioner helper defaults to a dynamically-quantized-only XNNPACK partitioner. The comment explains why:

- only dynamically quantized linear layers are partitioned by default
- non-DQLinear XNNPACK partitioning was left disabled because of instability/segfault concerns

That means the practical XNNPACK path for a transformer model like SmolVLM is centered on dynamic linear quantization, not broad static whole-graph quantization.

### Embedding constraints

- PT2E embedding quantization does not support groupwise quantization in `quantizer_lib.py`
- TorchAO source-transform embedding quantization is more flexible and supports 4-bit and 8-bit embedding compression

## 7. Trade-offs

These trade-offs are partly inferred from the quantization schemes and repository guidance. I did not find SmolVLM-specific accuracy benchmarks in this tree.

| Mode | Accuracy | Latency | Memory | Notes |
| --- | --- | --- | --- | --- |
| `W32A32` | Best | Worst | Worst | Matches current slow phone result |
| `W8A8` static | High | Better than FP32 | Much better than FP32 | Needs calibration; more natural for conv/CNN-style paths |
| `W8A8` dynamic | High | Better than FP32 | Much better than FP32 | Good transformer baseline; simpler PTQ flow |
| `W4A8` dynamic | Moderate to high | Best among quantized options here | Best among practical linear modes | Best mobile CPU target if accuracy remains acceptable |
| `W8A32` | High | Small to moderate gain | Better than FP32 | Easier than full activation quant, but less speedup |
| `W4A32` | Moderate to high | Moderate gain | Very good | Strong compression, but less complete than `W4A8` |

## 8. Recommendation

### Best fit for this target hardware

For CPU-only inference on the Snapdragon 8 Gen 3 phone, the best target configuration is:

- `W4A8` for text-decoder linear layers
- optionally `4w` for embeddings if memory pressure remains high
- leave unsupported parts in floating-point fallback

Concretely, the two strongest candidates are:

1. `xnnpack_dynamic_qc4`
2. `8da4w` plus `4w` embeddings if needed

### Why

- The current `W32A32` pipeline is operational but far too slow.
- The repository’s transformer/mobile guidance consistently points toward dynamic per-channel quantization for transformer models.
- The local Llama mobile docs explicitly describe 4-bit weight + 8-bit dynamic activation PTQ as the practical high-end-phone configuration.
- This gives the largest reduction in model bandwidth and memory traffic, which is usually the main limiter for CPU LLM-style decoding.

### Conservative fallback

If `W4A8` quality loss is unacceptable, the next-best option is `W8A8` dynamic per-channel.

That will likely preserve accuracy better, but it will not reduce memory traffic as much as `W4A8`.

## 9. Bottom line

- Current active method: no quantization, `W32A32`
- Current `smolvlm_cpu` path is CPU-only and XNNPACK-disabled
- The broader stack supports `W8A8` static/dynamic and `W4A8` dynamic low-bit transformer paths
- The best mobile CPU direction for this model is `W4A8` dynamic quantization for linear layers, with optional 4-bit embeddings
