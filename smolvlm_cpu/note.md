# SmolVLM CPU Android Experiment Note

## Summary

Goal: run `HuggingFaceTB/SmolVLM-500M-Instruct` on the connected Android phone with ExecuTorch, CPU only, from files under `smolvlm_cpu/`.

Status:
- Export to ExecuTorch `.pte`: succeeded
- Android CPU runner build: succeeded
- Push to device: succeeded
- On-device model load and multimodal prefill start: succeeded
- End-to-end text generation on phone: not completed within a reasonable validation window on CPU-only

## Files Added

- `smolvlm_cpu/export_smolvlm_cpu.py`
- `smolvlm_cpu/prepare_image.py`
- `smolvlm_cpu/smolvlm_runner.cpp`
- `smolvlm_cpu/CMakeLists.txt`
- `smolvlm_cpu/build_android.sh`
- `smolvlm_cpu/run_on_device.sh`

Artifacts produced:
- `smolvlm_cpu/artifacts/smolvlm_cpu.pte`
- `smolvlm_cpu/artifacts/statue_of_liberty_512_chw_f32.bin`
- `smolvlm_cpu/artifacts/tokenizer.json`

## Commands Used

ADB check:

```bash
adb devices -l
```

Prepare image tensor:

```bash
/home/zzl/tools/conda/envs/executorch-XNNPACK/bin/python \
  smolvlm_cpu/prepare_image.py \
  --image smolvlm_cpu/artifacts/statue_of_liberty.jpg \
  --output smolvlm_cpu/artifacts/statue_of_liberty_512_chw_f32.bin
```

Export model:

```bash
/home/zzl/tools/conda/envs/executorch-XNNPACK/bin/python \
  smolvlm_cpu/export_smolvlm_cpu.py \
  --model-dir /home/zzl/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-500M-Instruct/snapshots/a7da5b986cb59b408707209984f360a5f4ad7e47 \
  --output smolvlm_cpu/artifacts/smolvlm_cpu.pte \
  --max-seq-len 256 \
  --max-context-len 256
```

Build Android runner:

```bash
smolvlm_cpu/build_android.sh
```

Default device run:

```bash
smolvlm_cpu/run_on_device.sh
```

Shorter validation run without repush:

```bash
SKIP_PUSH=1 MAX_NEW_TOKENS=1 smolvlm_cpu/run_on_device.sh
```

## Issues And Fixes

1. `to_executorch()` initially failed because `exir/_serialize/program.fbs` and `exir/_serialize/scalar_type.fbs` were missing in this tree.
   Fix: copied `schema/program.fbs` and `schema/scalar_type.fbs` into `exir/_serialize/`.

2. Android sample CMake failed to configure.
   Fix: corrected `gflags_DIR` and `CMAKE_FIND_ROOT_PATH` in `smolvlm_cpu/CMakeLists.txt`.

3. First on-device run aborted in CPU flash attention with:

```text
FlashAttention does not support num kv heads > num query heads.
Got num query heads=15 num key heads:256
```

Cause: a cache-layout rewrite in `export_smolvlm_cpu.py` incorrectly changed K/V from the correct `[B, S, H, D]` layout to `[B, H, S, D]`.

Fix: removed that rewrite for the CPU export path and re-exported the model.

## Final Device Result

The corrected model loads on the phone and enters multimodal execution successfully. Representative phone output:

```text
Metadata: use_sdpa_with_kv_cache = 1
Metadata: get_max_context_len = 256
Metadata: get_max_seq_len = 256
RSS after loading model: 2138.8 MiB
Image tensor dim: 4, dtype: Float
```

Observed behavior after the fix:
- The previous SDPA crash is gone.
- `smolvlm_runner` stays active on-device at about `98.9%` CPU.
- A `max_new_tokens=64` run did not finish during a several-minute validation window.
- Even a `max_new_tokens=1` validation run did not return a token within about 5 minutes after startup.

Conclusion: the CPU-only ExecuTorch deployment is operational enough to export, load, and start multimodal inference on the phone, but practical text generation latency is currently too high for this setup.
