# Task: Run SmolVLM on Android Phone via ExecuTorch (CPU-only)

## Environment

| Item | Value |
|---|---|
| Host OS | Ubuntu 22.04 |
| Root password | `1` |
| Conda env | `executorch-XNNPACK` |
| Android NDK path | `/home/zzl/project/AndroidNDK26` |
| ExecuTorch root | `/home/zzl/project4/execu-XNNPACK/executorch` |
| Model path | `/home/zzl/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-500M-Instruct/snapshots/a7da5b986cb59b408707209984f360a5f4ad7e47` |
| Target device | OnePlus Ace5 (Snapdragon 8 Gen3), connected via ADB |

You may use `sudo` with password `1` for any privileged operations.

---

## Goal

Deploy and run **SmolVLM-500M-Instruct** on the Android phone using **CPU only** (no GPU, no NPU).

---

## Steps Required

1. **Verify ADB connection**
   ```bash
   adb devices
   ```

2. **Export the model to `.pte` format**
   - Activate conda env: `conda activate executorch-XNNPACK`
   - Export SmolVLM to a `.pte` file using ExecuTorch's export pipeline (CPU backend)
   - Reference docs: `/home/zzl/project4/execu-XNNPACK/executorch/docs`
   - Additional reference: `/home/zzl/project4/execu-VULKAN/executorch` (for broader examples)
   - Reference script (NPU example, do NOT run — use as structural reference only): `/home/zzl/project4/executorch/exp_scripts/new_run.sh`

3. **Build required binaries for Android (CPU)**
   - Use Android NDK at `/home/zzl/project/AndroidNDK26`
   - Build the ExecuTorch runtime and executor binary targeting `arm64-v8a`

4. **Push to device**
   - Push `.pte` model file, runtime libraries, and executor binary to the phone via ADB

5. **Execute on device**
   - Run the model on the phone and capture output

---

## Constraints

- **CPU only** — do not use XNNPACK delegate, Vulkan, or any NPU/HTP backend
- All working files and scripts must be placed under:
  ```
  /home/zzl/project4/execu-XNNPACK/executorch/smolvlm_cpu/
  ```

---

## Deliverables

1. All scripts and build files under `smolvlm_cpu/`
2. A `note.md` file in `smolvlm_cpu/` containing:
   - Summary of every operation performed
   - Commands used at each step
   - Final experiment results and output from the phone
   - Any issues encountered and how they were resolved