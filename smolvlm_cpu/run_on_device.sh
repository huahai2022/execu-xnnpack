#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/zzl/project4/execu-XNNPACK/executorch"
WORK_DIR="${ROOT_DIR}/smolvlm_cpu"
BUILD_DIR="${ROOT_DIR}/cmake-out-android-smolvlm-cpu"
DEVICE_DIR="/data/local/tmp/smolvlm_cpu"
PROMPT="${PROMPT:-Can you describe this image?}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
SKIP_PUSH="${SKIP_PUSH:-0}"

adb shell "mkdir -p ${DEVICE_DIR}"

if [[ "${SKIP_PUSH}" != "1" ]]; then
  adb push "${WORK_DIR}/artifacts/smolvlm_cpu.pte" "${DEVICE_DIR}/smolvlm_cpu.pte"
  adb push "${WORK_DIR}/artifacts/tokenizer.json" "${DEVICE_DIR}/tokenizer.json"
  adb push "${WORK_DIR}/artifacts/statue_of_liberty_512_chw_f32.bin" "${DEVICE_DIR}/image_512_chw_f32.bin"
  adb push "${BUILD_DIR}/smolvlm_cpu/smolvlm_runner" "${DEVICE_DIR}/smolvlm_runner"
fi

adb shell "chmod +x ${DEVICE_DIR}/smolvlm_runner"
adb shell "${DEVICE_DIR}/smolvlm_runner \
  --model_path=${DEVICE_DIR}/smolvlm_cpu.pte \
  --tokenizer_path=${DEVICE_DIR}/tokenizer.json \
  --image_bin_path=${DEVICE_DIR}/image_512_chw_f32.bin \
  --prompt='${PROMPT}' \
  --temperature=${TEMPERATURE} \
  --max_new_tokens=${MAX_NEW_TOKENS}"
