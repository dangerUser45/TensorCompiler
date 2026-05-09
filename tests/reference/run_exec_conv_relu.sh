#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "Usage: $0 <frontend_driver> <model.onnx> <input.f32> <expected.f32> <compare_f32> <output-dir>" >&2
  exit 2
fi

DRIVER="$1"
MODEL="$2"
INPUT="$3"
EXPECTED="$4"
COMPARE="$5"
OUT_DIR="$6"

mkdir -p "${OUT_DIR}"

EXE="${OUT_DIR}/exec_conv_relu_runner"
ACTUAL="${OUT_DIR}/exec_conv_relu_actual.f32"

"${DRIVER}" "${MODEL}" --emit-exe="${EXE}"
"${EXE}" --input "${INPUT}" --output "${ACTUAL}"
"${COMPARE}" "${ACTUAL}" "${EXPECTED}" --atol 1e-5 --rtol 1e-4
