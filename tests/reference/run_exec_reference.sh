#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 7 ]]; then
  echo "Usage: $0 <test-name> <frontend_driver> <model.onnx> <input.f32> <expected.f32> <compare_f32> <output-dir>" >&2
  exit 2
fi

TEST_NAME="$1"
DRIVER="$2"
MODEL="$3"
INPUT="$4"
EXPECTED="$5"
COMPARE="$6"
OUT_DIR="$7"

mkdir -p "${OUT_DIR}"

EXE="${OUT_DIR}/${TEST_NAME}_runner"
ACTUAL="${OUT_DIR}/${TEST_NAME}_actual.f32"

"${DRIVER}" "${MODEL}" --emit-exe="${EXE}"
"${EXE}" --input "${INPUT}" --output "${ACTUAL}"
"${COMPARE}" "${ACTUAL}" "${EXPECTED}" --atol 1e-5 --rtol 1e-4
