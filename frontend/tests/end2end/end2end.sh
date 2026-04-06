#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  end-to-end.sh <frontend_driver_bin> [expected_hashes_file] [models_dir]

Example:
  ./codex/end-to-end.sh ./build/frontend/frontend_driver
  ./codex/end-to-end.sh ./build/frontend/frontend_driver ./codex/expected_hashes.txt ./frontend/models

Expected file format (2 columns):
  <model_filename.onnx> <hex_hash>
EOF
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DRIVER_BIN="$1"
EXPECTED_FILE="${2:-${SCRIPT_DIR}/expected_hashes.txt}"
MODELS_DIR="${3:-${REPO_ROOT}/frontend/models}"
OUT_DIR="${REPO_ROOT}/build/e2e_hash"
HASH_DIR="${OUT_DIR}/hashes"
ACTUAL_FILE="${OUT_DIR}/actual_hashes.txt"
NORMALIZED_EXPECTED="${OUT_DIR}/expected_hashes.normalized.txt"

if [[ ! -x "${DRIVER_BIN}" ]]; then
  echo "ERROR: frontend_driver binary is not executable: ${DRIVER_BIN}" >&2
  exit 1
fi

if [[ ! -d "${MODELS_DIR}" ]]; then
  echo "ERROR: models directory not found: ${MODELS_DIR}" >&2
  exit 1
fi

if [[ ! -f "${EXPECTED_FILE}" ]]; then
  echo "ERROR: expected hashes file not found: ${EXPECTED_FILE}" >&2
  exit 1
fi

mkdir -p "${HASH_DIR}"
: > "${ACTUAL_FILE}"

awk '
  BEGIN { FS = "[[:space:]]+" }
  /^[[:space:]]*#/ { next }
  NF == 0 { next }
  NF != 2 {
    printf "ERROR: invalid line in expected file: %s\n", $0 > "/dev/stderr";
    exit 2;
  }
  { print $1, $2 }
' "${EXPECTED_FILE}" | sort > "${NORMALIZED_EXPECTED}"

if [[ ! -s "${NORMALIZED_EXPECTED}" ]]; then
  echo "ERROR: expected hashes file has no model entries: ${EXPECTED_FILE}" >&2
  exit 1
fi

while read -r model_file _expected_hash; do
  model_path="${MODELS_DIR}/${model_file}"
  if [[ ! -f "${model_path}" ]]; then
    echo "ERROR: model listed in expected hashes is missing: ${model_path}" >&2
    exit 1
  fi

  hash_file="${HASH_DIR}/${model_file%.onnx}.hash"
  "${DRIVER_BIN}" "${model_path}" --hash="${hash_file}" >/dev/null

  hash_value="$(tr -d '[:space:]' < "${hash_file}")"
  if [[ -z "${hash_value}" ]]; then
    echo "ERROR: empty hash for ${model_file}" >&2
    exit 1
  fi

  printf "%s %s\n" "${model_file}" "${hash_value}" >> "${ACTUAL_FILE}"
done < "${NORMALIZED_EXPECTED}"

sort -o "${ACTUAL_FILE}" "${ACTUAL_FILE}"

if diff -u "${NORMALIZED_EXPECTED}" "${ACTUAL_FILE}"; then
  echo "E2E hash check: OK"
  exit 0
fi

echo "E2E hash check: FAILED" >&2
echo "Actual hashes: ${ACTUAL_FILE}" >&2
exit 1
