#!/usr/bin/env python3
"""Generate raw .f32 reference files for frontend/models/mnist-8.onnx.

The MNIST graph is non-trivial (Conv + MaxPool + Reshape + MatMul chain), so
we compute the reference via ONNX Runtime instead of re-implementing the
pipeline by hand. The input tensor is a deterministic linear ramp normalized
to [0, 1) so the script produces byte-identical output across machines and
across runs.

Dependency versions used to bake the committed reference files:
- onnx        1.20.0
- onnxruntime 1.26.0
- numpy       2.x

Run:
    python3 tests/reference/generate_mnist8_reference.py
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "frontend" / "models" / "mnist-8.onnx"
REFERENCE_DIR = ROOT / "tests" / "reference"
INPUT_PATH = REFERENCE_DIR / "mnist8_input.f32"
EXPECTED_PATH = REFERENCE_DIR / "mnist8_expected.f32"

INPUT_SHAPE = (1, 1, 28, 28)
INPUT_NAME = "Input3"
OUTPUT_NAME = "Plus214_Output_0"


def write_f32(path: Path, array: np.ndarray) -> None:
    if array.dtype != np.float32:
        raise TypeError(f"expected float32, got {array.dtype}")
    flat = array.reshape(-1).astype(np.float32, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<" + "f" * flat.size, *flat.tolist()))


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"MNIST model not found: {MODEL_PATH}")

    n_elems = int(np.prod(INPUT_SHAPE))
    input_array = (
        np.arange(n_elems, dtype=np.float32) / np.float32(n_elems)
    ).reshape(INPUT_SHAPE)

    session = ort.InferenceSession(
        str(MODEL_PATH), providers=["CPUExecutionProvider"]
    )
    outputs = session.run([OUTPUT_NAME], {INPUT_NAME: input_array})
    expected = outputs[0]
    if expected.shape != (1, 10):
        raise RuntimeError(
            f"unexpected MNIST output shape: {expected.shape}, want (1, 10)"
        )

    write_f32(INPUT_PATH, input_array)
    write_f32(EXPECTED_PATH, expected)

    predicted = int(np.argmax(expected))
    print(f"wrote {INPUT_PATH} ({input_array.size * 4} bytes)")
    print(f"wrote {EXPECTED_PATH} ({expected.size * 4} bytes)")
    print(f"reference logits: {expected.reshape(-1).tolist()}")
    print(f"reference predicted class: {predicted}")


if __name__ == "__main__":
    main()
