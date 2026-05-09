#!/usr/bin/env python3
"""Generate raw .f32 reference files for frontend/models/exec_conv_relu.onnx."""

from __future__ import annotations

import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = ROOT / "tests" / "reference"
INPUT_PATH = REFERENCE_DIR / "exec_conv_relu_input.f32"
EXPECTED_PATH = REFERENCE_DIR / "exec_conv_relu_expected.f32"


def flatten_index(shape: tuple[int, ...], coords: tuple[int, ...]) -> int:
    linear = 0
    for dim, coord in zip(shape, coords):
        linear = linear * dim + coord
    return linear


def write_f32(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<" + "f" * len(values), *values))


def main() -> None:
    input_shape = (1, 2, 8, 8)
    weight_shape = (2, 2, 2, 2)
    output_shape = (1, 2, 7, 7)

    input_values = [float(i) / 128.0 for i in range(1 * 2 * 8 * 8)]
    weight_values = [float(i) / 16.0 for i in range(2 * 2 * 2 * 2)]
    bias_values = [0.25, -0.5]

    output_values: list[float] = []
    for n in range(output_shape[0]):
        for f in range(output_shape[1]):
            for oh in range(output_shape[2]):
                for ow in range(output_shape[3]):
                    acc = 0.0
                    for c in range(input_shape[1]):
                        for kh in range(weight_shape[2]):
                            for kw in range(weight_shape[3]):
                                input_idx = flatten_index(
                                    input_shape, (n, c, oh + kh, ow + kw)
                                )
                                weight_idx = flatten_index(
                                    weight_shape, (f, c, kh, kw)
                                )
                                acc += input_values[input_idx] * weight_values[weight_idx]
                    acc += bias_values[f]
                    output_values.append(max(acc, 0.0))

    write_f32(INPUT_PATH, input_values)
    write_f32(EXPECTED_PATH, output_values)
    print(f"wrote {INPUT_PATH}")
    print(f"wrote {EXPECTED_PATH}")


if __name__ == "__main__":
    main()
