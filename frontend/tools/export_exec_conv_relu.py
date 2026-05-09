#!/usr/bin/env python3
"""Export a deterministic Conv2d+bias+Relu ONNX fixture for frontend EXEC checks.

This script intentionally does not run as part of the CMake build. Run it
manually when you need to create frontend/models/exec_conv_relu.onnx.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "models" / "exec_conv_relu.onnx"


class ExecConvRelu(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=2,
            kernel_size=(2, 2),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv(x))


def build_model() -> ExecConvRelu:
    model = ExecConvRelu().eval()
    with torch.no_grad():
        weight = torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2) / 16.0
        bias = torch.tensor([0.25, -0.5], dtype=torch.float32)
        model.conv.weight.copy_(weight)
        model.conv.bias.copy_(bias)
    return model


def sample_input() -> torch.Tensor:
    return torch.arange(1 * 2 * 8 * 8, dtype=torch.float32).reshape(1, 2, 8, 8) / 128.0


def export_model(output: Path, opset: int) -> None:
    model = build_model()
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        sample_input(),
        str(output),
        export_params=True,
        opset_version=opset,
        do_constant_folding=False,
        dynamo=False,
        input_names=["input"],
        output_names=["output"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output ONNX path.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version for torch.onnx.export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_model(args.output, args.opset)
    print(f"ONNX model written to: {args.output}")


if __name__ == "__main__":
    main()
