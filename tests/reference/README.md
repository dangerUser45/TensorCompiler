# Reference Tests

This directory contains the cross-role EXEC reference fixture.

`exec_conv_relu_input.f32` is a raw little-endian `float32` tensor with shape
`1x2x8x8`.

`exec_conv_relu_expected.f32` is the expected raw little-endian `float32` output
for `frontend/models/exec_conv_relu.onnx`, shape `1x2x7x7`.

`mnist8_input.f32` is a raw little-endian `float32` tensor with shape
`1x1x28x28`.

`mnist8_expected.f32` is the expected raw little-endian `float32` output for
`frontend/models/mnist-8.onnx`, shape `1x10`.

Regenerate the Conv/Relu files with:

```bash
python3 tests/reference/generate_exec_conv_relu_reference.py
```

Regenerate the MNIST files with:

```bash
python3 tests/reference/generate_mnist8_reference.py
```

The executable check builds a runner through `frontend_driver --emit-exe`, runs
it on the raw input, and compares the output with `compare_f32`.
