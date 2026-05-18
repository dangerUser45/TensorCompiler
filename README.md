# ***Tensor Compiler***
<img src="readme_assets/preview.jpg" width="460" alt="img">

`Tensor Compiler` сейчас состоит из двух связанных модулей:

- `frontend` — импортирует ONNX, строит внутренний Graph IR, выполняет
  semantic и executable-ready verification, генерирует textual MLIR handoff и
  EXEC metadata sidecar.
- `backend` — принимает production MLIR handoff от frontend, lower'ит его в
  LLVM IR, генерирует ASM/object и собирает runnable CPU executable.

## Pipeline

```text
ONNX -> Graph IR -> Verify/VerifyExec -> MLIR + Metadata -> LLVM IR -> ASM/Object -> Executable
```

Текущий production path покрывает EXEC-compatible модели с одним runtime input,
одним runtime output, `float32` и static shapes.

## Структура

- [frontend/README.md](frontend/README.md) — импорт ONNX, verifier, MLIR/metadata handoff
- [backend/README.md](backend/README.md) — codegen, object emission, runtime linking
- `tests/reference/` — reference input/output для полного executable smoke path

## Сборка

Из корня репозитория:

```bash
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Быстрый пример

Сгенерировать и запустить executable для `exec_conv_relu.onnx`:

```bash
./build/frontend/frontend_driver frontend/models/exec_conv_relu.onnx --emit-exe=build/exec_conv_relu_runner
./build/exec_conv_relu_runner --input tests/reference/exec_conv_relu_input.f32 --output build/exec_conv_relu_actual.f32
./build/compare_f32 build/exec_conv_relu_actual.f32 tests/reference/exec_conv_relu_expected.f32 --atol 1e-5 --rtol 1e-4
```

Подробные контракты и ограничения описаны в модульных README:
[frontend/README.md](frontend/README.md) и [backend/README.md](backend/README.md).
