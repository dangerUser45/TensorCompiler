# Tensor Compiler Backend

Backend отвечает за:
- разбор production MLIR handoff от frontend;
- lowering поддержанного MLIR subset в LLVM IR;
- генерацию ASM и object file через LLVM toolchain;
- линковку object file с CPU runtime wrapper;
- запуск с raw `.f32` input/output через сгенерированный executable.

## Текущий pipeline

`Textual MLIR -> LLVM IR -> ASM/Object -> Executable -> Run(.f32)`

- `Textual MLIR` — production MLIR с entry `func.func @tc_model`.
- `LLVM IR` — C ABI функция `tc_model(float* input, float* output)`.
- `ASM/Object` — генерируются через `llc`.
- `Executable` — линкуется через `clang++` с `backend/runtime/tc_model_runner.cpp`.

## Поддержанный MLIR subset

Backend принимает MLIR, который генерирует frontend для EXEC-compatible моделей:

- `func.func @tc_model`
- `arith.constant dense<...>`
- `tensor.empty`
- `linalg.broadcast`
- `arith.addf`
- `arith.mulf`
- `arith.maximumf`
- `linalg.matmul`
- `linalg.transpose`
- `linalg.conv_2d_nchw_fchw`

Ограничения MVP:
- только `float32`;
- только static tensor shapes;
- Conv только NCHW/FCHW, stride/pad/dilation default frontend subset;
- executable runner принимает один input tensor и пишет один output tensor.

## Структура

- `include/frontend_mlir.hpp` — parser frontend MLIR subset
- `include/codegen_driver.hpp` — API для LLVM IR и ASM
- `include/object_emitter.hpp` — API для object emission
- `include/executable_linker.hpp` — API для линковки executable
- `src/frontend_mlir.cpp` — parser implementation
- `src/codegen_driver.cpp` — LLVM IR/ASM emission
- `src/object_emitter.cpp` — `.o` emission через `llc`
- `src/executable_linker.cpp` — runtime linking через `clang++`
- `runtime/tc_model_runner.cpp` — CPU runner для raw `.f32`
- `tests/` — backend smoke/unit/e2e tests

## Требования

- CMake >= 3.20
- C++17 compiler
- LLVM tools:
  - `llvm-config`
  - `llc`
  - `clang++`

MLIR package is optional in this repository state. If `MLIR_DIR` is not set,
backend still builds the production handoff path implemented for the frontend
MLIR subset.

## Сборка

Backend-only:

```bash
cmake -S backend -B backend/build
cmake --build backend/build --parallel
ctest --test-dir backend/build --output-on-failure
```

Root build with frontend CLI integration:

```bash
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Использование через frontend driver

Из root build:

```bash
./build/frontend/frontend_driver frontend/models/exec_conv_relu.onnx --emit-llvm=build/exec_conv_relu.ll
./build/frontend/frontend_driver frontend/models/exec_conv_relu.onnx --emit-asm=build/exec_conv_relu.s
./build/frontend/frontend_driver frontend/models/exec_conv_relu.onnx --emit-object=build/exec_conv_relu.o
./build/frontend/frontend_driver frontend/models/exec_conv_relu.onnx --emit-exe=build/exec_conv_relu_runner
```

Backend-related опции driver:

- `--emit-llvm[=<output.ll>]` — записать lowered LLVM IR
- `--emit-asm[=<output.s>]` — записать assembly через `llc`
- `--emit-object[=<output.o>]` — записать object file
- `--emit-exe[=<output_binary>]` — собрать runnable executable
- `--target=<triple>` — передать target triple в LLVM/`llc`
- `--pass-pipeline=<pipeline>` — зарезервировано для явного pass pipeline

Если путь не задан, используются build-директории `frontend_driver`:

- LLVM IR: `build/frontend/llvm/<model_name>.ll`
- ASM: `build/frontend/asm/<model_name>.s`
- object: `build/frontend/object/<model_name>.o`
- executable: `build/frontend/bin/<model_name>`
- metadata sidecar для `--emit-exe`: `build/frontend/metadata/<model_name>.json`

Backend path доступен только для моделей, которые проходят frontend
`--verify-exec` contract: один runtime input, один runtime output, `float32`,
static shapes, валидные initializer данные.

Запуск executable:

```bash
./build/exec_conv_relu_runner \
  --input tests/reference/exec_conv_relu_input.f32 \
  --output build/exec_conv_relu_actual.f32
```

Сравнение:

```bash
./build/compare_f32 \
  build/exec_conv_relu_actual.f32 \
  tests/reference/exec_conv_relu_expected.f32 \
  --atol 1e-5 --rtol 1e-4
```

## Runtime contract

Backend генерирует и линкует CPU runner под фиксированный ABI:

- exported symbol: `extern "C" void tc_model(const float* input0, float* output0)`
- runner CLI: `--input <input.f32> --output <output.f32>`
- input/output byte sizes берутся из frontend metadata JSON
- runtime ожидает exact raw little-endian `float32` payload без header

Если metadata JSON не содержит `byte_size` для input/output или object file не
был сгенерирован, линковка executable завершается backend diagnostic-ошибкой.

## Тесты

Backend tests:

```bash
ctest --test-dir backend/build --output-on-failure
```

Full executable reference test:

```bash
ctest --test-dir build --output-on-failure -R exec_conv_relu
```

Покрытие включает:

- parser test для frontend MLIR handoff subset
- codegen tests для LLVM IR и ASM
- object emission test через `llc`
- executable linker test с реальным запуском runner
- full reference test для `frontend/models/exec_conv_relu.onnx`

## Форматирование

Форматирование управляется корневым `.clang-format`.

```bash
clang-format -style=file -i backend/include/*.hpp backend/src/*.cpp backend/runtime/*.cpp backend/tests/*.cpp
```
