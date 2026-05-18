# Tensor Compiler Frontend

Frontend отвечает за:
- импорт ONNX в внутренний IR;
- semantic verification и executable-ready verification;
- генерацию графического дампа (`.dot`);
- генерацию production textual MLIR handoff для backend;
- генерацию EXEC metadata sidecar (`.json`).

## Текущий pipeline

```text
Load -> Import(Graph IR) -> [Verify | VerifyExec] -> [Dump | EmitMLIR | EmitMetadata]
```

- `Load` читает ONNX protobuf.
- `Import` строит typed `tc::frontend::Graph`.
- `Verify` выполняет semantic/graph checks без запуска backend.
- `VerifyExec` добавляет строгий executable contract: один runtime input, один
  runtime output, `float32`, static shapes, valid initializer data.
- `Dump` пишет Graphviz DOT.
- `EmitMLIR` пишет deterministic textual MLIR с entry `func.func @tc_model`.
- `EmitMetadata` пишет deterministic JSON sidecar для runtime/EXEC handoff.

Importer не выполняет скрытую execution-валидацию: ошибки import, semantic и
backend handoff разделены по стадиям.

## Поддерживаемые операции

Импортируются в deterministic Graph IR:

- `Relu`
- `Add`
- `Mul`
- `MatMul`
- `Transpose`
- `Gemm` как normalized `MatMul` или `MatMul + Add`
- `Conv` как fixed 2D subset; 3-input ONNX `Conv` нормализуется в `Conv + Add`

Production MLIR сейчас покрывает:

- `Relu` -> `arith.maximumf`
- `Add` -> `arith.addf`
- `Mul` -> `arith.mulf`
- `MatMul` -> `linalg.matmul`
- `Transpose` -> `linalg.transpose`
- `Conv` -> `linalg.conv_2d_nchw_fchw`

Unsupported production paths fail; production MLIR не возвращает успешный
fallback с `TODO(tc)`.

## Conv subset

Поддерживаемый `Conv` subset:

- dtype: `float32`
- input layout: `NCHW`
- weight layout: `FCHW`
- ranks: input/weight/output rank 4
- 2D convolution only
- `group=1`
- `auto_pad` absent or `NOTSET`
- `kernel_shape` explicit or inferred from weight shape
- `strides=[1,1]` by default, two positive values if present
- `pads=[0,0,0,0]` by default, four non-negative values if present
- `dilations=[1,1]` by default, two positive values if present
- optional bias is represented by normalized `Conv + Add`

Used `float32` ONNX initializers are embedded in MLIR as deterministic
`arith.constant dense<...>` values and are not runtime function arguments.

## Структура

**Публичные заголовки (`include/`):**
- `graph.hpp` — внутренний IR (`Graph`, `Node`, tensor/attr данные)
- `op_kind.hpp` — `OpKind` enum и маппинг из ONNX op string
- `op_traits.hpp` — arity и attribute trait queries по `OpKind`
- `type_info.hpp` — `DataT` enum и `TypeInfo<T>` (element type system)
- `graph_utils.hpp` — `BuildNodeContext`, утилиты атрибутов
- `frontend_constants.hpp` — naming constants для synthetic nodes (`kSyntheticAddSuffix` и др.)
- `shape_inference.hpp` — broadcast и shape inference utilities
- `dump_style.hpp` — Graphviz DOT style constants
- `onnx_importer.hpp` — API загрузки и импорта ONNX
- `graph_verifier.hpp` — API verifier и diagnostics report
- `dump_graph.hpp` — построение DOT-дампа
- `mlir_emitter.hpp` — Graph IR → textual MLIR
- `model_metadata.hpp` — EXEC metadata JSON sidecar

**Реализация (`src/`):**
- `onnx_importer.cpp` — ONNX → IR
- `op_kind.cpp` — `OpKindFromString` и сопутствующие функции
- `op_traits.cpp` — реализация op traits
- `type_info.cpp` — реализация type info
- `graph_utils.cpp` — реализация graph utilities
- `shape_inference.cpp` — реализация shape inference
- `graph_verifier.cpp` — проверки исполнимости графа
- `mlir_emitter.cpp` — production textual MLIR handoff
- `model_metadata.cpp` — deterministic metadata JSON
- `dump_graph.cpp` — Graphviz DOT
- `frontend_driver.cpp` — общий CLI driver
- `driver_defaults.hpp` — build-time default output paths
- `backend_runner.hpp` / `backend_runner.cpp` — backend integration для driver

**Прочее:**
- `onnx_lib/onnx.proto` — protobuf-схема ONNX
- `models/` — тестовые `.onnx` модели

## Требования

- CMake >= 3.20
- C++20 compiler
- Protobuf (`libprotobuf` и `protoc`)

## Сборка

Из корня репозитория:

```bash
cmake -S frontend -B frontend/build
cmake --build frontend/build --parallel
```

## Использование driver

```bash
./frontend/build/frontend_driver <model.onnx> [options]
```

Опции:
- `--verify` — запустить verifier и вывести диагностики
- `--verify-exec` — запустить executable-ready verifier
- `--dump` — включить генерацию `.dot` дампа с путем по умолчанию:
  `${CMAKE_CURRENT_BINARY_DIR}/dump/<model_name>.dot`
- `--dump=<output.dot>` — включить генерацию дампа и явно задать путь
- `--hash[=<output.hash>]` — записать deterministic graph hash
- `--emit-mlir[=<output.mlir>]` — записать production textual MLIR
- `--emit-metadata[=<output.json>]` — записать EXEC metadata JSON

Если путь не задан, используются build-директории:

- dump: `frontend/build/dump/<model_name>.dot`
- hash: `frontend/build/hash/<model_name>.hash`
- MLIR: `frontend/build/mlir/<model_name>.mlir`
- metadata: `frontend/build/metadata/<model_name>.json`

Примеры:

```bash
./frontend/build/frontend_driver frontend/models/single_relu.onnx --dump
./frontend/build/frontend_driver frontend/models/two_transposes.onnx --verify
./frontend/build/frontend_driver frontend/models/single_relu.onnx --verify --dump /tmp/single_relu.dot
./frontend/build/frontend_driver frontend/models/conv_2d_nchw_fchw.onnx --verify-exec
./frontend/build/frontend_driver frontend/models/conv_2d_nchw_fchw.onnx --emit-mlir=/tmp/conv.mlir
./frontend/build/frontend_driver frontend/models/conv_2d_nchw_fchw.onnx --emit-metadata=/tmp/conv.json
```

Коды возврата:
- `0` — успешно
- `1` — ошибка CLI/import/frontend/backend handoff/file output
- `2` — verifier вернул ошибки при `--verify` или `--verify-exec`

## Metadata JSON

Минимальный sidecar schema:

```json
{
  "model_entry": "tc_model",
  "input_count": 1,
  "inputs": [
    {
      "name": "input_0",
      "dtype": "float32",
      "shape": [1, 2, 8, 8],
      "layout": "NCHW",
      "byte_size": 512
    }
  ],
  "output_count": 1,
  "outputs": [
    {
      "name": "output_0",
      "dtype": "float32",
      "shape": [1, 2, 7, 7],
      "layout": "NCHW",
      "byte_size": 392
    }
  ]
}
```

JSON генерируется без внешней JSON-библиотеки и предназначен как простой
handoff contract для runtime/backend.

## Форматирование

Форматирование управляется корневым `.clang-format`.

```bash
clang-format -style=file -i frontend/include/*.hpp frontend/src/*.cpp frontend/src/*.hpp
```
