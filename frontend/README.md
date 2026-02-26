# Tensor Compiler Frontend

Frontend отвечает за:
- импорт ONNX в внутренний IR;
- верификацию графа для исполнения (опционально, отдельным этапом);
- генерацию графического дампа (`.dot`).

## Текущий pipeline

`Load -> Import(IR) -> [Verify] -> [Dump(.dot)]`

- `Load/Import` — модуль импортера;
- `Verify` — отдельный verifier;
- `Dump` — Graphviz DOT-представление графа.

## Структура

- `include/graph.hpp` — внутренний IR (`Graph`, `Node`, tensor/attr данные)
- `include/onnx_importer.hpp` — API загрузки и импорта ONNX
- `include/graph_verifier.hpp` — API verifier и diagnostics report
- `include/dump_graph.hpp` — построение DOT-дампа
- `src/onnx_importer.cpp` — ONNX -> IR
- `src/graph_verifier.cpp` — проверки исполнимости графа
- `src/frontend_driver.cpp` — общий CLI driver
- `onnx_lib/onnx.proto` — protobuf-схема ONNX
- `models/` — тестовые `.onnx` модели

## Требования

- CMake >= 3.20
- C++17 compiler
- Protobuf (`libprotobuf` и `protoc`)

## Сборка

Из корня репозитория:

```bash
cmake -S frontend -B frontend/build
cmake --build frontend/build -j4
```

## Использование driver

```bash
./frontend/build/frontend_driver <model.onnx> [--verify] [--dump[=<output.dot>]]
```

Опции:
- `--verify` — запустить verifier и вывести диагностики
- `--dump` — включить генерацию `.dot` дампа с путем по умолчанию:
  `${CMAKE_CURRENT_BINARY_DIR}/dump/<model_name>.dot`
- `--dump=<output.dot>` — включить генерацию дампа и явно задать путь
- `--dump <output.dot>` — также поддерживается

Примеры:

```bash
./frontend/build/frontend_driver frontend/models/single_relu.onnx --dump
./frontend/build/frontend_driver frontend/models/two_transposes.onnx --verify
./frontend/build/frontend_driver frontend/models/single_relu.onnx --verify --dump /tmp/single_relu.dot
```

Коды возврата:
- `0` — успешно
- `1` — ошибка аргументов/импорта/записи дампа
- `2` — verifier вернул ошибки (только если указан `--verify`)

## Форматирование

Форматирование управляется корневым `.clang-format`.

```bash
clang-format -style=file -i frontend/include/*.hpp frontend/src/*.cpp
```
