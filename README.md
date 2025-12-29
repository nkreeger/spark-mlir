# An out-of-tree MLIR dialect

This is an example of an out-of-tree [MLIR](https://mlir.llvm.org/) dialect along with a spark `opt`-like tool to operate on that dialect.

## Building - Component Build

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-spark
```
To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.

## Building - Monolithic Build

This setup assumes that you build the project as part of a monolithic LLVM build via the `LLVM_EXTERNAL_PROJECTS` mechanism.
To build LLVM, MLIR, the example and launch the tests run
```sh
mkdir build && cd build
cmake -G Ninja `$LLVM_SRC_DIR/llvm` \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS=spark-mlir -DLLVM_EXTERNAL_SPARK_MLIR_SOURCE_DIR=../
cmake --build . --target check-spark
```
Here, `$LLVM_SRC_DIR` needs to point to the root of the monorepo.

## Running the Spark Interpreter

The `spark-run` tool is a simple interpreter that executes MLIR files containing Spark dialect operations. The runtime implements `spark.foo` as a squaring operation (returns `input * input`).

### Basic Usage

```sh
# Run a simple test (computes 5²)
build/bin/spark-run test-run.mlir --function=test_spark_op --args=5
# Output: Final result: 25

# Run with multiple arguments (computes 3² + 4²)
build/bin/spark-run test-run.mlir --function=test_multiple --args=3 --args=4
# Output: Final result: 25

# Run a chained operation (computes (2²)²)
build/bin/spark-run test-run.mlir --function=test_chain --args=2
# Output: Final result: 16

# Run with a constant value (computes 5²)
build/bin/spark-run test-run.mlir --function=test_constant
# Output: Final result: 25
```

### Using example.mlir

The `example.mlir` file contains additional test functions. Since it includes unregistered dialects, use the `--allow-unregistered-dialect` flag:

```sh
# Run a simple spark.foo operation
build/bin/spark-run example.mlir --function=test_spark_op --args=10 --allow-unregistered-dialect
# Output: Final result: 100

# Run complex example with multiple operations
build/bin/spark-run example.mlir --function=complex_example --args=7 --args=2 --allow-unregistered-dialect
# Output: Final result: 53 (7² + 2² = 49 + 4)
```

### Supported Operations

The interpreter currently supports:
- `spark.foo` - Squares the input value
- `arith.constant` - Integer constants
- `arith.addi` - Integer addition
- `func.return` - Return values from functions

The interpreter prints an execution trace showing each operation and its result.
