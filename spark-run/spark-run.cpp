//===- spark-run.cpp - Simple Spark MLIR interpreter --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simple interpreter for MLIR containing Spark dialect operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Spark/SparkDialect.h"
#include "Spark/SparkOps.h"

#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace llvm;

static cl::opt<std::string> inputFilename(cl::Positional,
                                           cl::desc("<input file>"),
                                           cl::init("-"));

static cl::opt<std::string> functionName("function",
                                          cl::desc("Function to execute"),
                                          cl::init("test_spark_op"));

static cl::list<int32_t> args("args",
                               cl::desc("Integer arguments to pass"),
                               cl::ZeroOrMore);

static cl::opt<bool> allowUnregisteredDialects(
    "allow-unregistered-dialect",
    cl::desc("Allow unregistered dialects in the input"),
    cl::init(false));

namespace {
class SimpleInterpreter {
public:
  SimpleInterpreter() = default;

  int32_t execute(func::FuncOp func, ArrayRef<int32_t> arguments) {
    // Initialize function arguments
    auto funcArgs = func.getArguments();
    if (funcArgs.size() != arguments.size()) {
      llvm::errs() << "Error: Function expects " << funcArgs.size()
                   << " arguments but " << arguments.size() << " provided\n";
      return -1;
    }

    for (size_t i = 0; i < arguments.size(); ++i) {
      valueMap[funcArgs[i]] = arguments[i];
    }

    // Execute the function body
    for (auto &block : func.getBody()) {
      for (auto &op : block) {
        if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
          handleConstant(constantOp);
        } else if (auto fooOp = dyn_cast<spark::FooOp>(op)) {
          handleFoo(fooOp);
        } else if (auto barkOp = dyn_cast<spark::BarkOp>(op)) {
          handleBark(barkOp);
        } else if (auto addOp = dyn_cast<arith::AddIOp>(op)) {
          handleAdd(addOp);
        } else if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
          return handleReturn(returnOp);
        } else {
          llvm::errs() << "Warning: Unhandled operation: "
                       << op.getName().getStringRef() << "\n";
        }
      }
    }

    llvm::errs() << "Error: Function did not return\n";
    return -1;
  }

private:
  void handleConstant(arith::ConstantOp op) {
    if (auto intAttr = dyn_cast<IntegerAttr>(op.getValue())) {
      int32_t value = intAttr.getInt();
      valueMap[op.getResult()] = value;
      llvm::outs() << "  constant: " << value << "\n";
    }
  }

  void handleFoo(spark::FooOp op) {
    int32_t input = valueMap[op.getInput()];
    // Implement the runtime behavior: square the input
    int32_t result = input * input;
    valueMap[op.getRes()] = result;
    llvm::outs() << "  spark.foo(" << input << ") = " << result << "\n";
  }

  void handleBark(spark::BarkOp op) {
    int32_t input = valueMap[op.getInput()];
    // Implement the runtime behavior: negate the input
    int32_t result = -input;
    valueMap[op.getRes()] = result;
    llvm::outs() << "  spark.bark(" << input << ") = " << result << "\n";
  }

  void handleAdd(arith::AddIOp op) {
    int32_t lhs = valueMap[op.getLhs()];
    int32_t rhs = valueMap[op.getRhs()];
    int32_t result = lhs + rhs;
    valueMap[op.getResult()] = result;
    llvm::outs() << "  add: " << lhs << " + " << rhs << " = " << result
                 << "\n";
  }

  int32_t handleReturn(func::ReturnOp op) {
    if (op.getNumOperands() == 0) {
      return 0;
    }
    int32_t result = valueMap[op.getOperand(0)];
    llvm::outs() << "  return: " << result << "\n";
    return result;
  }

  llvm::DenseMap<Value, int32_t> valueMap;
};
} // namespace

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "Spark MLIR interpreter\n");

  // Set up the MLIR context and load dialects
  MLIRContext context;
  context.getOrLoadDialect<spark::SparkDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<func::FuncDialect>();

  if (allowUnregisteredDialects)
    context.allowUnregisteredDialects();

  // Parse the input MLIR file
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Failed to parse input file\n";
    return 1;
  }

  // Find the function to execute
  func::FuncOp targetFunc;
  module->walk([&](func::FuncOp func) {
    if (func.getSymName() == functionName) {
      targetFunc = func;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (!targetFunc) {
    llvm::errs() << "Function '" << functionName << "' not found\n";
    return 1;
  }

  llvm::outs() << "Executing function: @" << functionName << "\n";
  llvm::outs() << "Arguments: [";
  for (size_t i = 0; i < args.size(); ++i) {
    if (i > 0)
      llvm::outs() << ", ";
    llvm::outs() << args[i];
  }
  llvm::outs() << "]\n";
  llvm::outs() << "---\n";

  // Execute the function
  SimpleInterpreter interpreter;
  int32_t result = interpreter.execute(targetFunc, args);

  llvm::outs() << "---\n";
  llvm::outs() << "Final result: " << result << "\n";

  return 0;
}
