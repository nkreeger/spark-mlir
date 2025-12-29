//===- spark-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Spark/SparkDialect.h"
#include "Spark/SparkPasses.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::spark::registerPasses();

  // Register a default Spark pass pipeline
  mlir::PassPipelineRegistration<>(
      "spark-default",
      "Default Spark transformation pipeline",
      [](mlir::OpPassManager &pm) {
        pm.addPass(mlir::spark::createSparkSwitchBarFoo());
      });

  // TODO: Register spark passes here.

  mlir::DialectRegistry registry;
  registry.insert<mlir::spark::SparkDialect,
                  mlir::arith::ArithDialect, mlir::func::FuncDialect>();
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Spark optimizer driver\n", registry));
}
