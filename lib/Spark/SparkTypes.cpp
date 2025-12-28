//===- SparkTypes.cpp - Spark dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Spark/SparkTypes.h"

#include "Spark/SparkDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::spark;

#define GET_TYPEDEF_CLASSES
#include "Spark/SparkOpsTypes.cpp.inc"

void SparkDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Spark/SparkOpsTypes.cpp.inc"
      >();
}
