//===- SparkTypes.h - Spark dialect types -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SPARK_SPARKTYPES_H
#define SPARK_SPARKTYPES_H

#include "mlir/IR/BuiltinTypes.h"

#define GET_TYPEDEF_CLASSES
#include "Spark/SparkOpsTypes.h.inc"

#endif // SPARK_SPARKTYPES_H
