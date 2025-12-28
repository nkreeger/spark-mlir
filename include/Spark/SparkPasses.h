//===- SparkPasses.h - Spark passes  ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef SPARK_SPARKPASSES_H
#define SPARK_SPARKPASSES_H

#include "Spark/SparkDialect.h"
#include "Spark/SparkOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace spark {
#define GEN_PASS_DECL
#include "Spark/SparkPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Spark/SparkPasses.h.inc"
} // namespace spark
} // namespace mlir

#endif
