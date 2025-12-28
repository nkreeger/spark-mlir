//===- SparkDialect.td - Spark dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SPARK_DIALECT
#define SPARK_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Spark dialect definition.
//===----------------------------------------------------------------------===//

def Spark_Dialect : Dialect {
    let name = "spark";
    let summary = "A spark out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::spark";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base spark operation definition.
//===----------------------------------------------------------------------===//

class Spark_Op<string mnemonic, list<Trait> traits = []> :
        Op<Spark_Dialect, mnemonic, traits>;

#endif // SPARK_DIALECT
