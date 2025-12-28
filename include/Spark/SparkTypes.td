//===- SparkTypes.td - Spark dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SPARK_TYPES
#define SPARK_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Spark/SparkDialect.td"

//===----------------------------------------------------------------------===//
// Spark type definitions
//===----------------------------------------------------------------------===//

class Spark_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Spark_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Spark_CustomType : Spark_Type<"Custom", "custom"> {
    let summary = "Spark custom type";
    let description = "Custom type in spark dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}

#endif // SPARK_TYPES
