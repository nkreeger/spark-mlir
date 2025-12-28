// UNSUPPORTED: system-windows
// RUN: mlir-opt %s --load-dialect-plugin=%spark_libs/SparkPlugin%shlibext --pass-pipeline="builtin.module(spark-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @spark_types(%arg0: !spark.custom<"10">)
  func.func @spark_types(%arg0: !spark.custom<"10">) {
    return
  }
}
