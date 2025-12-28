// UNSUPPORTED: system-windows
// RUN: mlir-opt %s --load-pass-plugin=%spark_libs/SparkPlugin%shlibext --pass-pipeline="builtin.module(spark-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @abar()
  func.func @abar() {
    return
  }
}
