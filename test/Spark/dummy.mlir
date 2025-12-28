// RUN: spark-opt %s | spark-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = spark.foo %{{.*}} : i32
        %res = spark.foo %0 : i32
        return
    }

    // CHECK-LABEL: func @spark_types(%arg0: !spark.custom<"10">)
    func.func @spark_types(%arg0: !spark.custom<"10">) {
        return
    }
}
