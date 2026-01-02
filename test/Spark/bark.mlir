// RUN: spark-opt %s | spark-opt | FileCheck %s

module {
    // CHECK-LABEL: func @test_bark()
    func.func @test_bark() {
        %0 = arith.constant 5 : i32
        // CHECK: %{{.*}} = spark.bark %{{.*}} : i32
        %res = spark.bark %0 : i32
        return
    }

    // CHECK-LABEL: func @test_bark_chain()
    func.func @test_bark_chain() {
        %0 = arith.constant 7 : i32
        // CHECK: %{{.*}} = spark.bark %{{.*}} : i32
        %1 = spark.bark %0 : i32
        // CHECK: %{{.*}} = spark.bark %{{.*}} : i32
        %2 = spark.bark %1 : i32
        return
    }

    // CHECK-LABEL: func @test_bark_with_foo()
    func.func @test_bark_with_foo() {
        %0 = arith.constant 3 : i32
        // CHECK: %{{.*}} = spark.bark %{{.*}} : i32
        %1 = spark.bark %0 : i32
        // CHECK: %{{.*}} = spark.foo %{{.*}} : i32
        %2 = spark.foo %1 : i32
        return
    }

    // CHECK-LABEL: func @test_bark_with_arg(%arg0: i32)
    func.func @test_bark_with_arg(%arg0: i32) -> i32 {
        // CHECK: %{{.*}} = spark.bark %{{.*}} : i32
        %0 = spark.bark %arg0 : i32
        return %0 : i32
    }
}
