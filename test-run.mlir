// Simple test file for spark-run interpreter

// Test 1: Basic spark.foo operation (squares the input)
func.func @test_spark_op(%arg0: i32) -> i32 {
  %0 = spark.foo %arg0 : i32
  return %0 : i32
}

// Test 2: Multiple spark.foo operations with addition
func.func @test_multiple(%x: i32, %y: i32) -> i32 {
  %0 = spark.foo %x : i32
  %1 = spark.foo %y : i32
  %2 = arith.addi %0, %1 : i32
  return %2 : i32
}

// Test 3: Using constants
func.func @test_constant() -> i32 {
  %c5 = arith.constant 5 : i32
  %0 = spark.foo %c5 : i32
  return %0 : i32
}

// Test 4: Chained operations
func.func @test_chain(%arg0: i32) -> i32 {
  %0 = spark.foo %arg0 : i32
  %1 = spark.foo %0 : i32
  return %1 : i32
}
