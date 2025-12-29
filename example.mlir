// Example MLIR file using the spark dialect

// Function that demonstrates spark.foo operation
func.func @test_spark_op(%arg0: i32) -> i32 {
  // The spark.foo operation takes an i32 input and returns an i32
  %0 = spark.foo %arg0 : i32
  return %0 : i32
}

// Function that demonstrates custom types
func.func @test_custom_types() -> !spark.custom<"hello"> {
  // Create a constant with spark custom type
  %0 = arith.constant 42 : i32
  %1 = spark.foo %0 : i32

  // This would be a custom type value (simplified example)
  %result = "test.dummy"() : () -> !spark.custom<"hello">
  return %result : !spark.custom<"hello">
}

// Function named "bar" that will be renamed by the spark-switch-bar-foo pass
func.func @bar() {
  return
}

// Another function to show multiple operations
func.func @complex_example(%x: i32, %y: i32) -> i32 {
  %0 = spark.foo %x : i32
  %1 = spark.foo %y : i32
  %2 = arith.addi %0, %1 : i32
  return %2 : i32
}
