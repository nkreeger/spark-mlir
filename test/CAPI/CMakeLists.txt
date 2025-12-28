# Here we create a single aggregate shared library with the parts of the CAPI
# that we want to bundle together. Then we link a simple C executable
# against it to demonstrate that it does have the fully self contained
# core MLIR library and our own spark dialect.
add_mlir_aggregate(SparkCAPITestLib
  SHARED
  EMBED_LIBS
  MLIRCAPIIR
  MLIRCAPIArith
  SparkCAPI
)

add_llvm_executable(spark-capi-test
  spark-capi-test.c
)
llvm_update_compile_flags(spark-capi-test)
target_link_libraries(spark-capi-test
  PRIVATE SparkCAPITestLib)
