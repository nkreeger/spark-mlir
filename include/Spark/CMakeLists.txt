add_mlir_dialect(SparkOps spark)
add_mlir_doc(SparkDialect SparkDialect Spark/ -gen-dialect-doc)
add_mlir_doc(SparkOps SparkOps Spark/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS SparkPasses.td)
mlir_tablegen(SparkPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRSparkPassesIncGen)
