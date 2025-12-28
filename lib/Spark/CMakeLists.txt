add_mlir_dialect_library(MLIRSpark
        SparkTypes.cpp
        SparkDialect.cpp
        SparkOps.cpp
        SparkPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Spark

        DEPENDS
        MLIRSparkOpsIncGen
        MLIRSparkPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
