add_llvm_library(SparkPlugin
        # BUILDTREE_ONLY is only for testing purposes
        MODULE BUILDTREE_ONLY
        spark-plugin.cpp

        DEPENDS
        MLIRSpark
        PLUGIN_TOOL
        mlir-opt

        LINK_LIBS
        MLIRSpark
        )
target_include_directories(
        SparkPlugin
        PRIVATE
        "${SPARK_BINARY_DIR}/include"
)
