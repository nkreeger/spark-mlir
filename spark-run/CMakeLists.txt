set(LIBS
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRSpark
  )
add_llvm_executable(spark-run spark-run.cpp)

llvm_update_compile_flags(spark-run)
target_link_libraries(spark-run PRIVATE ${LIBS})

mlir_check_all_link_libraries(spark-run)
