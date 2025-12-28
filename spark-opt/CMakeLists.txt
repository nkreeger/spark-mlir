set(LIBS
    MLIRArithDialect
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRSpark
  )
add_llvm_executable(spark-opt spark-opt.cpp)

llvm_update_compile_flags(spark-opt)
target_link_libraries(spark-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(spark-opt)
