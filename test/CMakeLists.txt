llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(SPARK_TEST_DEPENDS
        FileCheck count not
        spark-capi-test
        spark-opt
        spark-translate
        )
if(NOT WIN32)
  list(APPEND SPARK_TEST_DEPENDS SparkPlugin)
endif()
if(MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND SPARK_TEST_DEPENDS SparkPythonModules)
endif()

add_lit_testsuite(check-spark "Running the spark regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${SPARK_TEST_DEPENDS}
        )
set_target_properties(check-spark PROPERTIES FOLDER "Tests")

add_lit_testsuites(SPARK ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${SPARK_TEST_DEPENDS})

add_subdirectory(CAPI)
