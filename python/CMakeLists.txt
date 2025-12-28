include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_spark`
# top level package (the API has been embedded in a relocatable way).
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(SparkPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SparkPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_spark"
  TD_FILE dialects/SparkOps.td
  SOURCES
    dialects/spark_nanobind.py
    _mlir_libs/_sparkDialectsNanobind/py.typed
  DIALECT_NAME spark)

declare_mlir_python_extension(SparkPythonSources.NanobindExtension
  MODULE_NAME _sparkDialectsNanobind
  ADD_TO_PARENT SparkPythonSources
  SOURCES
    SparkExtensionNanobind.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIArith
    MLIRCAPITransforms
    SparkCAPI
)


################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(SparkPythonCAPI
  INSTALL_COMPONENT SparkPythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    SparkPythonSources
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.builtin
)

################################################################################
# Instantiation of all Python modules
################################################################################

set(SparkPythonModules_ROOT_PREFIX "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}")

if(MLIR_PYTHON_STUBGEN_ENABLED)
  # Everything here is very tightly coupled. See the ample descriptions at the bottom of
  # mlir/python/CMakeLists.txt.

  # For a non-external projects build (e.g., installed distro) the type gen targets for the core _mlir module
  # need to be re-declared. On the contrary, for an external projects build, the MLIRPythonExtension.Core.type_stub_gen
  # target already exists and can just be added to DECLARED_SOURCES (see below).
  if(NOT EXTERNAL_PROJECT_BUILD)
    set(_core_type_stub_sources
      _mlir/__init__.pyi
      _mlir/ir.pyi
      _mlir/passmanager.pyi
      _mlir/rewrite.pyi
    )
    get_target_property(_core_extension_srcs MLIRPythonExtension.Core INTERFACE_SOURCES)
    mlir_generate_type_stubs(
      MODULE_NAME _mlir
      DEPENDS_TARGETS SparkPythonModules.extension._mlir.dso
      OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs/_mlir_libs"
      OUTPUTS "${_core_type_stub_sources}"
      DEPENDS_TARGET_SRC_DEPS "${_core_extension_srcs}"
      IMPORT_PATHS "${SparkPythonModules_ROOT_PREFIX}/_mlir_libs"
      VERBOSE
    )
    set(_mlir_typestub_gen_target "${NB_STUBGEN_CUSTOM_TARGET}")

    list(TRANSFORM _core_type_stub_sources PREPEND "_mlir_libs/")
    declare_mlir_python_sources(
      SparkPythonExtension.Core.type_stub_gen
      ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs"
      ADD_TO_PARENT SparkPythonSources
      SOURCES "${_core_type_stub_sources}"
    )
  endif()

  get_target_property(_spark_extension_srcs SparkPythonSources.NanobindExtension INTERFACE_SOURCES)
  mlir_generate_type_stubs(
    MODULE_NAME mlir_spark._mlir_libs._sparkDialectsNanobind
    DEPENDS_TARGETS
      SparkPythonModules.extension._mlir.dso
      SparkPythonModules.extension._sparkDialectsNanobind.dso
    OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs/_mlir_libs"
    OUTPUTS
      _sparkDialectsNanobind/__init__.pyi
      _sparkDialectsNanobind/spark.pyi
    DEPENDS_TARGET_SRC_DEPS "${_spark_extension_srcs}"
    IMPORT_PATHS "${SparkPythonModules_ROOT_PREFIX}/.."
  )
  set(_sparkDialectsNanobind_typestub_gen_target "${NB_STUBGEN_CUSTOM_TARGET}")

  declare_mlir_python_sources(
    SparkPythonSources.type_stub_gen
    ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs"
    ADD_TO_PARENT SparkPythonSources
    SOURCES
      _mlir_libs/_sparkDialectsNanobind/__init__.pyi
      _mlir_libs/_sparkDialectsNanobind/spark.pyi
  )
endif()

set(_declared_sources
  SparkPythonSources
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects.builtin
)
# For an external projects build, the MLIRPythonExtension.Core.type_stub_gen
# target already exists and can just be added to DECLARED_SOURCES.
if(EXTERNAL_PROJECT_BUILD AND MLIR_PYTHON_STUBGEN_ENABLED)
  list(APPEND _declared_sources MLIRPythonExtension.Core.type_stub_gen)
endif()

add_mlir_python_modules(SparkPythonModules
  ROOT_PREFIX "${SparkPythonModules_ROOT_PREFIX}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES "${_declared_sources}"
  COMMON_CAPI_LINK_LIBS
    SparkPythonCAPI
)

if(MLIR_PYTHON_STUBGEN_ENABLED)
  if(NOT EXTERNAL_PROJECT_BUILD)
    add_dependencies(SparkPythonModules "${_mlir_typestub_gen_target}")
  endif()
  add_dependencies(SparkPythonModules "${_sparkDialectsNanobind_typestub_gen_target}")
endif()
