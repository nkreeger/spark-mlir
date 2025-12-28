add_mlir_public_c_api_library(SparkCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
  MLIRSpark
)
