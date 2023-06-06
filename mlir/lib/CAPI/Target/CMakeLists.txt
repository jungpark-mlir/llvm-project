add_mlir_upstream_c_api_library(MLIRCAPITarget
  SPIRV.cpp

  LINK_LIBS PUBLIC
  MLIRSPIRVSerialization
)
