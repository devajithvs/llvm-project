include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

set(LLVM_LINK_COMPONENTS
  lineeditor
  Support
  )

add_mlir_library(MLIRQueryLib
  MlirQueryMain.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Tools/mlir-query

  LINK_LIBS PRIVATE
  MLIRIR
  MLIRParser
  MLIRQuery
  )
