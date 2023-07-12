include_directories( ${MLIR_MAIN_INCLUDE_DIR}/mlir/Tools/mlir-query)

set(LLVM_LINK_COMPONENTS
  lineeditor
  Support
  )

add_mlir_library(MLIRQueryLib
  MlirQueryMain.cpp

  LINK_LIBS PRIVATE
  MLIRIR
  MLIRParser
  MLIRQuery
  )
