set(LLVM_LINK_COMPONENTS
  lineeditor
  Support
  )

add_mlir_library(MLIRQueryLib
  MlirQueryMain.cpp
  Query.cpp
  QueryParser.cpp
  Parser.cpp
  Registry.cpp
  VariantValue.cpp
  Diagnostics.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Tools/mlir-query

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  )
