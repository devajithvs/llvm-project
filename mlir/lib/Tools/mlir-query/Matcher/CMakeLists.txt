include_directories(${MLIR_MAIN_INCLUDE_DIR}/mlir/Tools/mlir-query/Matcher)

add_mlir_library(MLIRQueryMatcher
  Parser.cpp
  Registry.cpp
  VariantValue.cpp
  Diagnostics.cpp
  )
