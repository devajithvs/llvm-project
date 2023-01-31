set(LLVM_LINK_COMPONENTS
  lineeditor
  support
  FrontendOpenMP
  )

add_mlir_library(mlirQuery
  Query.cpp
  QueryParser.cpp

  DEPENDS
  omp_gen
  )

target_link_libraries(mlirQuery
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangDynamicASTMatchers
  clangFrontend
  clangTooling
  clangSerialization
  )

add_subdirectory(tool)
