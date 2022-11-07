set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_QUERY_LIBS)

add_mlir_tool(mlir-query
  mlir-query.cpp
  )
llvm_update_compile_flags(mlir-query)
target_link_libraries(mlir-query
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  ${test_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRVDialect
  MLIRQueryLib
  MLIRSupport
  )

mlir_check_link_libraries(mlir-query)
