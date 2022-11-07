set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_QUERY_LIBS)

if(MLIR_INCLUDE_TESTS)
  set(test_libs
    MLIRTestDialect
    )
endif()

set(LIBS
  ${conversion_libs}
  ${dialect_libs}
  ${test_libs}
  MLIRDialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRVDialect
  MLIRQueryLib
  MLIRSupport
  )

add_mlir_tool(mlir-query
  mlir-query.cpp

  DEPENDS
  ${LIBS}
  )
llvm_update_compile_flags(mlir-query)
target_link_libraries(mlir-query PRIVATE ${LIBS})

mlir_check_all_link_libraries(mlir-query)
