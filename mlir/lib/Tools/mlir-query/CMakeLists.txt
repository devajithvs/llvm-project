add_mlir_library(MLIRQuery
  Query.cpp
  QueryParser.cpp

  LINK_LIBS PRIVATE
  MLIRQueryMatcher
  )

add_subdirectory(Matcher)
add_subdirectory(Tool)
