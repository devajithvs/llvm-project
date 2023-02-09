//===--- QueryParser.h - clang-query ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_QUERYPARSER_H
#define MLIR_TOOLS_MLIRQUERY_QUERYPARSER_H

#include "Query.h"

namespace mlir {
namespace query {

/// \brief Parse \p Line.
///
/// \return A reference to the parsed query object, which may be an
/// \c InvalidQuery if a parse error occurs.
QueryRef ParseQuery(llvm::StringRef Line);

} // namespace query
} // namespace mlir

#endif