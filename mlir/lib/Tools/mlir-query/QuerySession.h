//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_QUERYSESSION_H
#define MLIR_TOOLS_MLIRQUERY_QUERYSESSION_H

#include "llvm/ADT/ArrayRef.h"
#include "Query.h"

namespace mlir {
namespace query {

/// Represents the state for a particular clang-query session.
class QuerySession {
public:
  QuerySession()
      : OutKind(OK_Diag), BindRoot(true) {}

  OutputKind OutKind;
  bool BindRoot;
  bool Terminate;
};

} // namespace query
} // namespace mlir

#endif