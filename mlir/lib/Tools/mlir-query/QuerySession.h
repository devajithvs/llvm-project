//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_QUERYSESSION_H
#define MLIR_TOOLS_MLIRQUERY_QUERYSESSION_H

#include "Query.h"
#include "mlir/Tools/ParseUtilities.h"
#include "llvm/ADT/ArrayRef.h"

using namespace mlir;
namespace mlir {
namespace query {

/// Represents the state for a particular clang-query session.
class QuerySession {
public:
  QuerySession(Operation *Op) : Op(Op), OutKind(OK_Diag), BindRoot(true) {}

  Operation *Op;

  OutputKind OutKind;
  bool BindRoot;
  bool Terminate;
};

} // namespace query
} // namespace mlir

#endif