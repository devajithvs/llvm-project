//===--- QuerySession.h - mlir-query --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_QUERYSESSION_H
#define MLIR_TOOLS_MLIRQUERY_QUERYSESSION_H

// #include "MatcherParser.h"
#include "Query.h"
#include "mlir/Tools/ParseUtilities.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace query {

// Represents the state for a particular clang-query session.
class QuerySession {
public:
  QuerySession(Operation *rootOp,
               const std::shared_ptr<llvm::SourceMgr> &sourceMgr)
      : rootOp(rootOp), sourceMgr(sourceMgr), terminate(false) {}

  const std::shared_ptr<llvm::SourceMgr> &getSourceManager() {
    return sourceMgr;
  }

  Operation *rootOp;
  const std::shared_ptr<llvm::SourceMgr> sourceMgr;
  bool terminate;
  llvm::StringMap<matcher::VariantValue> namedValues;
};

} // namespace query
} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_QUERYSESSION_H
