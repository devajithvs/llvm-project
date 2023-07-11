//===--- QuerySession.h - mlir-query --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

// Represents the state for a particular clang-query session.
class QuerySession {
public:
  QuerySession(Operation *rootOp,
               const std::shared_ptr<llvm::SourceMgr> &sourceMgr)
      : rootOp(rootOp), sourceMgr(sourceMgr), outKind(OK_Diag), bindRoot(true),
        terminate(false) {}

  const std::shared_ptr<llvm::SourceMgr> &getSourceManager() {
    return sourceMgr;
  }

  Operation *rootOp;
  const std::shared_ptr<llvm::SourceMgr> sourceMgr;
  OutputKind outKind;
  bool bindRoot;
  bool terminate;
  llvm::StringMap<matcher::VariantValue> namedValues;
};

} // namespace query
} // namespace mlir

#endif