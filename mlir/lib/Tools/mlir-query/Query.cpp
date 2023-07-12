//===---- Query.cpp - mlir-query query ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Query.h"
#include "QuerySession.h"
#include "llvm/Support/raw_ostream.h"
#include <bits/stdc++.h>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/Support/Debug.h"
using llvm::dbgs;

#define DEBUG_TYPE "mlir-query"
#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
namespace query {

Query::~Query() {}

bool InvalidQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  OS << errStr << "\n";
  return false;
}

bool NoOpQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  return true;
}

bool HelpQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  OS << "Available commands:\n\n"
        "  match MATCHER, m MATCHER      "
        "Match the mlir against the given matcher.\n\n";
  return true;
}

std::vector<Operation *> getMatches(Operation *rootOp,
                                    const matcher::DynMatcher &matcher) {
  auto matchFinder = query::matcher::MatchFinder();
  return matchFinder.getMatches(rootOp, matcher);
}

bool MatchQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  Operation *rootOp = QS.rootOp;
  auto matches = getMatches(rootOp, matcher);

  MLIRContext *context = rootOp->getContext();
  context->printOpOnDiagnostic(false);
  SourceMgrDiagnosticHandler sourceMgrHandler(*QS.sourceMgr, context);

  unsigned matchCount = 0;
  OS << "\n";
  for (Operation *op : matches) {
    OS << "Match #" << ++matchCount << ":\n\n";
    // Placeholder "root" binding for the initial draft.
    op->emitRemark("\"root\" binds here");
  }
  OS << matchCount << (matchCount == 1 ? " match.\n\n" : " matches.\n\n");

  return true;
}

} // namespace query
} // namespace mlir