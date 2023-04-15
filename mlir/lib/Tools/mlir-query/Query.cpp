//===---- Query.cpp - mlir-query query -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Query.h"
#include "QuerySession.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Support/Debug.h"
using llvm::dbgs;

#define DEBUG_TYPE "mlir-query"
#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;

namespace mlir {
namespace query {

Query::~Query() {}

bool InvalidQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  OS << ErrStr << "\n";
  return false;
}

bool NoOpQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  return true;
}

bool HelpQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  OS << "Available commands:\n\n"
        "  match MATCHER, m MATCHER      "
        "Match the mlir against the given matcher.\n"
        "  set bind-root (true|false)    "
        "Set whether to bind the root matcher to \"root\".\n\n";
  return true;
}

// This could be done better but is not worth the variadic template trouble.
std::vector< matcher::DynTypedNode> getMatches(Operation *rootOp,
                                    const matcher::DynMatcher *matcher) {
  auto matchFinder = matcher::MatchFinder();
  return matchFinder.getMatches(rootOp, matcher);
}

bool MatchQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {

  Operation *rootOp = QS.Op;

  if (!matcher) {
    return false;
  }
  auto matches = getMatches(rootOp, matcher);

  unsigned MatchCount = 0;
  for (auto node : matches) {
    if (Operation* op =  node.get<Operation>()){
    auto opLoc = op->getLoc().cast<FileLineColLoc>();
    OS << "\nMatch #" << ++MatchCount << ":\n\n";
    OS << opLoc.getFilename().getValue() << ":" << opLoc.getLine() << ":"
       << opLoc.getColumn() << ": note: \"root\" binds here\n"
       << *op << "\n";
    //auto diag = mlir::emitError(opLoc, "Test message");
    }
  }
  OS << "\n" <<MatchCount << (MatchCount == 1 ? " match.\n\n" : " matches.\n\n");
  return true;
}

const QueryKind SetQueryKind<bool>::value;
const QueryKind SetQueryKind<OutputKind>::value;

} // namespace query
} // namespace mlir