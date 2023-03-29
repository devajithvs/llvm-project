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

namespace {
enum MatcherKind {
  M_OpName,
  M_OpAttr,
  M_OpConst,
};
} // namespace

// This could be done better but is not worth the variadic template trouble.
std::vector<Operation *> getMatches(Operation *rootOp,
                                    const matcher::Matcher *matcher) {
  auto matchFinder = matcher::MatchFinder();
  return matchFinder.getMatches(rootOp, matcher);
}

bool MatchQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {

  std::vector<Operation *> matches;
  Operation *rootOp = QS.Op;

  if (!matcher) {
    return false;
  }
  matches = getMatches(rootOp, matcher);

  unsigned MatchCount = 0;
  for (auto op : matches) {
    OS << "\nMatch #" << ++MatchCount << ":\n\n";
    // TODO: Get source location and filename
    OS << "testing: note: 'root' binds here\n" << *op << "\n\n";
  }
  OS << MatchCount << (MatchCount == 1 ? " match.\n" : " matches.\n");
  return true;
}

const QueryKind SetQueryKind<bool>::value;
const QueryKind SetQueryKind<OutputKind>::value;

} // namespace query
} // namespace mlir