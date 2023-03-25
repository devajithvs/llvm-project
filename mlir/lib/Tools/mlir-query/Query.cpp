//===---- Query.cpp - mlir-query query -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MatchersInternal.h"
#include "Query.h"
#include "QuerySession.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/Matchers.h"

#include "llvm/Support/Debug.h"
using llvm::dbgs;

#define DEBUG_TYPE "mlir-query"

#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")

// using namespace clang::ast_matchers;
// using namespace clang::ast_matchers::dynamic;
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
        "Match the loaded ASTs against the given matcher.\n"
        "  set bind-root (true|false)    "
        "Set whether to bind the root matcher to \"root\".\n"
        "  set output (diag|print|dump)  "
        "Set whether to print bindings as diagnostics,\n"
        "                                "
        "AST pretty prints or AST dumps.\n\n";
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
template <typename T>
std::vector<Operation *> findMatches(Operation *rootOp, T &matcherFn) {
  auto matcher = new matcher::SingleMatcher<T>(matcherFn);
  auto matchFinder = matcher::MatchFinder();
  return matchFinder.getMatches(rootOp, matcher);
}

bool MatchQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
 LLVM_DEBUG(DBGS() << "Running run4" << "\n");
  MatcherKind MKind = M_OpName;
  if (MatchExpr.empty())
    return false;
  
  // TODO: PARSING

 LLVM_DEBUG(DBGS() << "Running run5" << "\n");
 LLVM_DEBUG(DBGS() << "Running run5" << MatchExpr << "\n");
  std::vector<Operation *>  matches;
  Operation *rootOp = QS.Op;

  switch (MKind) {
  case M_OpName: {
    // TODO: implement parser
    auto matcherFn = mlir::detail::name_op_matcher(MatchExpr);
    matches = findMatches(rootOp, matcherFn);
    break;
  }
  case M_OpAttr: {
    auto matcherFn = mlir::detail::attr_op_matcher(MatchExpr);
    matches = findMatches(rootOp, matcherFn);
    break;
  }
  case M_OpConst: {
    auto matcherFn = m_Constant();
    matches = findMatches(rootOp, matcherFn);
    break;
  }
  }
 

 LLVM_DEBUG(DBGS() << "Running run6" << "\n");
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