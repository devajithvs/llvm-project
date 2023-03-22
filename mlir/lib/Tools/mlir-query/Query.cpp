//===---- Query.cpp - mlir-query query -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Query.h"
#include "QuerySession.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionInterfaces.h"

#include "mlir/IR/Matchers.h"

#include "llvm/Support/Debug.h"
using llvm::dbgs;

#define DEBUG_TYPE "mlir-query"

#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")

// using namespace clang::ast_matchers;
// using namespace clang::ast_matchers::dynamic;
using namespace mlir;

// // This could be done better but is not worth the variadic template trouble.
template <typename Matcher>
static std::vector<Operation*> getMatches(Operation* f, Matcher &matcher) {
  LLVM_DEBUG(DBGS() << "Running getMatches" << "\n");
  std::vector<Operation*> matches;
  f->walk([&matches, &matcher](Operation *op) {
    if (matcher.match(op)){
      matches.push_back(op);
    }
  });
  return matches;
}

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

template <typename T>
bool MatchQuery<T>::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  unsigned MatchCount = 0;

  LLVM_DEBUG(DBGS() << "Running run" << "\n");
  Operation *rootOp = QS.Op;
  LLVM_DEBUG(DBGS() << "Running run2" << "\n");
  // TODO: Parse matcher expression and create matcher.
  T matcher = Matcher;
  LLVM_DEBUG(DBGS() << "Running run3" << "\n");
  auto matches = getMatches(rootOp, matcher);
  LLVM_DEBUG(DBGS() << "Running run4" << "\n");
  for (auto op: matches){
    OS << "\nMatch #" << ++MatchCount << ":\n\n";
    // TODO: Get source location and filename
    OS << "testing: note: 'root' binds here\n" << *op << "\n\n";
  }
  LLVM_DEBUG(DBGS() << "Running run5" << "\n");

  OS << MatchCount << (MatchCount == 1 ? " match.\n" : " matches.\n");
  return true;
}


const QueryKind SetQueryKind<bool>::value;
const QueryKind SetQueryKind<OutputKind>::value;

} // namespace mlir
} // namespace clang