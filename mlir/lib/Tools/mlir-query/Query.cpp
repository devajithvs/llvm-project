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
// #include "clang/ASTMatchers/ASTMatchFinder.h"
// #include "clang/Frontend/ASTUnit.h"
// #include "clang/Frontend/TextDiagnostic.h"
#include "llvm/Support/raw_ostream.h"

// using namespace clang::ast_matchers;
// using namespace clang::ast_matchers::dynamic;

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

bool MatchQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  unsigned MatchCount = 0;

  OS << MatchCount << (MatchCount == 1 ? " match.\n" : " matches.\n");
  return true;
}

const QueryKind SetQueryKind<bool>::value;
const QueryKind SetQueryKind<OutputKind>::value;

} // namespace mlir
} // namespace clang