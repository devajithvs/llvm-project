//===---- Query.cpp - clang-query query -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Query.h"
#include "QuerySession.h"
#include "llvm/Support/raw_ostream.h"

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
        "  match MATCHER, m MATCHER          "
        "Match the loaded ASTs against the given matcher.\n"
        "  let NAME MATCHER, l NAME MATCHER  "
        "Give a matcher expression a name, to be used later\n"
        "                                    "
        "as part of other expressions.\n"
        "  set bind-root (true|false)        "
        "Set whether to bind the root matcher to \"root\".\n"
        "  set print-matcher (true|false)    "
        "Set whether to print the current matcher,\n"
        "  set traversal <kind>              "
        "Set traversal kind of clang-query session. Available kinds are:\n"
        "    AsIs                            "
        "Print and match the AST as clang sees it.  This mode is the "
        "default.\n"
        "    IgnoreUnlessSpelledInSource     "
        "Omit AST nodes unless spelled in the source.\n"
        "  set output <feature>              "
        "Set whether to output only <feature> content.\n"
        "  enable output <feature>           "
        "Enable <feature> content non-exclusively.\n"
        "  disable output <feature>          "
        "Disable <feature> content non-exclusively.\n"
        "  quit, q                           "
        "Terminates the query session.\n\n"
        "Several commands accept a <feature> parameter. The available features "
        "are:\n\n"
        "  print                             "
        "Pretty-print bound nodes.\n"
        "  diag                              "
        "Diagnostic location for bound nodes.\n"
        "  detailed-ast                      "
        "Detailed AST output for bound nodes.\n"
        "  srcloc                            "
        "Source locations and ranges for bound nodes.\n"
        "  dump                              "
        "Detailed AST output for bound nodes (alias of detailed-ast).\n\n";
  return true;
}

bool QuitQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  QS.Terminate = true;
  return true;
}

bool MatchQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  unsigned MatchCount = 0;

  for (auto &AST : QS.ASTs) {
    MatchFinder Finder;
    std::vector<BoundNodes> Matches;
    DynTypedMatcher MaybeBoundMatcher = Matcher;
    if (QS.BindRoot) {
      llvm::Optional<DynTypedMatcher> M = Matcher.tryBind("root");
      if (M)
        MaybeBoundMatcher = *M;
    }
  }

  OS << MatchCount << (MatchCount == 1 ? " match.\n" : " matches.\n");
  return true;
}

bool LetQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  if (Value) {
    QS.NamedValues[Name] = Value;
  } else {
    QS.NamedValues.erase(Name);
  }
  return true;
}

} // namespace query
} // namespace clang
