//===--- QueryParser.h - mlir-query ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_QUERYPARSER_H
#define MLIR_TOOLS_MLIRQUERY_QUERYPARSER_H

#include "Matcher/Diagnostics.h"
#include "Matcher/Parser.h"
#include "Query.h"
#include "QuerySession.h"

#include "mlir/IR/Matchers.h"
#include "llvm/LineEditor/LineEditor.h"
#include <cstddef>

using namespace llvm;

namespace mlir {
namespace query {

class QuerySession;

class QueryParser {
public:
  // Parse Line as a query and return a QueryRef representing the query, which
  // may be an InvalidQuery.
  static QueryRef parse(StringRef line, const QuerySession &QS);

  // Compute a list of completions for Line assuming a cursor at
  // Pos is the characters past the start of Line, ordered from most
  // likely to least likely and returns a vector of completions for Line.
  static std::vector<llvm::LineEditor::Completion>
  complete(StringRef line, size_t pos, const QuerySession &QS);

private:
  QueryParser(StringRef line, const QuerySession &QS)
      : line(line), completionPos(nullptr), QS(QS) {}

  StringRef lexWord();

  template <typename T>
  struct LexOrCompleteWord;

  QueryRef parseSetBool(bool QuerySession::*Var);
  template <typename QueryType>
  QueryRef parseSetOutputKind();
  QueryRef completeMatcherExpression();

  QueryRef endQuery(QueryRef Q);

  // Parse [Begin, End) and returns a reference to the parsed query object,
  // which may be an InvalidQuery if a parse error occurs.
  QueryRef doParse();

  StringRef line;

  const char *completionPos;
  std::vector<llvm::LineEditor::Completion> completions;

  const QuerySession &QS;
};

} // namespace query
} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_QUERYPARSER_H
