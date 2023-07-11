//===---- QueryParser.cpp - mlir-query command parser ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QueryParser.h"
#include "Query.h"
#include "QuerySession.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

#include "llvm/Support/Debug.h"
using llvm::dbgs;

#define DEBUG_TYPE "mlir-query"
#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace llvm;
using namespace mlir;

namespace mlir {
namespace query {

// Lex any amount of whitespace followed by a "word" (any sequence of
// non-whitespace characters) from the start of region [Begin,End).  If no word
// is found before End, return StringRef().  Begin is adjusted to exclude the
// lexed region.
StringRef QueryParser::lexWord() {
  line = line.drop_while([](char c) {
    // Don't trim newlines.
    return StringRef(" \t\v\f\r").contains(c);
  });

  if (line.empty())
    // Even though the line is empty, it contains a pointer and
    // a (zero) length. The pointer is used in the LexOrCompleteWord
    // code completion.
    return line;

  StringRef word;
  if (line.front() == '#') {
    word = line.substr(0, 1);
  } else {
    word = line.take_until([](char c) {
      // Don't trim newlines.
      return StringRef(" \t\v\f\r").contains(c);
    });
  }

  line = line.drop_front(word.size());
  return word;
}

// This is the StringSwitch-alike used by lexOrCompleteWord below. See that
// function for details.
template <typename T>
struct QueryParser::LexOrCompleteWord {
  StringRef word;
  StringSwitch<T> stringSwitch;

  QueryParser *queryParser;
  // Set to the completion point offset in word, or StringRef::npos if
  // completion point not in word.
  size_t wordCompletionPos;

  // Lexes a word and stores it in word. Returns a LexOrCompleteword<T> object
  // that can be used like a llvm::StringSwitch<T>, but adds cases as possible
  // completions if the lexed word contains the completion point.
  LexOrCompleteWord(QueryParser *queryParser, StringRef &outWord)
      : word(queryParser->lexWord()), stringSwitch(word), queryParser(queryParser),
        wordCompletionPos(StringRef::npos) {
    outWord = word;
    if (queryParser->completionPos && queryParser->completionPos <= word.data() + word.size()) {
      if (queryParser->completionPos < word.data())
        wordCompletionPos = 0;
      else
        wordCompletionPos = queryParser->completionPos - word.data();
    }
  }

  LexOrCompleteWord &Case(llvm::StringLiteral caseStr, const T &value,
                          bool isCompletion = true) {

    if (wordCompletionPos == StringRef::npos)
      stringSwitch.Case(caseStr, value);
    else if (caseStr.size() != 0 && isCompletion &&
             wordCompletionPos <= caseStr.size() &&
             caseStr.substr(0, wordCompletionPos) ==
                 word.substr(0, wordCompletionPos)) {

      queryParser->completions.push_back(LineEditor::Completion(
          (caseStr.substr(wordCompletionPos) + " ").str(),
          std::string(caseStr)));
    }
    return *this;
  }

  T Default(T value) { return stringSwitch.Default(value); }
};

QueryRef QueryParser::endQuery(QueryRef Q) {
  StringRef extra = line;
  StringRef extraTrimmed = extra.drop_while(
      [](char c) { return StringRef(" \t\v\f\r").contains(c); });

  if ((!extraTrimmed.empty() && extraTrimmed[0] == '\n') ||
      (extraTrimmed.size() >= 2 && extraTrimmed[0] == '\r' &&
       extraTrimmed[1] == '\n'))
    Q->remainingContent = extra;
  else {
    StringRef trailingWord = lexWord();
    if (!trailingWord.empty() && trailingWord.front() == '#') {
      line = line.drop_until([](char c) { return c == '\n'; });
      line = line.drop_while([](char c) { return c == '\n'; });
      return endQuery(Q);
    }
    if (!trailingWord.empty()) {
      return new InvalidQuery("unexpected extra input: '" + extra + "'");
    }
  }
  return Q;
}

namespace {
enum MatcherKind {
  M_OpName,
  M_OpAttr,
  M_OpConst,
};
enum ParsedQueryKind {
  PQK_Invalid,
  PQK_Comment,
  PQK_NoOp,
  PQK_Help,
  PQK_Match,
};

QueryRef makeInvalidQueryFromDiagnostics(const matcher::Diagnostics &diag) {
  std::string ErrStr;
  llvm::raw_string_ostream OS(ErrStr);
  diag.printToStreamFull(OS);
  return new InvalidQuery(OS.str());
}
} // namespace

QueryRef QueryParser::completeMatcherExpression() {
  std::vector<matcher::MatcherCompletion> comps =
      matcher::Parser::completeExpression(line, completionPos - line.begin(),
                                          nullptr, &QS.namedValues);
  for (auto I = comps.begin(), E = comps.end(); I != E; ++I) {
    completions.push_back(LineEditor::Completion(I->typedText, I->matcherDecl));
  }
  return QueryRef();
}

QueryRef QueryParser::doParse() {

  StringRef commandStr;
  ParsedQueryKind qKind = LexOrCompleteWord<ParsedQueryKind>(this, commandStr)
                              .Case("", PQK_NoOp)
                              .Case("#", PQK_Comment, /*IsCompletion=*/false)
                              .Case("help", PQK_Help)
                              .Case("m", PQK_Match, /*IsCompletion=*/false)
                              .Case("match", PQK_Match)
                              .Default(PQK_Invalid);

  switch (qKind) {
  case PQK_Comment:
  case PQK_NoOp:
    line = line.drop_until([](char c) { return c == '\n'; });
    line = line.drop_while([](char c) { return c == '\n'; });
    if (line.empty())
      return new NoOpQuery;
    return doParse();

  case PQK_Help:
    return endQuery(new HelpQuery);

  case PQK_Match: {
    if (completionPos) {
      return completeMatcherExpression();
    }

    matcher::Diagnostics diag;
    auto matcherSource = line.ltrim();
    auto origMatcherSource = matcherSource;
    std::optional<matcher::DynMatcher> matcher =
        matcher::Parser::parseMatcherExpression(matcherSource, nullptr,
                                                &QS.namedValues, &diag);
    if (!matcher) {
      return makeInvalidQueryFromDiagnostics(diag);
    }
    auto actualSource = origMatcherSource.slice(0, origMatcherSource.size() -
                                                       matcherSource.size());
    auto *Q = new MatchQuery(actualSource, *matcher);
    Q->remainingContent = matcherSource;
    return Q;
  }

  case PQK_Invalid:
    return new InvalidQuery("unknown command: " + commandStr);
  }

  llvm_unreachable("Invalid query kind");
}

QueryRef QueryParser::parse(StringRef line, const QuerySession &QS) {
  return QueryParser(line, QS).doParse();
}

std::vector<LineEditor::Completion>
QueryParser::complete(StringRef line, size_t pos, const QuerySession &QS) {
  QueryParser queryParser(line, QS);
  queryParser.completionPos = line.data() + pos;

  queryParser.doParse();
  return queryParser.completions;
}

} // namespace query
} // namespace mlir