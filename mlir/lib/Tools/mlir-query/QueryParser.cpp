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
  Line = Line.drop_while([](char c) {
    // Don't trim newlines.
    return StringRef(" \t\v\f\r").contains(c);
  });

  if (Line.empty())
    // Even though the Line is empty, it contains a pointer and
    // a (zero) length. The pointer is used in the LexOrCompleteWord
    // code completion.
    return Line;

  StringRef Word;
  if (Line.front() == '#') {
    Word = Line.substr(0, 1);
  } else {
    Word = Line.take_until([](char c) {
      // Don't trim newlines.
      return StringRef(" \t\v\f\r").contains(c);
    });
  }

  Line = Line.drop_front(Word.size());
  return Word;
}

// This is the StringSwitch-alike used by lexOrCompleteWord below. See that
// function for details.
template <typename T>
struct QueryParser::LexOrCompleteWord {
  StringRef Word;
  StringSwitch<T> Switch;

  QueryParser *P;
  // Set to the completion point offset in Word, or StringRef::npos if
  // completion point not in Word.
  size_t WordCompletionPos;

  // Lexes a word and stores it in Word. Returns a LexOrCompleteWord<T> object
  // that can be used like a llvm::StringSwitch<T>, but adds cases as possible
  // completions if the lexed word contains the completion point.
  LexOrCompleteWord(QueryParser *P, StringRef &OutWord)
      : Word(P->lexWord()), Switch(Word), P(P),
        WordCompletionPos(StringRef::npos) {
    OutWord = Word;
    if (P->CompletionPos && P->CompletionPos <= Word.data() + Word.size()) {
      if (P->CompletionPos < Word.data())
        WordCompletionPos = 0;
      else
        WordCompletionPos = P->CompletionPos - Word.data();
    }
  }

  LexOrCompleteWord &Case(llvm::StringLiteral CaseStr, const T &Value,
                          bool IsCompletion = true) {

    if (WordCompletionPos == StringRef::npos)
      Switch.Case(CaseStr, Value);
    else if (CaseStr.size() != 0 && IsCompletion &&
             WordCompletionPos <= CaseStr.size() &&
             CaseStr.substr(0, WordCompletionPos) ==
                 Word.substr(0, WordCompletionPos))
      P->Completions.push_back(LineEditor::Completion(
          (CaseStr.substr(WordCompletionPos) + " ").str(),
          std::string(CaseStr)));
    return *this;
  }

  T Default(T Value) { return Switch.Default(Value); }
};

QueryRef QueryParser::endQuery(QueryRef Q) {
  StringRef Extra = Line;
  StringRef ExtraTrimmed = Extra.drop_while(
      [](char c) { return StringRef(" \t\v\f\r").contains(c); });

  if ((!ExtraTrimmed.empty() && ExtraTrimmed[0] == '\n') ||
      (ExtraTrimmed.size() >= 2 && ExtraTrimmed[0] == '\r' &&
       ExtraTrimmed[1] == '\n'))
    Q->RemainingContent = Extra;
  else {
    StringRef TrailingWord = lexWord();
    if (!TrailingWord.empty() && TrailingWord.front() == '#') {
      Line = Line.drop_until([](char c) { return c == '\n'; });
      Line = Line.drop_while([](char c) { return c == '\n'; });
      return endQuery(Q);
    }
    if (!TrailingWord.empty()) {
      return new InvalidQuery("unexpected extra input: '" + Extra + "'");
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
  PQK_NoOp,
  PQK_Help,
  PQK_Match,
};

QueryRef makeInvalidQueryFromDiagnostics(const matcher::Diagnostics &Diag) {
  std::string ErrStr;
  llvm::raw_string_ostream OS(ErrStr);
  Diag.printToStreamFull(OS);
  return new InvalidQuery(OS.str());
}
} // namespace

QueryRef QueryParser::doParse() {

  StringRef CommandStr;
  ParsedQueryKind QKind = LexOrCompleteWord<ParsedQueryKind>(this, CommandStr)
                              .Case("", PQK_NoOp)
                              .Case("help", PQK_Help)
                              .Case("m", PQK_Match)
                              .Case("match", PQK_Match)
                              .Default(PQK_Invalid);

  switch (QKind) {
  case PQK_NoOp:
    Line = Line.drop_until([](char c) { return c == '\n'; });
    Line = Line.drop_while([](char c) { return c == '\n'; });
    if (Line.empty())
      return new NoOpQuery;
    return doParse();

  case PQK_Help:
    return endQuery(new HelpQuery);

  case PQK_Match: {
    matcher::Diagnostics Diag;
    auto MatchExpr = Line.ltrim();

    llvm::errs() << "getting matcher with parseMatcherExpression\n";
    auto matcher =
        matcher::Parser::parseMatcherExpression(MatchExpr, &Diag);
    llvm::errs() << "got matcher with parseMatcherExpression\n";

    if (!matcher.has_value()) {
      return makeInvalidQueryFromDiagnostics(Diag);
    }
    llvm::errs() << "pre MatchQuery\n";

    return new MatchQuery(matcher.value());
  }

  case PQK_Invalid:
    return new InvalidQuery("unknown command: " + CommandStr);
  }

  llvm_unreachable("Invalid query kind");
}

QueryRef QueryParser::parse(StringRef Line, const QuerySession &QS) {
  return QueryParser(Line, QS).doParse();
}

std::vector<LineEditor::Completion>
QueryParser::complete(StringRef Line, size_t Pos, const QuerySession &QS) {
  QueryParser P(Line, QS);
  P.CompletionPos = Line.data() + Pos;

  P.doParse();
  return P.Completions;
}

} // namespace query
} // namespace mlir