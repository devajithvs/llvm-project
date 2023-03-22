//===---- QueryParser.cpp - clang-query command parser --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "QueryParser.h"
#include "Query.h"
#include "QuerySession.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

#include "mlir/Conversion/ArithCommon/AttrToLLVMConverter.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MathToFuncs/MathToFuncs.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Matchers.h"
#include "Lexer.h"

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace llvm;
using namespace mlir;

#include "llvm/Support/Debug.h"
using llvm::dbgs;
//using mlir::detail::MatcherKind;

#define DEBUG_TYPE "mlir-query"

#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")

static bool isWhitespace(char C) {
  return C == ' ' || C == '\t' || C == '\r' || C == '\n';
}



mlir::detail::DynamicMatcher::~DynamicMatcher() {}
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
  if (Line.front() == '#')
    Word = Line.substr(0, 1);
  else
    Word = Line.take_until(isWhitespace);

  Line = Line.drop_front(Word.size());
  return Word;
}

// This is the StringSwitch-alike used by lexOrCompleteWord below. See that
// function for details.
template <typename T> struct QueryParser::LexOrCompleteWord {
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
    else if (CaseStr.size() != 0 && IsCompletion && WordCompletionPos <= CaseStr.size() &&
             CaseStr.substr(0, WordCompletionPos) ==
                 Word.substr(0, WordCompletionPos))
      P->Completions.push_back(LineEditor::Completion(
          (CaseStr.substr(WordCompletionPos) + " ").str(),
          std::string(CaseStr)));
    return *this;
  }

  T Default(T Value) { return Switch.Default(Value); }
};

static QueryRef ParseSetBool(bool QuerySession::*Var, StringRef ValStr) {
  unsigned Value = StringSwitch<unsigned>(ValStr)
                      .Case("false", 0)
                      .Case("true", 1)
                      .Default(~0u);
  if (Value == ~0u) {
    return new InvalidQuery("expected 'true' or 'false', got '" + ValStr + "'");
  }
  return new SetQuery<bool>(Var, Value);
}

static QueryRef ParseSetOutputKind(StringRef ValStr) {
  unsigned OutKind = StringSwitch<unsigned>(ValStr)
                         .Case("diag", OK_Diag)
                         .Case("print", OK_Print)
                         .Case("dump", OK_Dump)
                         .Default(~0u);
  if (OutKind == ~0u) {
    return new InvalidQuery("expected 'diag', 'print' or 'dump', got '" +
                            ValStr + "'");
  }
  return new SetQuery<OutputKind>(&QuerySession::OutKind, OutputKind(OutKind));
}


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


} // namespace


namespace {
enum MatcherKind {
  M_OpName,
  M_OpAttr,
};
enum ParsedQueryKind {
  PQK_Invalid,
  PQK_NoOp,
  PQK_Help,
  PQK_Match,
};



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
    // LLVM_DEBUG(DBGS() << Line << "\n");
    // auto MatcherSource = Line.ltrim();
    // auto OrigMatcherSource = MatcherSource;
    // LLVM_DEBUG(DBGS() << MatcherSource << "\n");
    // LLVM_DEBUG(DBGS() << "Working" << "\n");
    // // auto lexer = new Lexer(MatcherSource);
    
    // // auto curToken = lexer->lexToken();
    // // int i = 0;
    // // while(i<200) {
    // //   curToken = lexer->lexToken();
    // //   i++;
    // // }
    // detail::DynamicMatcherRef Matcher = parseMatcherExpression();
    // LLVM_DEBUG(DBGS() << "Working2" << "\n");
    // auto ActualSource = OrigMatcherSource.slice(0, OrigMatcherSource.size() -
    //                                                    MatcherSource.size());

    // return new MatchQuery(ActualSource, Matcher);

    LLVM_DEBUG(DBGS() << "Parsing matcher expression" << "\n");
    StringRef MatchExpr;
    LLVM_DEBUG(DBGS() << "New matcher source: "<<MatchExpr << "\n");
    MatcherKind MKind = LexOrCompleteWord<MatcherKind>(this, MatchExpr)
                                .Case("dialect.op1", M_OpName)
                                .Case("attributename", M_OpAttr)
                                .Default(M_OpName);
    LLVM_DEBUG(DBGS() << "New matcher source: "<<MatchExpr << "\n");
    if (MatchExpr.empty())
        return new InvalidQuery("expected variable name");
    switch (MKind) {
      case M_OpName: {
        auto M =  mlir::detail::name_op_matcher(MatchExpr);
        return new MatchQuery<mlir::detail::name_op_matcher>(MatchExpr, M);
      }
      case M_OpAttr: {
        auto M = mlir::detail::attr_op_matcher(MatchExpr);
        return new MatchQuery<mlir::detail::attr_op_matcher>(MatchExpr, M);
      }
    }
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