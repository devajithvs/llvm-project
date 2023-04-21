//===- Parser.cpp - Matcher expression parser -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Recursive parser implementation for the matcher expression grammar.
//
//===----------------------------------------------------------------------===//

#include "Parser.h"
#include "Registry.h"
#include "llvm/ADT/Twine.h"
#include <string>
#include <vector>

#include "llvm/Support/Debug.h"
using llvm::dbgs;

#define DEBUG_TYPE "mlir-query"
#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
namespace query {
namespace matcher {

// Simple structure to hold information for one token from the parser.
struct Parser::TokenInfo {
  // Different possible tokens.
  enum TokenKind {
    TK_Eof,
    TK_OpenParen,
    TK_CloseParen,
    TK_Comma,
    TK_Period,
    TK_Literal,
    TK_Ident,
    TK_InvalidChar,
    TK_Error
  };

  // Some known identifiers.
  static const char *const ID_Extract;

  TokenInfo() : Text(), Kind(TK_Eof), Range(), Value() {}

  StringRef Text;
  TokenKind Kind;
  SourceRange Range;
  VariantValue Value;
};

const char *const Parser::TokenInfo::ID_Extract = "extract";

// Simple tokenizer for the parser.
class Parser::CodeTokenizer {
public:
  explicit CodeTokenizer(StringRef MatcherCode, Diagnostics *Error)
      : Code(MatcherCode), StartOfLine(MatcherCode), Line(1), Error(Error) {
    NextToken = getNextToken();
  }

  // Returns but doesn't consume the next token.
  const TokenInfo &peekNextToken() const { return NextToken; }

  // Consumes and returns the next token.
  TokenInfo consumeNextToken() {
    TokenInfo ThisToken = NextToken;
    NextToken = getNextToken();
    return ThisToken;
  }

  TokenInfo::TokenKind nextTokenKind() const { return NextToken.Kind; }

private:
  TokenInfo getNextToken() {
    consumeWhitespace();
    TokenInfo Result;
    Result.Range.Start = currentLocation();

    LLVM_DEBUG(DBGS() << "get Next token" << Code << "\n");
    if (Code.empty()) {
      Result.Kind = TokenInfo::TK_Eof;
      Result.Text = "";
      return Result;
    }

    switch (Code[0]) {
    case ',':
      Result.Kind = TokenInfo::TK_Comma;
      Result.Text = Code.substr(0, 1);
      Code = Code.drop_front();
      break;
    case '.':
      Result.Kind = TokenInfo::TK_Period;
      Result.Text = Code.substr(0, 1);
      Code = Code.drop_front();
      break;
    case '(':
      Result.Kind = TokenInfo::TK_OpenParen;
      Result.Text = Code.substr(0, 1);
      Code = Code.drop_front();
      break;
    case ')':
      Result.Kind = TokenInfo::TK_CloseParen;
      Result.Text = Code.substr(0, 1);
      Code = Code.drop_front();
      break;

    case '"':
    case '\'':
      // Parse a string literal.
      consumeStringLiteral(&Result);
      break;

    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      // Parse an unsigned and float literal.
      consumeNumberLiteral(&Result);
      break;

    default:
      if (isalnum(Code[0]) || Code[0] == '_') {
        // Parse an identifier
        size_t TokenLength = 1;
        while (TokenLength < Code.size() &&
               (isalnum(Code[TokenLength]) || Code[TokenLength] == '_'))
          ++TokenLength;
        Result.Kind = TokenInfo::TK_Ident;
        Result.Text = Code.substr(0, TokenLength);
        Code = Code.drop_front(TokenLength);
      } else {
        Result.Kind = TokenInfo::TK_InvalidChar;
        Result.Text = Code.substr(0, 1);
        Code = Code.drop_front(1);
      }
      break;
    }

    Result.Range.End = currentLocation();
    return Result;
  }

  // Consume an unsigned and float literal.
  void consumeNumberLiteral(TokenInfo *Result) {
    bool isFloatingLiteral = false;
    unsigned Length = 1;
    if (Code.size() > 1) {
      // Consume the 'x' or 'b' radix modifier, if present.
      switch (tolower(Code[1])) {
      case 'x':
      case 'b':
        Length = 2;
      }
    }
    while (Length < Code.size() && isdigit(Code[Length]))
      ++Length;

    // Try to recognize a floating point literal.
    while (Length < Code.size()) {
      char c = Code[Length];
      if (c == '-' || c == '+' || c == '.' || isdigit(c)) {
        isFloatingLiteral = true;
        Length++;
      } else {
        break;
      }
    }

    Result->Text = Code.substr(0, Length);
    Code = Code.drop_front(Length);

    if (isFloatingLiteral) {
      char *end;
      errno = 0;
      std::string Text = Result->Text.str();
      double doubleValue = strtod(Text.c_str(), &end);
      if (*end == 0 && errno == 0) {
        Result->Kind = TokenInfo::TK_Literal;
        Result->Value = doubleValue;
        return;
      }
    } else {
      unsigned Value;
      if (!Result->Text.getAsInteger(0, Value)) {
        Result->Kind = TokenInfo::TK_Literal;
        Result->Value = Value;
        LLVM_DEBUG(dbgs() << "\nThis is the unsigned value from parser: "
                          << Value);
        return;
      }
    }

    SourceRange Range;
    Range.Start = Result->Range.Start;
    Range.End = currentLocation();
    Error->addError(Range, Error->ET_ParserNumberError) << Result->Text;
    Result->Kind = TokenInfo::TK_Error;
  }

  // Consume a string literal.
  // Code must be positioned at the start of the literal (the opening
  // quote). Consumed until it finds the same closing quote character.
  void consumeStringLiteral(TokenInfo *Result) {
    bool InEscape = false;
    const char Marker = Code[0];
    for (size_t Length = 1, Size = Code.size(); Length != Size; ++Length) {
      if (InEscape) {
        InEscape = false;
        continue;
      }
      if (Code[Length] == '\\') {
        InEscape = true;
        continue;
      }
      if (Code[Length] == Marker) {
        Result->Kind = TokenInfo::TK_Literal;
        Result->Text = Code.substr(0, Length + 1);
        Result->Value = Code.substr(1, Length - 1);
        Code = Code.drop_front(Length + 1);
        return;
      }
    }

    StringRef ErrorText = Code;
    Code = Code.drop_front(Code.size());
    SourceRange Range;
    Range.Start = Result->Range.Start;
    Range.End = currentLocation();
    Error->addError(Range, Error->ET_ParserStringError) << ErrorText;
    Result->Kind = TokenInfo::TK_Error;
  }

  // Consume all leading whitespace from Code.
  void consumeWhitespace() {
    while (!Code.empty() && StringRef(" \t\v\f\r").contains(Code[0])) {
      if (Code[0] == '\n') {
        ++Line;
        StartOfLine = Code.drop_front();
      }
      Code = Code.drop_front();
    }
  }

  SourceLocation currentLocation() {
    SourceLocation Location;
    Location.Line = Line;
    Location.Column = Code.data() - StartOfLine.data() + 1;
    return Location;
  }

  StringRef Code;
  StringRef StartOfLine;
  unsigned Line;
  Diagnostics *Error;
  TokenInfo NextToken;
};

Parser::Sema::~Sema() {}

// Parse and validate a matcher expression.
// return true on success, in which case Value has the matcher parsed.
// If the input is malformed, or some argument has an error, it
// returns false.
bool Parser::parseMatcherExpressionImpl(VariantValue *Value) {

  LLVM_DEBUG(DBGS() << "parseMatcherExpressionImpl"
                    << "\n");
  const TokenInfo NameToken = Tokenizer->consumeNextToken();
  // TODO: Remove this assert
  assert(NameToken.Kind == TokenInfo::TK_Ident);
  const TokenInfo OpenToken = Tokenizer->consumeNextToken();

  if (OpenToken.Kind != TokenInfo::TK_OpenParen) {
    Error->addError(OpenToken.Range, Error->ET_ParserNoOpenParen)
        << OpenToken.Text;
    return false;
  }

  std::vector<ParserValue> Args;
  TokenInfo EndToken;
  while (Tokenizer->nextTokenKind() != TokenInfo::TK_Eof) {
    if (Tokenizer->nextTokenKind() == TokenInfo::TK_CloseParen) {
      // End of args.
      EndToken = Tokenizer->consumeNextToken();
      break;
    }
    if (Args.size() > 0) {
      // We must find a , token to continue.
      const TokenInfo CommaToken = Tokenizer->consumeNextToken();
      if (CommaToken.Kind != TokenInfo::TK_Comma) {
        Error->addError(CommaToken.Range, Error->ET_ParserNoComma)
            << CommaToken.Text;
        return false;
      }
    }

    ParserValue ArgValue;
    ArgValue.Text = Tokenizer->peekNextToken().Text;
    ArgValue.Range = Tokenizer->peekNextToken().Range;
    if (!parseExpressionImpl(&ArgValue.Value)) {
      // TODO: Add appropriate error.
      // Error->addError(NameToken.Range, Error->ET_ParserMatcherArgFailure)
      //    << (Args.size() + 1) << NameToken.Text;
      return false;
    }

    Args.push_back(ArgValue);
  }

  if (EndToken.Kind == TokenInfo::TK_Eof) {
    Error->addError(OpenToken.Range, Error->ET_ParserNoCloseParen);
    return false;
  }

  bool extractFunction = false;
  if (Tokenizer->peekNextToken().Kind == TokenInfo::TK_Period) {
    // Parse ".extract()"
    Tokenizer->consumeNextToken(); // consume the period.
    const TokenInfo ExtractToken = Tokenizer->consumeNextToken();
    const TokenInfo OpenToken = Tokenizer->consumeNextToken();
    const TokenInfo CloseToken = Tokenizer->consumeNextToken();

    // TODO: We could use different error codes for each/some to be more
    //       explicit about the syntax error.
    if (ExtractToken.Kind != TokenInfo::TK_Ident ||
        ExtractToken.Text != TokenInfo::ID_Extract) {
      Error->addError(ExtractToken.Range, Error->ET_ParserMalformedBindExpr);
      return false;
    }
    if (OpenToken.Kind != TokenInfo::TK_OpenParen) {
      Error->addError(OpenToken.Range, Error->ET_ParserMalformedBindExpr);
      return false;
    }
    if (CloseToken.Kind != TokenInfo::TK_CloseParen) {
      Error->addError(CloseToken.Range, Error->ET_ParserMalformedBindExpr);
      return false;
    }
    extractFunction = true;
  }

  // Merge the start and end infos.
  SourceRange MatcherRange = NameToken.Range;
  MatcherRange.End = EndToken.Range.End;
  DynMatcher *Result = S->actOnMatcherExpression(NameToken.Text, MatcherRange,
                                                 extractFunction, Args, Error);

  if (Result == nullptr) {
    // TODO: Add appropriate error.
    // Error->addError(NameToken.Range, Error->ET_ParserMatcherFailure)
    //    << NameToken.Text;
    return false;
  }

  Value->takeMatcher(Result);
  return true;
}

// Parse an <Expresssion>
bool Parser::parseExpressionImpl(VariantValue *Value) {
  switch (Tokenizer->nextTokenKind()) {
  case TokenInfo::TK_Literal:

    LLVM_DEBUG(DBGS() << "Tokenizer TK_literal"
                      << "\n");
    *Value = Tokenizer->consumeNextToken().Value;
    return true;

  case TokenInfo::TK_Ident:
    LLVM_DEBUG(DBGS() << "Tokenizer TK_ident"
                      << "\n");
    return parseMatcherExpressionImpl(Value);

  case TokenInfo::TK_Eof:
    Error->addError(Tokenizer->consumeNextToken().Range,
                    Error->ET_ParserNoCode);
    LLVM_DEBUG(DBGS() << "Tokenizer TK_EOF"
                      << "\n");
    return false;

  case TokenInfo::TK_Error:
    LLVM_DEBUG(DBGS() << "Tokenizer TK_ERROR"
                      << "\n");
    // This error was already reported by the tokenizer.
    return false;

  case TokenInfo::TK_OpenParen:
  case TokenInfo::TK_CloseParen:
  case TokenInfo::TK_Comma:
  case TokenInfo::TK_Period:
  case TokenInfo::TK_InvalidChar:
    const TokenInfo Token = Tokenizer->consumeNextToken();
    Error->addError(Token.Range, Error->ET_ParserInvalidToken) << Token.Text;
    return false;
  }

  llvm_unreachable("Unknown token kind.");
}

Parser::Parser(CodeTokenizer *Tokenizer, Sema *S, Diagnostics *Error)
    : Tokenizer(Tokenizer), S(S), Error(Error) {}
class RegistrySema : public Parser::Sema {
public:
  virtual ~RegistrySema(){};
  DynMatcher *actOnMatcherExpression(StringRef MatcherName,
                                     const SourceRange &NameRange,
                                     bool ExtractFunction,
                                     ArrayRef<ParserValue> Args,
                                     Diagnostics *Error) override {
    return Registry::constructMatcherWrapper(MatcherName, NameRange,
                                             ExtractFunction, Args, Error);
  }
};

bool Parser::parseExpression(StringRef Code, VariantValue *Value,
                             Diagnostics *Error) {
  RegistrySema S;
  return parseExpression(Code, &S, Value, Error);
}

bool Parser::parseExpression(StringRef Code, Sema *S, VariantValue *Value,
                             Diagnostics *Error) {
  CodeTokenizer Tokenizer(Code, Error);
  if (!Parser(&Tokenizer, S, Error).parseExpressionImpl(Value))
    return false;
  if (Tokenizer.peekNextToken().Kind != TokenInfo::TK_Eof) {
    Error->addError(Tokenizer.peekNextToken().Range,
                    Error->ET_ParserTrailingCode);
    return false;
  }
  return true;
}

DynMatcher *Parser::parseMatcherExpression(StringRef Code, Diagnostics *Error) {
  RegistrySema S;
  return parseMatcherExpression(Code, &S, Error);
}

DynMatcher *Parser::parseMatcherExpression(StringRef Code, Parser::Sema *S,
                                           Diagnostics *Error) {
  VariantValue Value;
  if (!parseExpression(Code, S, &Value, Error)) {

    return nullptr;
  }
  if (!Value.isMatcher()) {
    Error->addError(SourceRange(), Error->ET_ParserNotAMatcher);
    return nullptr;
  }
  // FIXME: remove clone
  return Value.getMatcher().clone();
}

} // namespace matcher
} // namespace query
} // namespace mlir
