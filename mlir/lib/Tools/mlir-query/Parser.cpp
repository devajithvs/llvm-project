//===--- Parser.cpp - Matcher expression parser -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Recursive parser implementation for the matcher expression grammar.
///
//===----------------------------------------------------------------------===//

#include <string>
#include <vector>

#include "Parser.h"
#include "Registry.h"
#include "llvm/ADT/Twine.h"

#include "llvm/Support/Debug.h"
using llvm::dbgs;

#define DEBUG_TYPE "mlir-query"

#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")
namespace mlir {
namespace query {
namespace matcher {

/// \brief Simple structure to hold information for one token from the parser.
struct Parser::TokenInfo {
  /// \brief Different possible tokens.
  enum TokenKind {
    TK_Eof,
    TK_OpenParen,
    TK_CloseParen,
    TK_Comma,
    TK_Literal,
    TK_Ident,
    TK_InvalidChar,
    TK_Error
  };

  TokenInfo() : Text(), Kind(TK_Eof), Value() {}

  StringRef Text;
  TokenKind Kind;
  VariantValue Value;
};

/// \brief Simple tokenizer for the parser.
class Parser::CodeTokenizer {
public:
  explicit CodeTokenizer(StringRef MatcherCode)
      : Code(MatcherCode), StartOfLine(MatcherCode) {
    NextToken = getNextToken();
  }

  /// \brief Returns but doesn't consume the next token.
  const TokenInfo &peekNextToken() const { return NextToken; }

  /// \brief Consumes and returns the next token.
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

    LLVM_DEBUG(DBGS() << "get Next token" << Code << "\n");
    if (Code.empty()) {
      Result.Kind = TokenInfo::TK_Eof;
      Result.Text = "";
      return Result;
    }
    LLVM_DEBUG(DBGS() << "get Next token - not empty"
                      << "\n");

    switch (Code[0]) {
    case ',':
      Result.Kind = TokenInfo::TK_Comma;
      Result.Text = Code.substr(0, 1);
      Code = Code.drop_front();
      break;
    case '(':
      LLVM_DEBUG(DBGS() << "openParen token"
                        << "\n");
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

    return Result;
  }

  /// \brief Consume a string literal.
  ///
  /// \c Code must be positioned at the start of the literal (the opening
  /// quote). Consumed until it finds the same closing quote character.
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

    Code = Code.drop_front(Code.size());
    Result->Kind = TokenInfo::TK_Error;
  }

  /// Consume all leading whitespace from \c Code.
  void consumeWhitespace() {
    Code = Code.drop_while([](char c) {
      // Don't trim newlines.
      return StringRef(" \t\v\f\r").contains(c);
    });
  }

  StringRef Code;
  StringRef StartOfLine;
  TokenInfo NextToken;
};

Parser::Sema::~Sema() {}

/// \brief Parse and validate a matcher expression.
/// \return \c true on success, in which case \c Value has the matcher parsed.
///   If the input is malformed, or some argument has an error, it
///   returns \c false.
bool Parser::parseMatcherExpressionImpl(VariantValue *Value) {

  LLVM_DEBUG(DBGS() << "parseMatcherExpressionImpl"
                    << "\n");
  const TokenInfo NameToken = Tokenizer->consumeNextToken();
  // TODO: Remove this assert
  LLVM_DEBUG(DBGS() << "parseMatcherExpressionImpl after nameToken"
                    << "\n");
  assert(NameToken.Kind == TokenInfo::TK_Ident);
  const TokenInfo OpenToken = Tokenizer->consumeNextToken();

  LLVM_DEBUG(DBGS() << "parseMatcherExpressionImpl after openToken"
                    << "\n");
  if (OpenToken.Kind != TokenInfo::TK_OpenParen) {
    return false;
  }

  LLVM_DEBUG(DBGS() << "parseMatcherExpressionImpl extracting args"
                    << "\n");
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
        return false;
      }
    }

    ParserValue ArgValue;
    ArgValue.Text = Tokenizer->peekNextToken().Text;
    if (!parseExpressionImpl(&ArgValue.Value)) {
      return false;
    }

    Args.push_back(ArgValue);
  }

  if (EndToken.Kind == TokenInfo::TK_Eof) {
    return false;
  }

  LLVM_DEBUG(DBGS() << "parseMatcherExpressionImpl after extracting args"
                    << "\n");
  LLVM_DEBUG(DBGS() << NameToken.Text << "parseMatcherExpressionImpl  args"
                    << "\n");
  // Merge the start and end infos.
  Matcher *Result = S->actOnMatcherExpression(NameToken.Text, Args);

  LLVM_DEBUG(DBGS() << NameToken.Text << "After call  "
                    << "\n");
  if (Result == NULL) {
    return false;
  }

  Value->takeMatcher(Result);
  LLVM_DEBUG(DBGS() << NameToken.Text << "takeMatcher call after  "
                    << "\n");
  return true;
}

/// \brief Parse an <Expresssion>
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
  case TokenInfo::TK_InvalidChar:
    const TokenInfo Token = Tokenizer->consumeNextToken();
    return false;
  }

  llvm_unreachable("Unknown token kind.");
}

Parser::Parser(CodeTokenizer *Tokenizer, Sema *S)
    : Tokenizer(Tokenizer), S(S) {}

class RegistrySema : public Parser::Sema {
public:
  virtual ~RegistrySema(){};
  Matcher *actOnMatcherExpression(StringRef MatcherName,
                                  ArrayRef<ParserValue> Args) override {
    LLVM_DEBUG(DBGS() << "Acto on matching expr run8"
                      << "\n");
    return Registry::constructMatcher(MatcherName, Args);
  }
};

bool Parser::parseExpression(StringRef Code, VariantValue *Value) {
  RegistrySema S;
  return parseExpression(Code, &S, Value);
}

bool Parser::parseExpression(StringRef Code, Sema *S, VariantValue *Value) {
  LLVM_DEBUG(DBGS() << "parseMatcherExpression code, S, Value"
                    << "\n");
  CodeTokenizer Tokenizer(Code);
  LLVM_DEBUG(DBGS() << "Tokenizer code, S, Value"
                    << "\n");
  return Parser(&Tokenizer, S).parseExpressionImpl(Value);
}

Matcher *Parser::parseMatcherExpression(StringRef Code) {
  RegistrySema S;
  LLVM_DEBUG(DBGS() << "parseMatcherExpression on matching expr run8"
                    << "\n");
  return parseMatcherExpression(Code, &S);
}

Matcher *Parser::parseMatcherExpression(StringRef Code, Parser::Sema *S) {
  VariantValue Value;
  LLVM_DEBUG(DBGS() << "parseMatcherExpression "
                    << "\n");
  if (!parseExpression(Code, S, &Value)) {

    return NULL;
  }
  LLVM_DEBUG(DBGS() << "parseMatcherExpression before.isMatcher"
                    << "\n");
  if (!Value.isMatcher()) {
    return NULL;
  }
  // TODO: Why clone?
  // return Value.getMatcher();
  LLVM_DEBUG(DBGS() << "parseMatcherExpression before.getMatcher"
                    << "\n");
  LLVM_DEBUG(DBGS() << Value.getMatcher().clone() << "\n");
  return Value.getMatcher().clone();
}

} // namespace matcher
} // namespace query
} // namespace mlir
