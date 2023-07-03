//===--- MatcherParser.h - Matcher expression parser ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simple matcher expression parser.
//
// The parser understands matcher expressions of the form:
//   MatcherName(Arg0, Arg1, ..., ArgN)
// as well as simple types like strings.
// The parser does not know how to process the matchers. It delegates this task
// to a Sema object received as an argument.
//
// Grammar for the expressions supported:
// <Expression>        := <StringLiteral> | <MatcherExpression>
// <StringLiteral>     := "quoted string"
// <MatcherExpression> := <MatcherName>(<ArgumentList>)
// <MatcherName>       := [a-zA-Z]+
// <ArgumentList>      := <Expression> | <Expression>,<ArgumentList>
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_MATCHERPARSER_H
#define MLIR_TOOLS_MLIRQUERY_MATCHERPARSER_H

#include "MatcherDiagnostics.h"
#include "MatcherRegistry.h"
#include "MatcherVariantValue.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace query {
namespace matcher {

// Matcher expression parser.
class Parser {
public:
  // Interface to connect the parser with the registry and more.
  //
  // The parser uses the Sema instance passed into
  // parseMatcherExpression() to handle all matcher tokens. The simplest
  // processor implementation would simply call into the registry to create
  // the matchers.
  // However, a more complex processor might decide to intercept the matcher
  // creation and do some extra work. For example, it could apply some
  // transformation to the matcher by adding some id() nodes, or could detect
  // specific matcher nodes for more efficient lookup.
  class Sema {
  public:
    virtual ~Sema();

    // Process a matcher expression.
    // All the arguments passed here have already been processed.
    // Ctor is a matcher constructor looked up by lookupMatcherCtor.
    // Args is the argument list for the matcher.
    // Returns the matcher object constructed by the processor, or nullptr
    // if an error occurred. In that case, Error will contain a
    // description of the error.
    // The caller takes ownership of the Matcher object returned.
    virtual VariantMatcher actOnMatcherExpression(MatcherCtor Ctor, 
                                                  const SourceRange &NameRange,
                                                  bool ExtractFunction, StringRef FunctionName,
                                                  StringRef BindID,
                                                  ArrayRef<ParserValue> Args,
                                                  Diagnostics *Error) = 0;

    // Look up a matcher by name in the matcher name found by the parser.
    // NameRange is the location of the name in the matcher source, useful for
    // error reporting. Returns the matcher constructor, or
    // optional<MatcherCtor>() if an error occurred. In that case, Error will
    // contain a description of the error.
    virtual std::optional<MatcherCtor>
    lookupMatcherCtor(StringRef MatcherName, const SourceRange &NameRange,
                      Diagnostics *Error) = 0;
  };

  // Parse a matcher expression, creating matchers from the registry.

  // This overload creates matchers calling directly into the registry. If the
  // caller needs more control over how the matchers are created, then it can
  // use the overload below that takes a Sema.

  // MatcherCode is the matcher expression to parse.
  // Returns the matcher object constructed, or nullptr if an error occurred.
  // In that case, Error will contain a description of the error.
  // The caller takes ownership of the DynMatcher object returned.
  static std::optional<DynMatcher> parseMatcherExpression(StringRef MatcherCode, Diagnostics *Error);

  // Parse a matcher expression.

  // MatcherCode The matcher expression to parse.

  // S is the Sema instance that will help the parser
  // construct the matchers.
  // Returns the matcher object constructed by the processor, or nullptr
  // if an error occurred. In that case, Error will contain a
  // description of the error.
  // The caller takes ownership of the DynMatcher object returned.
  static std::optional<DynMatcher> parseMatcherExpression(StringRef MatcherCode, Sema *S, Diagnostics *Error);

  // Parse an expression, creating matchers from the registry.

  // Parses any expression supported by this parser. In general, the
  // parseMatcherExpression function is a better approach to get a matcher
  // object.
  static bool parseExpression(StringRef Code, VariantValue *Value,
                              Diagnostics *Error);

  // Parse an expression.

  // Parses any expression supported by this parser. In general, the
  // parseMatcherExpression function is a better approach to get a matcher
  // object.
  static bool parseExpression(StringRef Code, Sema *S, VariantValue *Value,
                              Diagnostics *Error);

private:
  class CodeTokenizer;
  struct TokenInfo;

  Parser(CodeTokenizer *Tokenizer, Sema *S, Diagnostics *Error);

  bool parseExpressionImpl(VariantValue *Value);
  bool parseMatcherExpressionImpl(VariantValue *Value);

  CodeTokenizer *const Tokenizer;
  Sema *const S;
  Diagnostics *Error;
};

} // namespace matcher
} // namespace query
} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERPARSER_H
