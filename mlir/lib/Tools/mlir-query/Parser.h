//===--- Parser.h - Matcher expression parser -------------------*- C++ -*-===//
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

#ifndef MLIR_TOOLS_MLIRQUERY_MATCHERS_PARSER_H
#define MLIR_TOOLS_MLIRQUERY_MATCHERS_PARSER_H

#include "Diagnostics.h"
#include "VariantValue.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace query {
namespace matcher {

/// \brief Matcher expression parser.
class Parser {
public:
  /// \brief Interface to connect the parser with the registry and more.
  ///
  /// The parser uses the Sema instance passed into
  /// parseMatcherExpression() to handle all matcher tokens. The simplest
  /// processor implementation would simply call into the registry to create
  /// the matchers.
  /// However, a more complex processor might decide to intercept the matcher
  /// creation and do some extra work. For example, it could apply some
  /// transformation to the matcher by adding some id() nodes, or could detect
  /// specific matcher nodes for more efficient lookup.
  class Sema {
  public:
    virtual ~Sema();

    /// \brief Process a matcher expression.
    ///
    /// All the arguments passed here have already been processed.
    ///
    /// \param MatcherName The matcher name found by the parser.
    ///
    /// \param Args The argument list for the matcher.
    ///
    /// \return The matcher object constructed by the processor, or NULL
    ///   if an error occurred. In that case, \c Error will contain a
    ///   description of the error.
    ///   The caller takes ownership of the Matcher object returned.
    virtual DynMatcher *actOnMatcherExpression(StringRef MatcherName,
                                               const SourceRange &NameRange,
                                               bool ExtractFunction,
                                               ArrayRef<ParserValue> Args,
                                               Diagnostics *Error) = 0;
  };

  /// \brief Parse a matcher expression, creating matchers from the registry.
  ///
  /// This overload creates matchers calling directly into the registry. If the
  /// caller needs more control over how the matchers are created, then it can
  /// use the overload below that takes a Sema.
  ///
  /// \param MatcherCode The matcher expression to parse.
  ///
  /// \return The matcher object constructed, or NULL if an error occurred.
  //    In that case, \c Error will contain a description of the error.
  ///   The caller takes ownership of the DynMatcher object returned.
  static DynMatcher *parseMatcherExpression(StringRef MatcherCode,
                                            Diagnostics *Error);

  /// \brief Parse a matcher expression.
  ///
  /// \param MatcherCode The matcher expression to parse.
  ///
  /// \param S The Sema instance that will help the parser
  ///   construct the matchers.
  /// \return The matcher object constructed by the processor, or NULL
  ///   if an error occurred. In that case, \c Error will contain a
  ///   description of the error.
  ///   The caller takes ownership of the DynMatcher object returned.
  static DynMatcher *parseMatcherExpression(StringRef MatcherCode, Sema *S,
                                            Diagnostics *Error);

  /// \brief Parse an expression, creating matchers from the registry.
  ///
  /// Parses any expression supported by this parser. In general, the
  /// \c parseMatcherExpression function is a better approach to get a matcher
  /// object.
  static bool parseExpression(StringRef Code, VariantValue *Value,
                              Diagnostics *Error);

  /// \brief Parse an expression.
  ///
  /// Parses any expression supported by this parser. In general, the
  /// \c parseMatcherExpression function is a better approach to get a matcher
  /// object.
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

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERS_PARSER_H
