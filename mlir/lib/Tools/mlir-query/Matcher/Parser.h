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
//   matcherName(Arg0, Arg1, ..., ArgN)
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

#include "Diagnostics.h"
#include "Registry.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace query {
namespace matcher {

// Matcher expression parser.
class Parser {
public:
  // Interface to connect the parser with the registry and more. The parser uses
  // the Sema instance passed into parseMatcherExpression() to handle all
  // matcher tokens.
  class Sema {
  public:
    virtual ~Sema();

    // Process a matcher expression. The caller takes ownership of the Matcher
    // object returned.
    virtual VariantMatcher actOnMatcherExpression(MatcherCtor ctor,
                                                  SourceRange nameRange,
                                                  ArrayRef<ParserValue> args,
                                                  Diagnostics *error) = 0;

    // Look up a matcher by name in the matcher name found by the parser.
    // nameRange is the location of the name in the matcher source, useful for
    // error reporting. Returns the matcher constructor, or
    // optional<MatcherCtor>() if an error occurred. In that case, error will
    // contain a description of the error.
    virtual std::optional<MatcherCtor>
    lookupMatcherCtor(llvm::StringRef matcherName) = 0;

    virtual bool isBuilderMatcher(MatcherCtor) const = 0;

    virtual internal::MatcherDescriptorPtr
    buildMatcherCtor(MatcherCtor, SourceRange nameRange,
                     ArrayRef<ParserValue> args, Diagnostics *error) const = 0;

    // Compute the list of completion types for Context.
    virtual std::vector<ArgKind> getAcceptedCompletionTypes(
        llvm::ArrayRef<std::pair<MatcherCtor, unsigned>> Context);

    // Compute the list of completions that match any of acceptedTypes.
    virtual std::vector<MatcherCompletion>
    getMatcherCompletions(llvm::ArrayRef<ArgKind> acceptedTypes);
  };

  // Sema implementation that uses the matcher registry to process the tokens.
  class RegistrySema : public Parser::Sema {
  public:
    ~RegistrySema() override;

    std::optional<MatcherCtor>
    lookupMatcherCtor(llvm::StringRef matcherName) override;

    VariantMatcher actOnMatcherExpression(MatcherCtor ctor,
                                          SourceRange nameRange,
                                          ArrayRef<ParserValue> args,
                                          Diagnostics *error) override;

    bool isBuilderMatcher(MatcherCtor ctor) const override;

    std::vector<ArgKind> getAcceptedCompletionTypes(
        llvm::ArrayRef<std::pair<MatcherCtor, unsigned>> context) override;

    internal::MatcherDescriptorPtr
    buildMatcherCtor(MatcherCtor, SourceRange nameRange,
                     ArrayRef<ParserValue> args,
                     Diagnostics *error) const override;

    std::vector<MatcherCompletion>
    getMatcherCompletions(llvm::ArrayRef<ArgKind> acceptedTypes) override;
  };

  using NamedValueMap = llvm::StringMap<VariantValue>;

  // Parse a matcher expression. The caller takes ownership of the DynMatcher
  // object returned.
  static std::optional<DynMatcher>
  parseMatcherExpression(llvm::StringRef &matcherCode, Sema *sema,
                         const NamedValueMap *namedValues, Diagnostics *error);
  static std::optional<DynMatcher>
  parseMatcherExpression(llvm::StringRef &matcherCode, Sema *sema,
                         Diagnostics *error) {
    return parseMatcherExpression(matcherCode, sema, nullptr, error);
  }
  static std::optional<DynMatcher>
  parseMatcherExpression(llvm::StringRef &matcherCode, Diagnostics *error) {
    return parseMatcherExpression(matcherCode, nullptr, error);
  }

  /// Parse an expression. Parses any expression supported by this parser.
  static bool parseExpression(llvm::StringRef &code, Sema *sema,
                              const NamedValueMap *namedValues,
                              VariantValue *value, Diagnostics *error);

  static bool parseExpression(llvm::StringRef &code, Sema *sema,
                              VariantValue *value, Diagnostics *error) {
    return parseExpression(code, sema, nullptr, value, error);
  }
  static bool parseExpression(llvm::StringRef &code, VariantValue *value,
                              Diagnostics *error) {
    return parseExpression(code, nullptr, value, error);
  }

  /// Complete an expression at the given offset.
  static std::vector<MatcherCompletion>
  completeExpression(llvm::StringRef &code, unsigned completionOffset,
                     Sema *sema, const NamedValueMap *namedValues);
  static std::vector<MatcherCompletion>
  completeExpression(llvm::StringRef &code, unsigned completionOffset,
                     Sema *sema) {
    return completeExpression(code, completionOffset, sema, nullptr);
  }
  static std::vector<MatcherCompletion>
  completeExpression(llvm::StringRef &code, unsigned completionOffset) {
    return completeExpression(code, completionOffset, nullptr);
  }

private:
  class CodeTokenizer;
  struct ScopedContextEntry;
  struct TokenInfo;

  Parser(CodeTokenizer *tokenizer, Sema *sema, const NamedValueMap *namedValues,
         Diagnostics *error);

  bool parseID(std::string &id);

  bool parseExpressionImpl(VariantValue *value);

  bool buildAndValidateMatcher(std::vector<ParserValue> &args, MatcherCtor ctor,
                               const TokenInfo &nameToken,
                               const TokenInfo &openToken,
                               const TokenInfo &endToken, VariantValue *value);
  bool parseMatcherBuilder(MatcherCtor ctor, const TokenInfo &nameToken,
                           const TokenInfo &openToken, VariantValue *value);
  bool parseMatcherExpressionImpl(const TokenInfo &nameToken,
                                  const TokenInfo &openToken,
                                  std::optional<MatcherCtor> ctor,
                                  VariantValue *value);
  bool parseIdentifierPrefixImpl(VariantValue *value);

  void addCompletion(const TokenInfo &compToken,
                     const MatcherCompletion &completion);
  void addExpressionCompletions();

  std::vector<MatcherCompletion>
  getNamedValueCompletions(ArrayRef<ArgKind> acceptedTypes);

  CodeTokenizer *const tokenizer;
  Sema *const sema;
  const NamedValueMap *const namedValues;
  Diagnostics *const error;

  using ContextStackTy = std::vector<std::pair<MatcherCtor, unsigned>>;

  ContextStackTy contextStack;
  std::vector<MatcherCompletion> completions;
};

} // namespace matcher
} // namespace query
} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERPARSER_H
