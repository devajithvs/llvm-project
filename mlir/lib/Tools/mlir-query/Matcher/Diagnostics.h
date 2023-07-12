//===--- MatcherDiagnostics.h - Helper class for error diagnostics --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Diagnostics class to manage error messages.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_MATCHERDIAGNOSTICS_H
#define MLIR_TOOLS_MLIRQUERY_MATCHERDIAGNOSTICS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

namespace mlir {
namespace query {
namespace matcher {

struct SourceLocation {
  SourceLocation() : line(), column() {}
  unsigned line;
  unsigned column;
};

struct SourceRange {
  SourceLocation start;
  SourceLocation end;
};

// Helper class to manage error messages.
class Diagnostics {
public:
  // Parser context types.
  enum ContextType { CT_MatcherArg, CT_MatcherConstruct };

  // All errors from the system.
  enum ErrorType {
    ET_None,

    ET_RegistryMatcherNotFound,
    ET_RegistryWrongArgCount,
    ET_RegistryWrongArgType,
    ET_RegistryNotBindable,
    ET_RegistryValueNotFound,
    ET_RegistryUnknownEnumWithReplace,
    ET_RegistryMatcherNoWithSupport,

    ET_ParserStringError,
    ET_ParserNoOpenParen,
    ET_ParserNoCloseParen,
    ET_ParserNoComma,
    ET_ParserNoCode,
    ET_ParserNotAMatcher,
    ET_ParserInvalidToken,
    ET_ParserMalformedExprNoOpenParen,
    ET_ParserMalformedExprNoIdentifier,
    ET_ParserMalformedExprNoCloseParen,
    ET_ParserTrailingCode,
    ET_ParserNumberError,
    ET_ParserOverloadedType,
    ET_ParserMalformedChainedExpr,
    ET_ParserFailedToBuildMatcher
  };

  // Helper stream class.
  class ArgStream {
  public:
    ArgStream(std::vector<std::string> *out) : out(out) {}
    template <class T>
    ArgStream &operator<<(const T &arg) {
      return operator<<(llvm::Twine(arg));
    }
    ArgStream &operator<<(const llvm::Twine &arg);

  private:
    std::vector<std::string> *out;
  };

  // Class defining a parser context.
  // Used by the parser to specify (possibly recursive) contexts where the
  // parsing/construction can fail. Any error triggered within a context will
  // keep information about the context chain.
  // This class should be used as a RAII instance in the stack.
  struct Context {
  public:
    // About to call the constructor for a matcher.
    enum ConstructMatcherEnum { ConstructMatcher };
    Context(ConstructMatcherEnum, Diagnostics *error,
            llvm::StringRef matcherName, SourceRange matcherRange);
    // About to recurse into parsing one argument for a matcher.
    enum MatcherArgEnum { MatcherArg };
    Context(MatcherArgEnum, Diagnostics *error, llvm::StringRef matcherName,
            SourceRange matcherRange, unsigned argNumber);
    ~Context();

  private:
    Diagnostics *const error;
  };

  // Context for overloaded matcher construction.
  // This context will take care of merging all errors that happen within it
  // as "candidate" overloads for the same matcher.
  struct OverloadContext {
  public:
    OverloadContext(Diagnostics *error);
    ~OverloadContext();

    // Revert all errors that happened within this context.
    void revertErrors();

  private:
    Diagnostics *const error;
    unsigned beginIndex;
  };

  // Add an error to the diagnostics.
  // All the context information will be kept on the error message.
  // Returns a helper class to allow the caller to pass the arguments for the
  // error message, using the << operator.
  ArgStream addError(SourceRange range, ErrorType error);

  // Information stored for one frame of the context.
  struct ContextFrame {
    ContextType type;
    SourceRange range;
    std::vector<std::string> args;
  };

  // Information stored for each error found.
  struct ErrorContent {
    std::vector<ContextFrame> contextStack;
    struct Message {
      SourceRange range;
      ErrorType type;
      std::vector<std::string> args;
    };
    std::vector<Message> messages;
  };
  llvm::ArrayRef<ErrorContent> errors() const { return errorValues; }

  // Returns a simple string representation of each error.
  // Each error only shows the error message without any context.
  void printToStream(llvm::raw_ostream &OS) const;
  std::string toString() const;

  // Returns the full string representation of each error.
  // Each error message contains the full context.
  void printToStreamFull(llvm::raw_ostream &OS) const;
  std::string toStringFull() const;

private:
  // Helper function used by the constructors of ContextFrame.
  ArgStream pushContextFrame(ContextType type, SourceRange range);

  std::vector<ContextFrame> contextStack;
  std::vector<ErrorContent> errorValues;
};

} // namespace matcher
} // namespace query
} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERDIAGNOSTICS_H