//===--- MatcherRegistry.h - Matcher registry -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registry of all known matchers.
//
// The registry provides a generic interface to construct any matcher by name.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_MATCHERREGISTRY_H
#define MLIR_TOOLS_MLIRQUERY_MATCHERREGISTRY_H

#include "Marshallers.h"
#include "MatcherDiagnostics.h"
#include "MatcherVariantValue.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir {
namespace query {
namespace matcher {

namespace internal {
class MatcherDescriptor;

/// A smart (owning) pointer for MatcherDescriptor. We can't use unique_ptr
/// because MatcherDescriptor is forward declared
class MatcherDescriptorPtr {
public:
  explicit MatcherDescriptorPtr(MatcherDescriptor *);
  ~MatcherDescriptorPtr();
  MatcherDescriptorPtr(MatcherDescriptorPtr &&) = default;
  MatcherDescriptorPtr &operator=(MatcherDescriptorPtr &&) = default;
  MatcherDescriptorPtr(const MatcherDescriptorPtr &) = delete;
  MatcherDescriptorPtr &operator=(const MatcherDescriptorPtr &) = delete;

  MatcherDescriptor *get() { return Ptr; }

private:
  MatcherDescriptor *Ptr;
};

} // namespace internal

using MatcherCtor = const internal::MatcherDescriptor *;

struct MatcherCompletion {
  MatcherCompletion() = default;
  MatcherCompletion(StringRef TypedText, StringRef MatcherDecl)
      : TypedText(TypedText), MatcherDecl(MatcherDecl) {}

  bool operator==(const MatcherCompletion &Other) const {
    return TypedText == Other.TypedText && MatcherDecl == Other.MatcherDecl;
  }

  /// The text to type to select this matcher.
  std::string TypedText;

  /// The "declaration" of the matcher, with type information.
  std::string MatcherDecl;
};

class Registry {
public:
  Registry() = delete;

  static internal::MatcherDescriptorPtr
  buildMatcherCtor(MatcherCtor, SourceRange NameRange,
                   ArrayRef<ParserValue> Args, Diagnostics *Error);

  static bool isBuilderMatcher(MatcherCtor Ctor);

  // Look up a matcher in the registry by name,
  /// \return An opaque value which may be used to refer to the matcher
  /// constructor, or std::optional<MatcherCtor>() if not found.
  static std::optional<MatcherCtor> lookupMatcherCtor(StringRef MatcherName);

  /// Compute the list of completion types for \p Context.
  ///
  /// Each element of \p Context represents a matcher invocation, going from
  /// outermost to innermost. Elements are pairs consisting of a reference to
  /// the matcher constructor and the index of the next element in the
  /// argument list of that matcher (or for the last element, the index of
  /// the completion point in the argument list). An empty list requests
  /// completion for the root matcher.
  static std::vector<ArgKind> getAcceptedCompletionTypes(
      llvm::ArrayRef<std::pair<MatcherCtor, unsigned>> Context);

  /// Compute the list of completions that match any of
  /// \p AcceptedTypes.
  ///
  /// \param AcceptedTypes All types accepted for this completion.
  ///
  /// \return All completions for the specified types.
  /// Completions should be valid when used in \c lookupMatcherCtor().
  /// The matcher constructed from the return of \c lookupMatcherCtor()
  /// should be convertible to some type in \p AcceptedTypes.
  static std::vector<MatcherCompletion>
  getMatcherCompletions(ArrayRef<ArgKind> AcceptedTypes);

  /// Construct a matcher from the registry.

  // Construct a matcher from the registry.
  // Ctor The matcher constructor to instantiate.

  // Args is the argument list for the matcher. The number and types of the
  // values must be valid for the matcher requested. Otherwise, the function
  // will return an error.

  // Returns the matcher if no error was found. nullptr if the matcher is not
  // found, or if the number of arguments or argument types do not
  // match the signature. In that case Error will contain the description
  // of the error.
  // TODO: Cleanup - Remove one of these
  static VariantMatcher constructMatcher(MatcherCtor Ctor,
                                         SourceRange NameRange,
                                         ArrayRef<ParserValue> Args,
                                         Diagnostics *Error);

  static VariantMatcher constructFunctionMatcher(MatcherCtor Ctor,
                                                 SourceRange NameRange,
                                                 StringRef FunctionName,
                                                 ArrayRef<ParserValue> Args,
                                                 Diagnostics *Error);

  // TODO: FIX ALL COMMENTS
  // TODO: FIX PRINT
  // TODO: Convert to camelCase
  /// Construct a matcher from the registry and bind it.
  ///
  /// Similar the \c constructMatcher() above, but it then tries to bind the
  /// matcher to the specified \c BindID.
  /// If the matcher is not bindable, it sets an error in \c Error and returns
  /// a null matcher.
  static VariantMatcher constructBoundMatcher(MatcherCtor Ctor,
                                              SourceRange NameRange,
                                              StringRef BindID,
                                              ArrayRef<ParserValue> Args,
                                              Diagnostics *Error);
};

} // namespace matcher
} // namespace query
} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERREGISTRY_H