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

#include "Diagnostics.h"
#include "Marshallers.h"
#include "VariantValue.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir {
namespace query {
namespace matcher {

namespace internal {
class MatcherDescriptor;

// A smart (owning) pointer for MatcherDescriptor. We can't use unique_ptr
// because MatcherDescriptor is forward declared
class MatcherDescriptorPtr {
public:
  explicit MatcherDescriptorPtr(MatcherDescriptor *);
  ~MatcherDescriptorPtr();
  MatcherDescriptorPtr(MatcherDescriptorPtr &&) = default;
  MatcherDescriptorPtr &operator=(MatcherDescriptorPtr &&) = default;
  MatcherDescriptorPtr(const MatcherDescriptorPtr &) = delete;
  MatcherDescriptorPtr &operator=(const MatcherDescriptorPtr &) = delete;

  MatcherDescriptor *get() { return ptr; }

private:
  MatcherDescriptor *ptr;
};

} // namespace internal

using MatcherCtor = const internal::MatcherDescriptor *;

struct MatcherCompletion {
  MatcherCompletion() = default;
  MatcherCompletion(StringRef typedText, StringRef matcherDecl)
      : typedText(typedText), matcherDecl(matcherDecl) {}

  bool operator==(const MatcherCompletion &Other) const {
    return typedText == Other.typedText && matcherDecl == Other.matcherDecl;
  }

  // The text to type to select this matcher.
  std::string typedText;

  // The "declaration" of the matcher, with type information.
  std::string matcherDecl;
};

class Registry {
public:
  Registry() = delete;

  static internal::MatcherDescriptorPtr
  buildMatcherCtor(MatcherCtor, SourceRange nameRange,
                   ArrayRef<ParserValue> args, Diagnostics *error);

  static bool isBuilderMatcher(MatcherCtor ctor);

  // Look up a matcher in the registry by name and returns an opaque value which
  // may be used to refer to the matcher constructor, or std::nullptr if not
  // found.
  static std::optional<MatcherCtor> lookupMatcherCtor(StringRef matcherName);

  // Compute the list of completion types for context.
  //
  // Each element of context represents a matcher invocation, going from
  // outermost to innermost. Elements are pairs consisting of a reference to the
  // matcher constructor and the index of the next element in the argument list
  // of that matcher (or for the last element, the index of the completion point
  // in the argument list). An empty list requests completion for the root
  // matcher.
  static std::vector<ArgKind> getAcceptedCompletionTypes(
      llvm::ArrayRef<std::pair<MatcherCtor, unsigned>> context);

  /// Compute the list of completions that match any of acceptedTypes.
  static std::vector<MatcherCompletion>
  getMatcherCompletions(ArrayRef<ArgKind> acceptedTypes);

  /// Construct a matcher from the registry.
  static VariantMatcher constructMatcher(MatcherCtor ctor,
                                         SourceRange nameRange,
                                         ArrayRef<ParserValue> args,
                                         Diagnostics *error);
};

} // namespace matcher
} // namespace query
} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERREGISTRY_H