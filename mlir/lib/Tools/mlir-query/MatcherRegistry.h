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
} // namespace internal

typedef const internal::MatcherDescriptor *MatcherCtor;

class Registry {
public:
  // Look up a matcher in the registry by name,
  // Consult the registry of known matchers and construct the appropriate
  // matcher by name.
  // return An opaque value which may be used to refer to the matcher
  // constructor, or optional<MatcherCtor>() if not found.  In that case
  // Error will contain the description of the error.
  static std::optional<MatcherCtor>
  lookupMatcherCtor(StringRef MatcherName, const SourceRange &NameRange,
                    Diagnostics *Error);

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
                                      const SourceRange &NameRange,
                                      ArrayRef<ParserValue> Args,
                                      Diagnostics *Error);

  static VariantMatcher constructMatcherWrapper(MatcherCtor Ctor, const SourceRange &NameRange,
                          bool ExtractFunction, StringRef FunctionName,
                          ArrayRef<ParserValue> Args, Diagnostics *Error);
  
  // TODO: FIX COMMENT
  /// \brief Construct a matcher from the registry and bind it.
  ///
  /// Similar the \c constructMatcher() above, but it then tries to bind the
  /// matcher to the specified \c BindID.
  /// If the matcher is not bindable, it sets an error in \c Error and returns
  /// a null matcher.
  static VariantMatcher constructBoundMatcher(MatcherCtor Ctor,
                                              const SourceRange &NameRange,
                                              StringRef BindID,
                                              ArrayRef<ParserValue> Args,
                                              Diagnostics *Error);
};

} // namespace matcher
} // namespace query
} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERREGISTRY_H