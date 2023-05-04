//===--- Registry.h - Matcher registry ------------------------------------===//
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

#ifndef MLIR_TOOLS_MLIRQUERY_MATCHERS_REGISTRY_H
#define MLIR_TOOLS_MLIRQUERY_MATCHERS_REGISTRY_H

#include "Diagnostics.h"
#include "Marshallers.h"
#include "VariantValue.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace query {
namespace matcher {

class Registry {
public:
  // Construct a matcher from the registry by name.

  // Consult the registry of known matchers and construct the appropriate
  // matcher by name. MatcherName is the name of the matcher to instantiate.

  // Args is the argument list for the matcher. The number and types of the
  // values must be valid for the matcher requested. Otherwise, the function
  // will return an error.

  // Returns the matcher if no error was found. nullptr if the matcher is not
  // found, or if the number of arguments or argument types do not
  // match the signature. In that case Error will contain the description
  // of the error.
  // TODO: Cleanup - Remove one of these
  static DynMatcher *constructMatcher(StringRef MatcherName,
                                      const SourceRange &NameRange,
                                      ArrayRef<ParserValue> Args,
                                      Diagnostics *Error);
  static DynMatcher *
  constructMatcherWrapper(StringRef MatcherName, const SourceRange &NameRange,
                          bool ExtractFunction, StringRef FunctionName,
                          ArrayRef<ParserValue> Args, Diagnostics *Error);
};

} // namespace matcher
} // namespace query
} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERS_REGISTRY_H