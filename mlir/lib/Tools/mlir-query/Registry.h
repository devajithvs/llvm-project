//===--- Registry.h - Matcher registry -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Registry of all known matchers.
///
/// The registry provides a generic interface to construct any matcher by name.
///
//===----------------------------------------------------------------------===//

#ifndef MLIR_QUERY_MATCHERS_DYNAMIC_REGISTRY_H
#define MLIR_QUERY_MATCHERS_DYNAMIC_REGISTRY_H

#include "VariantValue.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace query {
namespace matcher {

class Registry {
public:
  /// \brief Construct a matcher from the registry by name.
  ///
  /// Consult the registry of known matchers and construct the appropriate
  /// matcher by name.
  ///
  /// \param MatcherName The name of the matcher to instantiate.
  ///
  /// \param Args The argument list for the matcher. The number and types of the
  ///   values must be valid for the matcher requested. Otherwise, the function
  ///   will return an error.
  ///
  /// \return The matcher if no error was found. NULL if the matcher is not
  //    found, or if the number of arguments or argument types do not
  ///   match the signature. In that case \c Error will contain the description
  ///   of the error.
  static MatcherImplementation *constructMatcher(StringRef MatcherName, ArrayRef<ParserValue> Args);

};

}  // namespace matcher
}  // namespace query
}  // namespace mlir

#endif  // MLIR_QUERY_MATCHERS_DYNAMIC_REGISTRY_H