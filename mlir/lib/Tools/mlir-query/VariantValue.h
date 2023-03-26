//===--- VariantValue.h - Polymorphic value type -*- C++ -*-===/
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Polymorphic value type.
//
// Supports all the types required for dynamic Matcher construction.
// Used by the registry to construct matchers in a generic way.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_MATCHERS_VARIANTVALUE_H
#define MLIR_TOOLS_MLIRQUERY_MATCHERS_VARIANTVALUE_H

#include "MatchersInternal.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/type_traits.h"

namespace mlir {
namespace query {
namespace matcher {

/// \brief Variant value class.
///
/// Basically, a tagged union with value type semantics.
/// It is used by the registry as the return value and argument type for the
/// matcher factory methods.
/// It can be constructed from any of the supported types. It supports
/// copy/assignment.
///
/// Supported types:
///  - \c StringRef
///  - \c Any \c Matcher
class VariantValue {
public:
  VariantValue() : Type(VT_Nothing) {}

  VariantValue(const VariantValue &Other);
  ~VariantValue();
  VariantValue &operator=(const VariantValue &Other);

  /// \brief Specific constructors for each supported type.
  VariantValue(const StringRef &String);
  VariantValue(const Matcher &Matcher);

  /// \brief String value functions.
  bool isString() const;
  const StringRef &getString() const;
  void setString(const StringRef &String);

  /// \brief Matcher value functions.
  bool isMatcher() const;
  const Matcher &getMatcher() const;
  void setMatcher(const Matcher &Matcher);
  /// \brief Set the value to be \c Matcher by taking ownership of the object.
  void takeMatcher(Matcher *Matcher);

  /// \brief Specialized Matcher<T> is/get functions.
  template <class T>
  bool isTypedMatcher() const {
    // TODO: Add some logic to test if T is actually valid for the underlying
    // type of the matcher.
    return isMatcher();
  }

private:
  void reset();

  /// \brief All supported value types.
  enum ValueType { VT_Nothing, VT_String, VT_Matcher };

  /// \brief All supported value types.
  union AllValues {
    StringRef *String;
    Matcher *Matcher;
  };

  ValueType Type;
  AllValues Value;
};

/// A VariantValue instance annotated with its parser context.
struct ParserValue {
  ParserValue() {}
  StringRef Text;
  VariantValue Value;
};

} // end namespace matcher
} // end namespace query
} // end namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERS_VARIANTVALUE_H