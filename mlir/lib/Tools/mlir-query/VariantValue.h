//===--- VariantValue.h - Polymorphic value type --------------------------===//
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

// Variant value class.
//
// Basically, a tagged union with value type semantics.
// It is used by the registry as the return value and argument type for the
// matcher factory methods.
// It can be constructed from any of the supported types. It supports
// copy/assignment.
//
// Supported types:
//  - bool
//  - double
//  - unsigned
//  - StringRef
//  - Any Matcher
class VariantValue {
public:
  VariantValue() : Type(VT_Nothing) {}

  VariantValue(const VariantValue &Other);
  ~VariantValue();
  VariantValue &operator=(const VariantValue &Other);

  // Specific constructors for each supported type.
  VariantValue(bool Boolean);
  VariantValue(double Double);
  VariantValue(unsigned Unsigned);
  VariantValue(const StringRef &String);
  VariantValue(const DynMatcher &Matchers);

  // Boolean value functions.
  bool isBoolean() const;
  bool getBoolean() const;
  void setBoolean(bool Boolean);

  // Double value functions.
  bool isDouble() const;
  double getDouble() const;
  void setDouble(double Double);

  // Unsigned value functions.
  bool isUnsigned() const;
  unsigned getUnsigned() const;
  void setUnsigned(unsigned Unsigned);

  // String value functions.
  bool isString() const;
  const StringRef &getString() const;
  void setString(const StringRef &String);

  // Matcher value functions.
  bool isMatcher() const;
  const DynMatcher &getMatcher() const;
  void setMatcher(const DynMatcher &Matcher);

  // Set the value to be DynMatcher by taking ownership of the
  // object.
  void takeMatcher(DynMatcher *Matcher);

  // Specialized Matcher<T> is/get functions.
  template <class T>
  bool isTypedMatcher() const {
    // TODO: Add some logic to test if T is actually valid for the underlying
    // type of the matcher.
    return isMatcher();
  }

private:
  void reset();

  // All supported value types.
  enum ValueType {
    VT_Nothing,
    VT_Boolean,
    VT_Double,
    VT_Unsigned,
    VT_String,
    VT_Matcher,
  };

  // All supported value types.
  union AllValues {
    unsigned Unsigned;
    double Double;
    bool Boolean;
    StringRef *String;
    DynMatcher *Matcher;
  };

  ValueType Type;
  AllValues Value;
};

} // end namespace matcher
} // end namespace query
} // end namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERS_VARIANTVALUE_H