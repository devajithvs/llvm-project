//===--- MatcherVariantValue.cpp - Polymorphic value type -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Polymorphic value type.
//
//===----------------------------------------------------------------------===//

#include "MatcherVariantValue.h"

namespace mlir {
namespace query {
namespace matcher {

VariantValue::VariantValue(const VariantValue &Other) : Type(VT_Nothing) {
  *this = Other;
}

VariantValue::VariantValue(bool Boolean) : Type(VT_Nothing) {
  setBoolean(Boolean);
}

VariantValue::VariantValue(double Double) : Type(VT_Nothing) {
  setDouble(Double);
}

VariantValue::VariantValue(unsigned Unsigned) : Type(VT_Nothing) {
  setUnsigned(Unsigned);
}

VariantValue::VariantValue(const StringRef &String) : Type(VT_Nothing) {
  setString(String);
}

VariantValue::VariantValue(const DynMatcher &Matcher) : Type(VT_Nothing) {
  setMatcher(Matcher);
}

VariantValue::~VariantValue() { reset(); }

VariantValue &VariantValue::operator=(const VariantValue &Other) {
  if (this == &Other)
    return *this;
  reset();
  switch (Other.Type) {
  case VT_Boolean:
    setBoolean(Other.getBoolean());
    break;
  case VT_Double:
    setDouble(Other.getDouble());
    break;
  case VT_Unsigned:
    setUnsigned(Other.getUnsigned());
    break;
  case VT_String:
    setString(Other.getString());
    break;
  case VT_Matcher:
    setMatcher(Other.getMatcher());
    break;
  case VT_Nothing:
    Type = VT_Nothing;
    break;
  }
  return *this;
}

void VariantValue::reset() {
  switch (Type) {
  case VT_String:
    delete Value.String;
    break;
  case VT_Matcher:
    delete Value.Matcher;
    break;
  // Cases that do nothing.
  case VT_Boolean:
  case VT_Double:
  case VT_Unsigned:
  case VT_Nothing:
    break;
  }
  Type = VT_Nothing;
}

bool VariantValue::isBoolean() const { return Type == VT_Boolean; }

bool VariantValue::getBoolean() const {
  assert(isBoolean());
  return Value.Boolean;
}

void VariantValue::setBoolean(bool NewValue) {
  reset();
  Type = VT_Boolean;
  Value.Boolean = NewValue;
}

bool VariantValue::isDouble() const { return Type == VT_Double; }

double VariantValue::getDouble() const {
  assert(isDouble());
  return Value.Double;
}

void VariantValue::setDouble(double NewValue) {
  reset();
  Type = VT_Double;
  Value.Double = NewValue;
}

bool VariantValue::isUnsigned() const { return Type == VT_Unsigned; }

unsigned VariantValue::getUnsigned() const {
  assert(isUnsigned());
  return Value.Unsigned;
}

void VariantValue::setUnsigned(unsigned NewValue) {
  reset();
  Type = VT_Unsigned;
  Value.Unsigned = NewValue;
}

bool VariantValue::isString() const { return Type == VT_String; }

const StringRef &VariantValue::getString() const {
  assert(isString());
  return *Value.String;
}

void VariantValue::setString(const StringRef &NewValue) {
  reset();
  Type = VT_String;
  Value.String = new StringRef(NewValue);
}

bool VariantValue::isMatcher() const { return Type == VT_Matcher; }

const DynMatcher &VariantValue::getMatcher() const {
  assert(isMatcher());
  return *Value.Matcher;
}

void VariantValue::setMatcher(const DynMatcher &NewValue) {
  reset();
  Type = VT_Matcher;
  // FIXME
  Value.Matcher = NewValue.clone();
}

void VariantValue::takeMatcher(DynMatcher *NewValue) {
  reset();
  Type = VT_Matcher;
  Value.Matcher = NewValue;
}

} // end namespace matcher
} // end namespace query
} // end namespace mlir