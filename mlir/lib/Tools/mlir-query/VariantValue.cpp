//===--- VariantValue.cpp - Polymorphic value type -*- C++ -*-===/
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

#include "VariantValue.h"

namespace mlir {
namespace query {
namespace matcher {

VariantValue::VariantValue(const VariantValue &Other) : Type(VT_Nothing) {
  *this = Other;
}

VariantValue::VariantValue(const DynTypedMatcher &Matcher) : Type(VT_Nothing) {
  setMatcher(Matcher);
}

VariantValue::VariantValue(const StringRef &String) : Type(VT_Nothing) {
  setString(String);
}

VariantValue::~VariantValue() { reset(); }

VariantValue &VariantValue::operator=(const VariantValue &Other) {
  if (this == &Other)
    return *this;
  reset();
  switch (Other.Type) {
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
  case VT_Nothing:
    break;
  }
  Type = VT_Nothing;
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

const DynTypedMatcher &VariantValue::getMatcher() const {
  assert(isMatcher());
  return *Value.Matcher;
}

void VariantValue::setMatcher(const Matcher &NewValue) {
  reset();
  Type = VT_Matcher;
  // FIXME
  Value.Matcher = NewValue.clone();
}

void VariantValue::takeMatcher(DynTypedMatcher *NewValue) {
  reset();
  Type = VT_Matcher;
  Value.Matcher = NewValue;
}

} // end namespace matcher
} // end namespace query
} // end namespace mlir