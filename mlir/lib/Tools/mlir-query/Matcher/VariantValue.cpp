//===--- MatcherVariantvalue.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "VariantValue.h"

namespace mlir {
namespace query {
namespace matcher {

std::string ArgKind::asString() const {
  switch (getArgKind()) {
  case AK_Matcher:
    return "Matcher";
  case AK_Boolean:
    return "boolean";
  case AK_Double:
    return "double";
  case AK_Unsigned:
    return "unsigned";
  case AK_String:
    return "string";
  }
  llvm_unreachable("unhandled ArgKind");
}

std::optional<DynMatcher> VariantMatcher::MatcherOps::constructVariadicOperator(
    DynMatcher::VariadicOperator varOp,
    ArrayRef<VariantMatcher> innerMatchers) const {
  std::vector<DynMatcher> dynMatchers;
  for (const auto &innerMatcher : innerMatchers) {
    if (!innerMatcher.value)
      return std::nullopt;
    std::optional<DynMatcher> inner = innerMatcher.value->getDynMatcher();
    if (!inner)
      return std::nullopt;
    dynMatchers.push_back(*inner);
  }
  return *DynMatcher::constructVariadic(varOp, dynMatchers);
}

VariantMatcher::Payload::~Payload() {}

class VariantMatcher::SinglePayload : public VariantMatcher::Payload {
public:
  SinglePayload(DynMatcher matcher) : matcher(matcher) {}

  std::optional<DynMatcher> getSingleMatcher() const override {
    return matcher;
  }

  std::optional<DynMatcher> getDynMatcher() const override { return matcher; }

  std::string getTypeAsString() const override { return "Matcher"; }

private:
  DynMatcher matcher;
};

class VariantMatcher::VariadicOpPayload : public VariantMatcher::Payload {
public:
  VariadicOpPayload(DynMatcher::VariadicOperator varOp,
                    std::vector<VariantMatcher> args)
      : varOp(varOp), args(std::move(args)) {}

  std::optional<DynMatcher> getSingleMatcher() const override {
    return std::nullopt;
  }

  std::optional<DynMatcher> getDynMatcher() const override {
    std::vector<DynMatcher> dynMatchers;
    for (auto variantMatcher : args) {
      std::optional<DynMatcher> dynMatcher = variantMatcher.getDynMatcher();
      if (dynMatcher)
        dynMatchers.push_back(dynMatcher.value());
    }
    auto result = DynMatcher::constructVariadic(varOp, dynMatchers);
    return *result;
  }

  std::string getTypeAsString() const override { return "VariadicOp"; }

private:
  const DynMatcher::VariadicOperator varOp;
  const std::vector<VariantMatcher> args;
};

VariantMatcher::VariantMatcher() {}

VariantMatcher VariantMatcher::SingleMatcher(DynMatcher matcher) {
  return VariantMatcher(std::make_shared<SinglePayload>(matcher));
}

VariantMatcher
VariantMatcher::VariadicOperatorMatcher(DynMatcher::VariadicOperator varOp,
                                        ArrayRef<VariantMatcher> args) {
  return VariantMatcher(
      std::make_shared<VariadicOpPayload>(varOp, std::move(args)));
}

std::optional<DynMatcher> VariantMatcher::getSingleMatcher() const {
  return value ? value->getSingleMatcher() : std::nullopt;
}

std::optional<DynMatcher> VariantMatcher::getDynMatcher() const {
  return value ? value->getDynMatcher() : std::nullopt;
}

void VariantMatcher::reset() { value.reset(); }

std::string VariantMatcher::getTypeAsString() const { return "<Nothing>"; }

VariantValue::VariantValue(const VariantValue &other) : type(VT_Nothing) {
  *this = other;
}

VariantValue::VariantValue(bool Boolean) : type(VT_Nothing) {
  setBoolean(Boolean);
}

VariantValue::VariantValue(double Double) : type(VT_Nothing) {
  setDouble(Double);
}

VariantValue::VariantValue(unsigned Unsigned) : type(VT_Nothing) {
  setUnsigned(Unsigned);
}

VariantValue::VariantValue(const StringRef String) : type(VT_Nothing) {
  setString(String);
}

VariantValue::VariantValue(const VariantMatcher &Matcher) : type(VT_Nothing) {
  setMatcher(Matcher);
}

VariantValue::~VariantValue() { reset(); }

VariantValue &VariantValue::operator=(const VariantValue &other) {
  if (this == &other)
    return *this;
  reset();
  switch (other.type) {
  case VT_Boolean:
    setBoolean(other.getBoolean());
    break;
  case VT_Double:
    setDouble(other.getDouble());
    break;
  case VT_Unsigned:
    setUnsigned(other.getUnsigned());
    break;
  case VT_String:
    setString(other.getString());
    break;
  case VT_Matcher:
    setMatcher(other.getMatcher());
    break;
  case VT_Nothing:
    type = VT_Nothing;
    break;
  }
  return *this;
}

void VariantValue::reset() {
  switch (type) {
  case VT_String:
    delete value.String;
    break;
  case VT_Matcher:
    delete value.Matcher;
    break;
  // Cases that do nothing.
  case VT_Boolean:
  case VT_Double:
  case VT_Unsigned:
  case VT_Nothing:
    break;
  }
  type = VT_Nothing;
}

bool VariantValue::isBoolean() const { return type == VT_Boolean; }

bool VariantValue::getBoolean() const {
  assert(isBoolean());
  return value.Boolean;
}

void VariantValue::setBoolean(bool newValue) {
  reset();
  type = VT_Boolean;
  value.Boolean = newValue;
}

bool VariantValue::isDouble() const { return type == VT_Double; }

double VariantValue::getDouble() const {
  assert(isDouble());
  return value.Double;
}

void VariantValue::setDouble(double newValue) {
  reset();
  type = VT_Double;
  value.Double = newValue;
}

bool VariantValue::isUnsigned() const { return type == VT_Unsigned; }

unsigned VariantValue::getUnsigned() const {
  assert(isUnsigned());
  return value.Unsigned;
}

void VariantValue::setUnsigned(unsigned newValue) {
  reset();
  type = VT_Unsigned;
  value.Unsigned = newValue;
}

bool VariantValue::isString() const { return type == VT_String; }

const StringRef &VariantValue::getString() const {
  assert(isString());
  return *value.String;
}

void VariantValue::setString(const StringRef &newValue) {
  reset();
  type = VT_String;
  value.String = new StringRef(newValue);
}

bool VariantValue::isMatcher() const { return type == VT_Matcher; }

const VariantMatcher &VariantValue::getMatcher() const {
  assert(isMatcher());
  return *value.Matcher;
}

void VariantValue::setMatcher(const VariantMatcher &newValue) {
  reset();
  type = VT_Matcher;
  value.Matcher = new VariantMatcher(newValue);
}

std::string VariantValue::getTypeAsString() const {
  switch (type) {
  case VT_String:
    return "String";
  case VT_Matcher:
    return "Matcher";
  case VT_Unsigned:
    return "Unsigned";
  case VT_Boolean:
    return "Boolean";
  case VT_Double:
    return "Double";
  case VT_Nothing:
    return "Nothing";
  }
  llvm_unreachable("Invalid Type");
}

} // end namespace matcher
} // end namespace query
} // end namespace mlir