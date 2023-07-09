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


VariantMatcher::MatcherOps::~MatcherOps() {}
VariantMatcher::Payload::~Payload() {}

class VariantMatcher::SinglePayload : public VariantMatcher::Payload {
public:
  SinglePayload(DynMatcher Matcher) : Matcher(Matcher) {}

  std::optional<DynMatcher> getSingleMatcher() const override {
    return Matcher;
  }

  std::optional<DynMatcher> getDynMatcher() const override {
    return Matcher;
  }

  std::string getTypeAsString() const override {
    return "Matcher";
  }

  void makeTypedMatcher(MatcherOps &Ops) const override {
    if (Ops.canConstructFrom(Matcher))
      Ops.constructFrom(Matcher);
  }

private:
  DynMatcher Matcher;
};

class VariantMatcher::PolymorphicPayload : public VariantMatcher::Payload {
public:
  PolymorphicPayload(ArrayRef<DynMatcher> MatchersIn)
      : Matchers(MatchersIn) {}

  virtual ~PolymorphicPayload() {}

  std::optional<DynMatcher> getSingleMatcher() const override {
    if (Matchers.size() != 1)
      return std::optional<DynMatcher>();
    return Matchers[0];
  }

  // TODO: Remove poly
  std::optional<DynMatcher> getDynMatcher() const override {
    if (Matchers.size() != 1)
      return std::optional<DynMatcher>();
    return Matchers[0];
  }

  std::string getTypeAsString() const override {
    return "Matcher";
  }

  void makeTypedMatcher(MatcherOps &Ops) const override {
    const DynMatcher *Found = nullptr;
    for (size_t i = 0, e = Matchers.size(); i != e; ++i) {
      if (Found)
        return;
      Found = &Matchers[i];
    }
    if (Found)
      Ops.constructFrom(*Found);
  }

  const std::vector<DynMatcher> Matchers;
};

class VariantMatcher::VariadicOpPayload : public VariantMatcher::Payload {
public:
  // TODO: Rename Func
  VariadicOpPayload(DynMatcher::VariadicOperator Func,
                    ArrayRef<VariantMatcher> Args)
      : Func(Func), Args(Args) {}

  std::optional<DynMatcher> getSingleMatcher() const override {
    llvm::errs() << "empty VariadicOpPayload" << "\n";

    return std::optional<DynMatcher>();
  }

  std::optional<DynMatcher> getDynMatcher() const override {
    std::vector<DynMatcher> DynMatchers;
    for (auto variantMatcher : Args) {
      std::optional<DynMatcher> dynMatcher = variantMatcher.getDynMatcher();
      DynMatchers.push_back(dynMatcher.value());
    }
    auto result = DynMatcher::constructVariadic(Func, DynMatchers);
    return *result;
  }

  // TODO: Remove
  std::string getTypeAsString() const override {
    return "Op";
  }

  void makeTypedMatcher(MatcherOps &Ops) const override {
    Ops.constructVariadicOperator(Func, Args);
  }

private:
  const DynMatcher::VariadicOperator Func;
  const std::vector<VariantMatcher> Args;
};

VariantMatcher::VariantMatcher() {}

VariantMatcher VariantMatcher::SingleMatcher(DynMatcher Matcher) {
  return VariantMatcher(new SinglePayload(Matcher));
}

VariantMatcher
VariantMatcher::PolymorphicMatcher(ArrayRef<DynMatcher> Matchers) {
  return VariantMatcher(new PolymorphicPayload(Matchers));
}

VariantMatcher VariantMatcher::VariadicOperatorMatcher(
     DynMatcher::VariadicOperator varOp,
    ArrayRef<VariantMatcher> Args) {
  return VariantMatcher(new VariadicOpPayload(varOp, std::move(Args)));
}

std::optional<DynMatcher> VariantMatcher::getSingleMatcher() const {
  if (Value) llvm::errs() << "getSingleMatcher success: " << "\n";
  if (!Value) llvm::errs() << "getSingleMatcher failed: " << "\n";

  return Value ? Value->getSingleMatcher() : std::optional<DynMatcher>();
}

std::optional<DynMatcher> VariantMatcher::getDynMatcher() const {
  if (Value) llvm::errs() << "getDynMatcher success: " << "\n";
  if (!Value) llvm::errs() << "getDynMatcher failed: " << "\n";

  return Value ? Value->getDynMatcher() : std::optional<DynMatcher>();
}

void VariantMatcher::reset() { Value.reset(); }

// TODO: Remove
std::string VariantMatcher::getTypeAsString() const {
  return "<Nothing>";
}


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

VariantValue::VariantValue(const VariantMatcher &Matcher) : Type(VT_Nothing) {
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

const VariantMatcher &VariantValue::getMatcher() const {
  assert(isMatcher());
  llvm::errs() << "getMatcher working\n";
  return *Value.Matcher;
}

void VariantValue::setMatcher(const VariantMatcher &NewValue) {
  reset();
  Type = VT_Matcher;
  Value.Matcher = new VariantMatcher(NewValue);
}

std::string VariantValue::getTypeAsString() const {
  switch (Type) {
  case VT_String: return "String";
  case VT_Matcher: return "Matcher";
  case VT_Unsigned: return "Unsigned";
  case VT_Boolean: return "Boolean";
  case VT_Double: return "Double";
  case VT_Nothing: return "Nothing";
  }
  llvm_unreachable("Invalid Type");
}

} // end namespace matcher
} // end namespace query
} // end namespace mlir