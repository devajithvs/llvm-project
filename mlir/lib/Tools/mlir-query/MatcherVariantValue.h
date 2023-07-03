//===--- MatcherVariantValue.h - Polymorphic value type -------------------===//
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

#ifndef MLIR_TOOLS_MLIRQUERY_MATCHERVARIANTVALUE_H
#define MLIR_TOOLS_MLIRQUERY_MATCHERVARIANTVALUE_H

#include "MatchersInternal.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/type_traits.h"

namespace mlir {
namespace query {
namespace matcher {

/// \brief A variant matcher object.
///
/// The purpose of this object is to abstract simple and polymorphic matchers
/// into a single object type.
/// Polymorphic matchers might be implemented as a list of all the possible
/// overloads of the matcher. \c VariantMatcher knows how to select the
/// appropriate overload when needed.
/// To get a real matcher object out of a \c VariantMatcher you can do:
///  - getSingleMatcher() which returns a matcher, only if it is not ambiguous
///    to decide which matcher to return. Eg. it contains only a single
///    matcher, or a polymorphic one with only one overload.
///  - hasTypedMatcher<T>()/getTypedMatcher<T>(): These calls will determine if
///    the underlying matcher(s) can unambiguously return a Matcher<T>.
class VariantMatcher {
  /// \brief Methods that depend on T from hasTypedMatcher/getTypedMatcher.
  class MatcherOps {
  public:
    virtual ~MatcherOps();
    virtual bool canConstructFrom(const DynMatcher &Matcher) const = 0;
    virtual void constructFrom(const DynMatcher &Matcher) = 0;
    virtual void constructVariadicOperator(VariadicOperatorFunction Func, ArrayRef<VariantMatcher> InnerMatchers) = 0;
  };

  /// \brief Payload interface to be specialized by each matcher type.
  ///
  /// It follows a similar interface as VariantMatcher itself.
  class Payload : public llvm::RefCountedBase<Payload> {
  public:
    virtual ~Payload();
    virtual std::optional<DynMatcher> getSingleMatcher() const = 0;
    virtual std::string getTypeAsString() const = 0;
    virtual void makeTypedMatcher(MatcherOps &Ops) const = 0;
  };

public:
  /// \brief A null matcher.
  VariantMatcher();

  /// \brief Clones the provided matcher.
  static VariantMatcher SingleMatcher(const DynMatcher &Matcher);

  /// \brief Clones the provided matchers.
  ///
  /// They should be the result of a polymorphic matcher.
  static VariantMatcher PolymorphicMatcher(ArrayRef<DynMatcher> Matchers);

  /// \brief Creates a 'variadic' operator matcher.
  ///
  /// It will bind to the appropriate type on getTypedMatcher<T>().
  static VariantMatcher VariadicOperatorMatcher(VariadicOperatorFunction Func, ArrayRef<VariantMatcher> Args);

  /// \brief Makes the matcher the "null" matcher.
  void reset();

  /// \brief Whether the matcher is null.
  bool isNull() const { return !Value; }

  /// \brief Return a single matcher, if there is no ambiguity.
  ///
  /// \returns the matcher, if there is only one matcher. An empty Optional, if
  /// the underlying matcher is a polymorphic matcher with more than one
  /// representation.
  std::optional<DynMatcher> getSingleMatcher() const;


  /// \brief Return this matcher as a \c DynMatcher.
  ///
  /// Handles the different types (Single, Polymorphic) accordingly.
  DynMatcher getTypedMatcher() const {
    TypedMatcherOps Ops;
    Value->makeTypedMatcher(Ops);
    assert(Ops.hasMatcher() && "hasMatcher() == false");
    return Ops.matcher();
  }

  /// \brief String representation of the type of the value.
  ///
  /// If the underlying matcher is a polymorphic one, the string will show all
  /// the types.
  std::string getTypeAsString() const;

private:
  explicit VariantMatcher(Payload *Value) : Value(Value) {}

  class SinglePayload;
  class PolymorphicPayload;
  class VariadicOpPayload;

  class TypedMatcherOps : public MatcherOps {
  public:
    // TODO: Cleanup
    bool canConstructFrom(const DynMatcher &Matcher) const override {
      return true;
    }

    void constructFrom(const DynMatcher& Matcher) override {
      Out.reset(&Matcher);
    }

    void constructVariadicOperator(VariadicOperatorFunction Func, ArrayRef<VariantMatcher> InnerMatchers) override {
      std::vector<DynMatcher> DynMatchers;
      for (size_t i = 0, e = InnerMatchers.size(); i != e; ++i) {
        DynMatchers.push_back(InnerMatchers[i].getTypedMatcher());
      }
      Out.reset(new DynMatcher(new VariadicOperatorMatcherInterface(Func, DynMatchers)));
    }

    bool hasMatcher() const { return Out.get() != nullptr; }
    const DynMatcher &matcher() const { return *Out; }

  private:
    std::shared_ptr<const DynMatcher> Out;
  };

  llvm::IntrusiveRefCntPtr<const Payload> Value;
};

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
//  - VariantMatcher
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
  VariantValue(const VariantMatcher &Matcher);

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
  const VariantMatcher &getMatcher() const;
  void setMatcher(const VariantMatcher &Matcher);
  
  /// \brief String representation of the type of the value.
  std::string getTypeAsString() const;
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
    VariantMatcher *Matcher;
  };

  ValueType Type;
  AllValues Value;
};

} // end namespace matcher
} // end namespace query
} // end namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERVARIANTVALUE_H