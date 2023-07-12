//===--- MatcherVariantValue.h --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Supports all the types required for dynamic Matcher construction.
// Used by the registry to construct matchers in a generic way.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_MATCHERVARIANTVALUE_H
#define MLIR_TOOLS_MLIRQUERY_MATCHERVARIANTVALUE_H

#include "Diagnostics.h"
#include "MatchersInternal.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/type_traits.h"

namespace mlir {
namespace query {
namespace matcher {

/// Kind identifier.
///
/// It supports all types that VariantValue can contain.
class ArgKind {
public:
  enum Kind { AK_Matcher, AK_Boolean, AK_Double, AK_Unsigned, AK_String };
  ArgKind(Kind k) : k(k) {}

  Kind getArgKind() const { return k; }

  bool operator<(const ArgKind &other) const { return k < other.k; }

  /// To the requested destination type.
  /// String representation of the type.
  std::string asString() const;

private:
  Kind k;
};

/// A variant matcher object.
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
    /// Constructs a variadic typed matcher from \p innerMatchers.
    /// Will try to convert each inner matcher to the destination type and
    /// return std::nullopt if it fails to do so.
    std::optional<DynMatcher>
    constructVariadicOperator(DynMatcher::VariadicOperator varOp,
                              ArrayRef<VariantMatcher> innerMatchers) const;
  };

  /// \brief Payload interface to be specialized by each matcher type.
  ///
  /// It follows a similar interface as VariantMatcher itself.
  class Payload {
  public:
    virtual ~Payload();
    virtual std::optional<DynMatcher> getSingleMatcher() const = 0;
    virtual std::optional<DynMatcher> getDynMatcher() const = 0;
    virtual std::string getTypeAsString() const = 0;
  };

public:
  /// A null matcher.
  VariantMatcher();

  /// \brief Clones the provided matcher.
  static VariantMatcher SingleMatcher(DynMatcher matcher);

  /// \brief Clones the provided matchers.
  ///
  /// They should be the result of a polymorphic matcher.
  static VariantMatcher PolymorphicMatcher(ArrayRef<DynMatcher> matchers);

  /// \brief Creates a 'variadic' operator matcher.
  ///
  /// It will bind to the appropriate type on getTypedMatcher<T>().
  static VariantMatcher
  VariadicOperatorMatcher(DynMatcher::VariadicOperator varOp,
                          ArrayRef<VariantMatcher> args);

  /// Makes the matcher the "null" matcher.
  void reset();

  /// Checks if the matcher is null.
  bool isNull() const { return !value; }

  /// Returns a single matcher, if there is no ambiguity.
  ///
  /// Returns the matcher, if there is only one matcher.
  std::optional<DynMatcher> getSingleMatcher() const;

  std::optional<DynMatcher> getDynMatcher() const;

  /// \brief String representation of the type of the value.
  ///
  /// If the underlying matcher is a polymorphic one, the string will show all
  /// the types.
  std::string getTypeAsString() const;

private:
  explicit VariantMatcher(std::shared_ptr<Payload> value)
      : value(std::move(value)) {}

  class SinglePayload;
  class VariadicOpPayload;

  std::shared_ptr<const Payload> value;
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
  VariantValue() : type(VT_Nothing) {}

  VariantValue(const VariantValue &other);
  ~VariantValue();
  VariantValue &operator=(const VariantValue &other);

  // Specific constructors for each supported type.
  VariantValue(bool Boolean);
  VariantValue(double Double);
  VariantValue(unsigned Unsigned);
  VariantValue(const StringRef String);
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

  ValueType type;
  AllValues value;
};

// A VariantValue instance annotated with its parser context.
struct ParserValue {
  ParserValue() {}
  llvm::StringRef text;
  SourceRange range;
  VariantValue value;
};

} // end namespace matcher
} // end namespace query
} // end namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERVARIANTVALUE_H