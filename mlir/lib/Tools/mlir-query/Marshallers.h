//===--- Marshallers.h - Generic matcher function marshallers -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functions templates and classes to wrap matcher construct functions.
//
// A collection of template function and classes that provide a generic
// marshalling layer on top of matcher construct functions.
// These are used by the registry to export all marshaller constructors with
// the same generic interface. This mechanism is inspired by clang-query.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_MATCHERS_MARSHALLERS_H
#define MLIR_TOOLS_MLIRQUERY_MATCHERS_MARSHALLERS_H

#include "MatcherDiagnostics.h"
#include "MatcherVariantValue.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/type_traits.h"

#include "llvm/Support/Debug.h"
using llvm::dbgs;

#define DEBUG_TYPE "mlir-query"
#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
namespace query {
namespace matcher {

namespace internal {

// Helper template class to just from argument type to the right is/get
// functions in VariantValue.
// Used to verify and extract the matcher arguments below.
template <class T>
struct ArgTypeTraits;
template <class T>
struct ArgTypeTraits<const T &> : public ArgTypeTraits<T> {};

template <>
struct ArgTypeTraits<StringRef> {

  static bool hasCorrectType(const VariantValue &Value) {
    return Value.isString();
  }

  static const StringRef &get(const VariantValue &Value) {
    return Value.getString();
  }

  static ArgKind getKind() { return ArgKind(ArgKind::AK_String); }

  static std::optional<std::string> getBestGuess(const VariantValue &) {
    return std::nullopt;
  }
};

template <>
struct ArgTypeTraits<DynMatcher> {

  static bool hasCorrectType(const VariantValue &Value) {
    return Value.isMatcher();
  }

  static DynMatcher get(const VariantValue &Value) {
    return *Value.getMatcher().getDynMatcher();
  }

  static ArgKind getKind() { return ArgKind(ArgKind::AK_Matcher); }

  static std::optional<std::string> getBestGuess(const VariantValue &) {
    return std::nullopt;
  }
};

template <>
struct ArgTypeTraits<bool> {

  static bool hasCorrectType(const VariantValue &Value) {
    return Value.isBoolean();
  }

  static bool get(const VariantValue &Value) { return Value.getBoolean(); }

  static ArgKind getKind() { return ArgKind(ArgKind::AK_Boolean); }

  static std::optional<std::string> getBestGuess(const VariantValue &) {
    return std::nullopt;
  }
};

template <>
struct ArgTypeTraits<double> {

  static bool hasCorrectType(const VariantValue &Value) {
    return Value.isDouble();
  }

  static double get(const VariantValue &Value) { return Value.getDouble(); }

  static ArgKind getKind() { return ArgKind(ArgKind::AK_Double); }

  static std::optional<std::string> getBestGuess(const VariantValue &) {
    return std::nullopt;
  }
};

template <>
struct ArgTypeTraits<unsigned> {
  static bool hasCorrectType(const VariantValue &Value) {
    return Value.isUnsigned();
  }

  static unsigned get(const VariantValue &Value) { return Value.getUnsigned(); }

  static ArgKind getKind() { return ArgKind(ArgKind::AK_Unsigned); }

  static std::optional<std::string> getBestGuess(const VariantValue &) {
    return std::nullopt;
  }
};

// Generic Matcher descriptor interface.
// Provides a create() method that constructs the matcher from the provided
// arguments.
class MatcherDescriptor {
public:
  virtual ~MatcherDescriptor() = default;
  virtual VariantMatcher create(SourceRange NameRange,
                                const ArrayRef<ParserValue> Args,
                                Diagnostics *Error) const = 0;

  virtual bool isBuilderMatcher() const { return false; }

  virtual std::unique_ptr<MatcherDescriptor>
  buildMatcherCtor(SourceRange NameRange, ArrayRef<ParserValue> Args,
                   Diagnostics *Error) const {
    return {};
  }

  /// Returns whether the matcher is variadic. Variadic matchers can take any
  /// number of arguments, but they must be of the same type.
  virtual bool isVariadic() const = 0;

  /// Returns the number of arguments accepted by the matcher if not variadic.
  virtual unsigned getNumArgs() const = 0;

  /// Given that the matcher is being converted to type \p ThisKind, append the
  /// set of argument types accepted for argument \p ArgNo to \p ArgKinds.
  virtual void getArgKinds(unsigned ArgNo,
                           std::vector<ArgKind> &ArgKinds) const = 0;
};

class FixedArgCountMatcherDescriptor : public MatcherDescriptor {
public:
  using MarshallerType = VariantMatcher (*)(void (*Func)(),
                                            StringRef MatcherName,
                                            SourceRange NameRange,
                                            ArrayRef<ParserValue> Args,
                                            Diagnostics *Error);

  /// \param Marshaller Function to unpack the arguments and call \c Func
  /// \param Func Matcher construct function. This is the function that
  ///   compile-time matcher expressions would use to create the matcher.
  FixedArgCountMatcherDescriptor(MarshallerType Marshaller, void (*Func)(),
                                 StringRef MatcherName,
                                 ArrayRef<ArgKind> ArgKinds)
      : Marshaller(Marshaller), Func(Func), MatcherName(MatcherName),
        ArgKinds(ArgKinds.begin(), ArgKinds.end()) {}

  VariantMatcher create(SourceRange NameRange, ArrayRef<ParserValue> Args,
                        Diagnostics *Error) const override {
    return Marshaller(Func, MatcherName, NameRange, Args, Error);
  }

  bool isVariadic() const override { return false; }
  unsigned getNumArgs() const override { return ArgKinds.size(); }

  void getArgKinds(unsigned ArgNo, std::vector<ArgKind> &Kinds) const override {
    Kinds.push_back(ArgKinds[ArgNo]);
  }

private:
  const MarshallerType Marshaller;
  void (*const Func)();
  const StringRef MatcherName;
  const std::vector<ArgKind> ArgKinds;
};

// Convert the return values of the functions into a VariantMatcher.
//
// There are 2 cases right now: The return value is a Matcher<T> or is a
// polymorphic matcher. For the former, we just construct the VariantMatcher.
// For the latter, we instantiate all the possible Matcher<T> of the poly
// matcher.
inline VariantMatcher outvalueToVariantMatcher(DynMatcher Matcher) {
  return VariantMatcher::SingleMatcher(Matcher);
}

/// Variadic marshaller function.
template <typename ResultT, typename ArgT,
          ResultT (*Func)(ArrayRef<const ArgT *>)>
VariantMatcher
variadicMatcherDescriptor(StringRef MatcherName, SourceRange NameRange,
                          ArrayRef<ParserValue> Args, Diagnostics *Error) {
  SmallVector<ArgT *, 8> InnerArgsPtr;
  InnerArgsPtr.resize_for_overwrite(Args.size());
  SmallVector<ArgT, 8> InnerArgs;
  InnerArgs.reserve(Args.size());

  for (size_t i = 0, e = Args.size(); i != e; ++i) {
    using ArgTraits = ArgTypeTraits<ArgT>;

    const ParserValue &Arg = Args[i];
    const VariantValue &Value = Arg.Value;
    if (!ArgTraits::hasCorrectType(Value)) {
      Error->addError(Arg.Range, Error->ET_RegistryWrongArgType)
          << (i + 1) << ArgTraits::getKind().asString()
          << Value.getTypeAsString();
      return {};
    }
    if (!ArgTraits::hasCorrectValue(Value)) {
      if (std::optional<std::string> BestGuess =
              ArgTraits::getBestGuess(Value)) {
        Error->addError(Arg.Range, Error->ET_RegistryUnknownEnumWithReplace)
            << i + 1 << Value.getString() << *BestGuess;
      } else if (Value.isString()) {
        Error->addError(Arg.Range, Error->ET_RegistryValueNotFound)
            << Value.getString();
      } else {
        // This isn't ideal, but it's better than reporting an empty string as
        // the error in this case.
        Error->addError(Arg.Range, Error->ET_RegistryWrongArgType)
            << (i + 1) << ArgTraits::getKind().asString()
            << Value.getTypeAsString();
      }
      return {};
    }
    assert(InnerArgs.size() < InnerArgs.capacity());
    InnerArgs.emplace_back(ArgTraits::get(Value));
    InnerArgsPtr[i] = &InnerArgs[i];
  }
  return outvalueToVariantMatcher(
      *DynMatcher::constructDynMatcherFromMatcherFn(Func(InnerArgsPtr)));
}

// Helper function to check if argument count matches expected count
inline bool checkArgCount(SourceRange NameRange, size_t expectedArgCount,
                          ArrayRef<ParserValue> Args, Diagnostics *Error) {
  if (Args.size() != expectedArgCount) {
    Error->addError(NameRange, Error->ET_RegistryWrongArgCount)
        << expectedArgCount << Args.size();
    return false;
  }
  return true;
}

// Helper function for checking argument type
template <typename ArgType, size_t Index>
inline bool checkArgTypeAtIndex(StringRef MatcherName,
                                ArrayRef<ParserValue> Args,
                                Diagnostics *Error) {
  if (!ArgTypeTraits<ArgType>::hasCorrectType(Args[Index].Value)) {
    Error->addError(Args[Index].Range, Error->ET_RegistryWrongArgType)
        << MatcherName << Index + 1;
    return false;
  }
  return true;
}

// Marshaller function for fixed number of arguments
template <typename ReturnType, typename... ArgTypes, size_t... Is>
static VariantMatcher
matcherMarshallFixedImpl(void (*Func)(), StringRef MatcherName,
                         SourceRange NameRange, ArrayRef<ParserValue> Args,
                         Diagnostics *Error, std::index_sequence<Is...>) {
  using FuncType = ReturnType (*)(ArgTypes...);
  if (!checkArgCount(NameRange, sizeof...(ArgTypes), Args, Error)) {
    return VariantMatcher();
  }

  if ((... && checkArgTypeAtIndex<ArgTypes, Is>(MatcherName, Args, Error))) {
    ReturnType fnPointer = reinterpret_cast<FuncType>(Func)(
        ArgTypeTraits<ArgTypes>::get(Args[Is].Value)...);
    return outvalueToVariantMatcher(
        *DynMatcher::constructDynMatcherFromMatcherFn(fnPointer));
  } else {
    return VariantMatcher();
  }
}

template <typename ReturnType, typename... ArgTypes>
static VariantMatcher
matcherMarshallFixed(void (*Func)(), StringRef MatcherName,
                     SourceRange NameRange, ArrayRef<ParserValue> Args,
                     Diagnostics *Error) {
  return matcherMarshallFixedImpl<ReturnType, ArgTypes...>(
      Func, MatcherName, NameRange, Args, Error,
      std::index_sequence_for<ArgTypes...>{});
}

/// \brief Variadic operator marshaller function.
class VariadicOperatorMatcherDescriptor : public MatcherDescriptor {
public:
  using VarOp = DynMatcher::VariadicOperator;
  VariadicOperatorMatcherDescriptor(unsigned MinCount, unsigned MaxCount,
                                    VarOp varOp, StringRef MatcherName)
      : MinCount(MinCount), MaxCount(MaxCount), varOp(varOp),
        MatcherName(MatcherName) {}

  VariantMatcher create(SourceRange NameRange, ArrayRef<ParserValue> Args,
                        Diagnostics *Error) const override {
    if (Args.size() < MinCount || MaxCount < Args.size()) {
      Error->addError(NameRange, Error->ET_RegistryWrongArgCount)
          << Args.size();
      return VariantMatcher();
    }

    std::vector<VariantMatcher> InnerArgs;
    for (size_t i = 0, e = Args.size(); i != e; ++i) {
      const ParserValue &Arg = Args[i];
      const VariantValue &Value = Arg.Value;
      if (!Value.isMatcher()) {
        Error->addError(Arg.Range, Error->ET_RegistryWrongArgType)
            << (i + 1) << "Matcher<>" << Value.getTypeAsString();
        return VariantMatcher();
      }
      InnerArgs.push_back(Value.getMatcher());
    }
    return VariantMatcher::VariadicOperatorMatcher(varOp, std::move(InnerArgs));
  }

  bool isVariadic() const override { return true; }
  unsigned getNumArgs() const override { return 0; }

  void getArgKinds(unsigned ArgNo, std::vector<ArgKind> &Kinds) const override {
    Kinds.push_back(ArgKind(ArgKind::AK_Matcher));
  }

private:
  const unsigned MinCount;
  const unsigned MaxCount;
  const VarOp varOp;
  const StringRef MatcherName;
};

// Helper functions to select the appropriate marshaller functions.
// They detect the number of arguments, arguments types, and return type.

// Fixed number of arguments overload
template <typename ReturnType, typename... ArgTypes>
std::unique_ptr<MatcherDescriptor>
makeMatcherAutoMarshall(ReturnType (*Func)(ArgTypes...),
                        StringRef MatcherName) {
  std::vector<ArgKind> AKs = {ArgTypeTraits<ArgTypes>::getKind()...};
  return std::make_unique<FixedArgCountMatcherDescriptor>(
      matcherMarshallFixed<ReturnType, ArgTypes...>,
      reinterpret_cast<void (*)()>(Func), MatcherName, AKs);
}

// Variadic operator overload.
template <unsigned MinCount, unsigned MaxCount>
std::unique_ptr<MatcherDescriptor>
makeMatcherAutoMarshall(VariadicOperatorMatcherFunc<MinCount, MaxCount> Func,
                        StringRef MatcherName) {
  return std::make_unique<VariadicOperatorMatcherDescriptor>(
      MinCount, MaxCount, Func.Op, MatcherName);
}

} // namespace internal
} // namespace matcher
} // namespace query
} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERS_MARSHALLERS_H