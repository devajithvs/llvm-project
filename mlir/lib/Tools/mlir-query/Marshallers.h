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

#include "MatcherVariantValue.h"
#include "MatcherDiagnostics.h"
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
  
  static ArgKind getKind() {
    return ArgKind(ArgKind::AK_String);
  }

  static std::optional<std::string> getBestGuess(const VariantValue &) {
    return std::nullopt;
  }
};

template <>
struct ArgTypeTraits<DynMatcher> {

  static bool hasCorrectType(const VariantValue& Value) {
    return Value.isMatcher();
  }

  static DynMatcher get(const VariantValue &Value) {
    return *Value.getMatcher().getDynMatcher();
  }

  static ArgKind getKind() {
    return ArgKind(ArgKind::AK_Matcher);
  }

  static std::optional<std::string> getBestGuess(const VariantValue &) {
    return std::nullopt;
  }
};

template <>
struct ArgTypeTraits<bool> {

  static bool hasCorrectType(const VariantValue &Value) {
    return Value.isBoolean();
  }

  static bool get(const VariantValue &Value) {
    return Value.getBoolean();
  }

  static ArgKind getKind() {
    return ArgKind(ArgKind::AK_Boolean);
  }

  static std::optional<std::string> getBestGuess(const VariantValue &) {
    return std::nullopt;
  }
};

template <>
struct ArgTypeTraits<double> {
  
  static bool hasCorrectType(const VariantValue &Value) {
    return Value.isDouble();
  }

  static double get(const VariantValue &Value) {
    return Value.getDouble();
  }

  static ArgKind getKind() {
    return ArgKind(ArgKind::AK_Double);
  }

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

  static ArgKind getKind() {
    return ArgKind(ArgKind::AK_Unsigned);
  }

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
  // FIXME: We should provide the ability to constrain the output of this
  // function based on the types of other matcher arguments.
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
                                 StringRef MatcherName, ArrayRef<ArgKind> ArgKinds)
      : Marshaller(Marshaller), Func(Func), MatcherName(MatcherName), ArgKinds(ArgKinds.begin(), ArgKinds.end()) {}

  VariantMatcher create(SourceRange NameRange,
                  ArrayRef<ParserValue> Args,
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
  void (* const Func)();
  const StringRef MatcherName;
  const std::vector<ArgKind> ArgKinds;
};

// Helper methods to extract and merge all possible typed matchers
/// out of the polymorphic object.
template <class PolyMatcher>
static void mergePolyMatchers(const PolyMatcher &Poly,
                              std::vector<DynMatcher> &Out) {
  
  Out.push_back(DynMatcher(Poly));
  llvm::errs() << "recursice merge PolyMatcher begin" << "\n";
  mergePolyMatchers(Poly, Out);
}

// Convert the return values of the functions into a VariantMatcher.
//
// There are 2 cases right now: The return value is a Matcher<T> or is a
// polymorphic matcher. For the former, we just construct the VariantMatcher.
// For the latter, we instantiate all the possible Matcher<T> of the poly
// matcher.
inline VariantMatcher outvalueToVariantMatcher(DynMatcher Matcher) {
    llvm::errs() << "outvalueToVariantMatcher" << "\n";

  return VariantMatcher::SingleMatcher(Matcher);
}

template <typename T>
static VariantMatcher outvalueToVariantMatcher(const T &PolyMatcher) {
  // TODO: Refractor
    llvm::errs() << "outvalueToVariantMatcher PolyMatcher" << "\n";
  std::vector<DynMatcher> Matchers = {DynMatcher(PolyMatcher)};
  VariantMatcher Out = VariantMatcher::PolymorphicMatcher(std::move(Matchers));
  return Out;
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
          << (i + 1) << ArgTraits::getKind().asString() << Value.getTypeAsString();
      return {};
    }
    if (!ArgTraits::hasCorrectValue(Value)) {
      if (std::optional<std::string> BestGuess = ArgTraits::getBestGuess(Value)) {
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
  return outvalueToVariantMatcher(Func(InnerArgsPtr));
}

/// Matcher descriptor for variadic functions.
///
/// This class simply wraps a VariadicFunction with the right signature to export
/// it as a MatcherDescriptor.
/// This allows us to have one implementation of the interface for as many free
/// functions as we want, reducing the number of symbols and size of the
/// object file.
class VariadicFuncMatcherDescriptor : public MatcherDescriptor {
public:
  using RunFunc = VariantMatcher (*)(StringRef MatcherName,
                                     SourceRange NameRange,
                                     ArrayRef<ParserValue> Args,
                                     Diagnostics *Error);

  template <typename ResultT, typename ArgT, ResultT (*F)(ArrayRef<const ArgT *>)>
  VariadicFuncMatcherDescriptor(
      VariadicFunction<ResultT, ArgT, F> Func,
      StringRef MatcherName)
      : Func(&variadicMatcherDescriptor<ResultT, ArgT, F>),
        MatcherName(MatcherName),
        ArgsKind(ArgTypeTraits<ArgT>::getKind()) {}

  VariantMatcher create(SourceRange NameRange,
                        ArrayRef<ParserValue> Args,
                        Diagnostics *Error) const override {
    return Func(MatcherName, NameRange, Args, Error);
  }

  bool isVariadic() const override { return true; }
  unsigned getNumArgs() const override { return 0; }

private:
  const RunFunc Func;
  const StringRef MatcherName;
  const ArgKind ArgsKind;
};

// 0-arg marshaller function.
template <typename ReturnType>
static VariantMatcher matcherMarshall0(void (*Func)(), StringRef MatcherName,
                             SourceRange NameRange,
                             ArrayRef<ParserValue> Args, Diagnostics *Error) {
  using FuncType = ReturnType (*)();
  if (Args.size() != 0) {
    Error->addError(NameRange, Error->ET_RegistryWrongArgCount)
        << 0 << Args.size();
    return VariantMatcher();
  }
  ReturnType fnPointer = reinterpret_cast<FuncType>(Func)();
  return outvalueToVariantMatcher(*DynMatcher::constructDynMatcherFromMatcherFn(fnPointer));
}

// 1-arg marshaller function.
template <typename ReturnType, typename ArgType1>
static VariantMatcher matcherMarshall1(void (*Func)(), StringRef MatcherName,
                             SourceRange NameRange,
                             ArrayRef<ParserValue> Args, Diagnostics *Error) {
  using FuncType = ReturnType (*)(ArgType1);
  // TODO: Extract this into a separate function.
  if (Args.size() != 1) {
    Error->addError(NameRange, Error->ET_RegistryWrongArgCount)
        << 1 << Args.size();
    return VariantMatcher();
  }
  if (!ArgTypeTraits<ArgType1>::hasCorrectType(Args[0].Value)) {
    Error->addError(Args[0].Range, Error->ET_RegistryWrongArgType)
        << MatcherName << 1;
    return VariantMatcher();
  }
  ReturnType fnPointer = reinterpret_cast<FuncType>(Func)(ArgTypeTraits<ArgType1>::get(Args[0].Value));
  llvm::errs() << "Post cast 1-arg marshaller function\n";
  return outvalueToVariantMatcher(*DynMatcher::constructDynMatcherFromMatcherFn(fnPointer));
}

/// \brief 2-arg marshaller function.
template <typename ReturnType, typename ArgType1, typename ArgType2>
static VariantMatcher matcherMarshall2(void (*Func)(), StringRef MatcherName,
                                       SourceRange NameRange,
                                       ArrayRef<ParserValue> Args,
                                       Diagnostics *Error) {
  using FuncType = ReturnType (*)(ArgType1, ArgType2);
  if (Args.size() != 2) {
    Error->addError(NameRange, Error->ET_RegistryWrongArgCount)
        << 1 << Args.size();
    return VariantMatcher();
  }
  if (!ArgTypeTraits<ArgType1>::hasCorrectType(Args[0].Value)) {
    Error->addError(Args[0].Range, Error->ET_RegistryWrongArgType)
        << MatcherName << 1;
    return VariantMatcher();
  }
  if (!ArgTypeTraits<ArgType2>::hasCorrectType(Args[1].Value)) {
    Error->addError(Args[1].Range, Error->ET_RegistryWrongArgType)
        << MatcherName << 1;
    return VariantMatcher();
  }
  ReturnType fnPointer = reinterpret_cast<FuncType>(Func)(
      ArgTypeTraits<ArgType1>::get(Args[0].Value),
      ArgTypeTraits<ArgType2>::get(Args[1].Value));
  return outvalueToVariantMatcher(*DynMatcher::constructDynMatcherFromMatcherFn(fnPointer));
}

/// \brief Variadic operator marshaller function.
class VariadicOperatorMatcherDescriptor : public MatcherDescriptor {
public:
  using VarOp = DynMatcher::VariadicOperator;
  VariadicOperatorMatcherDescriptor(unsigned MinCount, unsigned MaxCount,
                                    VarOp varOp, StringRef MatcherName)
      : MinCount(MinCount), MaxCount(MaxCount), varOp(varOp),
        MatcherName(MatcherName) {}

  VariantMatcher create(SourceRange NameRange,
                                ArrayRef<ParserValue> Args,
                                Diagnostics *Error) const override {
    if (Args.size() < MinCount || MaxCount < Args.size()) {
      // TODO
      // const std::string MaxStr =
      //     (MaxCount == std::numeric_limits<unsigned>::max() ? "" : Twine(MaxCount));
      // Error->addError(NameRange, Error->ET_RegistryWrongArgCount)
      //     << ("(" + Twine(MinCount) + ", " + MaxStr + ")") << Args.size();
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
// They detect the number of arguments, arguments types and return type.

// 0-arg overload
template <typename ReturnType>
std::unique_ptr<MatcherDescriptor>
makeMatcherAutoMarshall(ReturnType (*Func)(), StringRef MatcherName) {
  return std::make_unique<FixedArgCountMatcherDescriptor>(
      matcherMarshall0<ReturnType>, reinterpret_cast<void (*)()>(Func),
      MatcherName, std::nullopt);
}

// 1-arg overload
template <typename ReturnType, typename ArgType1>
std::unique_ptr<MatcherDescriptor>
makeMatcherAutoMarshall(ReturnType (*Func)(ArgType1), StringRef MatcherName) {
  ArgKind AK = ArgTypeTraits<ArgType1>::getKind();
  return std::make_unique<FixedArgCountMatcherDescriptor>(
      matcherMarshall1<ReturnType, ArgType1>,
      reinterpret_cast<void (*)()>(Func), MatcherName, AK);
}

// 2-arg overload
template <typename ReturnType, typename ArgType1, typename ArgType2>
std::unique_ptr<MatcherDescriptor>
makeMatcherAutoMarshall(ReturnType (*Func)(ArgType1, ArgType2),
                        StringRef MatcherName) {
  ArgKind AKs[] = { ArgTypeTraits<ArgType1>::getKind(),
                    ArgTypeTraits<ArgType2>::getKind() };
  return std::make_unique<FixedArgCountMatcherDescriptor>(
      matcherMarshall2<ReturnType, ArgType1, ArgType2>,
      reinterpret_cast<void (*)()>(Func), MatcherName, AKs);
}

// Variadic overload.
template <typename ResultT, typename ArgT,
          ResultT (*Func)(ArrayRef<const ArgT *>)>
std::unique_ptr<MatcherDescriptor> makeMatcherAutoMarshall(
    VariadicFunction<ResultT, ArgT, Func> VarFunc,
    StringRef MatcherName) {
  return std::make_unique<VariadicFuncMatcherDescriptor>(VarFunc, MatcherName);
}

// Variadic operator overload.
template <unsigned MinCount, unsigned MaxCount>
std::unique_ptr<MatcherDescriptor> makeMatcherAutoMarshall(
    VariadicOperatorMatcherFunc<MinCount, MaxCount> Func,
    StringRef MatcherName) {
  return std::make_unique<VariadicOperatorMatcherDescriptor>(
      MinCount, MaxCount, Func.Op, MatcherName);
}

} // namespace internal
} // namespace matcher
} // namespace query
} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERS_MARSHALLERS_H