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
  static StringRef asString() { return "String"; }
  static bool is(const VariantValue &Value) { return Value.isString(); }
  static const StringRef &get(const VariantValue &Value) {
    return Value.getString();
  }
};

template <>
struct ArgTypeTraits<DynMatcher> {
  static StringRef asString() { return "DynMatcher"; }
  static bool is(const VariantValue &Value) { return Value.isMatcher(); }
  static DynMatcher get(const VariantValue &Value) {
    return Value.getMatcher().getTypedMatcher();
  }
};

template <>
struct ArgTypeTraits<bool> {
  static StringRef asString() { return "Boolean"; }
  static bool is(const VariantValue &Value) { return Value.isBoolean(); }
  static bool get(const VariantValue &Value) { return Value.getBoolean(); }
};

template <>
struct ArgTypeTraits<double> {
  static StringRef asString() { return "Double"; }
  static bool is(const VariantValue &Value) { return Value.isDouble(); }
  static double get(const VariantValue &Value) { return Value.getDouble(); }
};

template <>
struct ArgTypeTraits<unsigned> {
  static StringRef asString() { return "Unsigned"; }
  static bool is(const VariantValue &Value) { return Value.isUnsigned(); }
  static unsigned get(const VariantValue &Value) { return Value.getUnsigned(); }
};

// Generic Matcher descriptor interface.
// Provides a create() method that constructs the matcher from the provided
// arguments.
class MatcherDescriptor {
public:
  virtual ~MatcherDescriptor() = default;
  virtual VariantMatcher create(const SourceRange &NameRange,
                          const ArrayRef<ParserValue> Args,
                          Diagnostics *Error) const = 0;
};

// Simple callback implementation. Marshaller and function are provided.
//
// This class wraps a function of arbitrary signature and a marshaller
/// The marshaller is in charge of taking the VariantValue arguments, checking
/// their types, unpacking them and calling the underlying function.
// TODO

// template <typename MarshallerType, typename FuncType>
// class FixedArgCountMatcherDescriptor : public MatcherDescriptor {
// public:
//   FixedArgCountMatcherDescriptor(MarshallerType Marshaller, FuncType Func,
//                                      StringRef MatcherName)
//       : Marshaller(Marshaller), Func(Func), MatcherName(MatcherName) {}

//   DynMatcher *create(const SourceRange &NameRange,
//                   const ArrayRef<ParserValue> &Args,
//                   Diagnostics *Error) const override {
//     return Marshaller(Func, MatcherName, NameRange, Args, Error);
//   }

// private:
//   const MarshallerType Marshaller;
//   const FuncType Func;
//   const StringRef MatcherName;
// };

class FixedArgCountMatcherDescriptor : public MatcherDescriptor {
public:
  typedef VariantMatcher (*MarshallerType)(void (*Func)(),
                                           StringRef MatcherName,
                                           const SourceRange &NameRange,
                                           ArrayRef<ParserValue> Args,
                                           Diagnostics *Error);

  /// \param Marshaller Function to unpack the arguments and call \c Func
  /// \param Func Matcher construct function. This is the function that
  ///   compile-time matcher expressions would use to create the matcher.
  FixedArgCountMatcherDescriptor(MarshallerType Marshaller, void (*Func)(),
                                 StringRef MatcherName)
      : Marshaller(Marshaller), Func(Func), MatcherName(MatcherName) {
        llvm::errs() << "MatcherDescriptor: " << (int*)this << ":func:" << (int*)Func
                    << "\n";
      }

  VariantMatcher create(const SourceRange &NameRange,
                  ArrayRef<ParserValue> Args,
                  Diagnostics *Error) const override {
    llvm::errs() << "Marshaller: " << (int*) Marshaller << ":func:" << (int*)Func << "\n";
    return Marshaller(Func, MatcherName, NameRange, Args, Error);
  }

private:
  const MarshallerType Marshaller;
  void (* const Func)();
  const StringRef MatcherName;
};

/// \brief Simple callback implementation. Free function is wrapped.
///
/// This class simply wraps a free function with the right signature to export
/// it as a MatcherDescriptor.
/// This allows us to have one implementation of the interface for as many free
/// functions as we want, reducing the number of symbols and size of the
/// object file.
class FreeFuncMatcherDescriptor : public MatcherDescriptor {
public:
  typedef VariantMatcher (*RunFunc)(StringRef MatcherName,
                                    const SourceRange &NameRange,
                                    ArrayRef<ParserValue> Args,
                                    Diagnostics *Error);

  FreeFuncMatcherDescriptor(RunFunc Func, StringRef MatcherName)
      : Func(Func), MatcherName(MatcherName) {}

  VariantMatcher create(const SourceRange &NameRange,
                        ArrayRef<ParserValue> Args, Diagnostics *Error) const override {
    return Func(MatcherName, NameRange, Args, Error);
  }

private:
  const RunFunc Func;
  const StringRef MatcherName;
};

// \brief Helper methods to extract and merge all possible typed matchers
/// out of the polymorphic object.
template <class PolyMatcher>
static void mergePolyMatchers(const PolyMatcher &Poly,
                              std::vector<DynMatcher> &Out) {
  
  Out.push_back(DynMatcher(Poly));
  llvm::errs() << "recursice merge PolyMatcher begin" << "\n";
  mergePolyMatchers(Poly, Out);
}

// \brief Convert the return values of the functions into a VariantMatcher.
//
// There are 2 cases right now: The return value is a Matcher<T> or is a
// polymorphic matcher. For the former, we just construct the VariantMatcher.
// For the latter, we instantiate all the possible Matcher<T> of the poly
// matcher.
static VariantMatcher outvalueToVariantMatcher(DynMatcher Matcher) {
    llvm::errs() << "outvalueToVariantMatcher" << "\n";

  return VariantMatcher::SingleMatcher(Matcher);
}

template <typename T>
static VariantMatcher outvalueToVariantMatcher(const T &PolyMatcher) {
  // TODO: Refractor
    llvm::errs() << "outvalueToVariantMatcher PolyMatcher" << "\n";
  std::vector<DynMatcher> Matchers = {DynMatcher(PolyMatcher)};
    llvm::errs() << "pre merge PolyMatcher" << "\n";
    llvm::errs() << "post merge PolyMatcher" << "\n";
  VariantMatcher Out = VariantMatcher::PolymorphicMatcher(Matchers);
    llvm::errs() << "post PolymorphicMatcher" << "\n";
  return Out;
}

// 0-arg marshaller function.
template <typename ReturnType>
static VariantMatcher matcherMarshall0(void (*Func)(), StringRef MatcherName,
                             const SourceRange &NameRange,
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
                             const SourceRange &NameRange,
                             ArrayRef<ParserValue> Args, Diagnostics *Error) {
  using FuncType = ReturnType (*)(ArgType1);
  // TODO: Extract this into a separate function.
  if (Args.size() != 1) {
    Error->addError(NameRange, Error->ET_RegistryWrongArgCount)
        << 1 << Args.size();
    return VariantMatcher();
  }
  if (!ArgTypeTraits<ArgType1>::is(Args[0].Value)) {
    Error->addError(Args[0].Range, Error->ET_RegistryWrongArgType)
        << MatcherName << 1;
    return VariantMatcher();
  }
  ReturnType fnPointer = reinterpret_cast<FuncType>(Func)(ArgTypeTraits<ArgType1>::get(Args[0].Value));
  return outvalueToVariantMatcher(*DynMatcher::constructDynMatcherFromMatcherFn(fnPointer));
}

/// \brief 2-arg marshaller function.
template <typename ReturnType, typename ArgType1, typename ArgType2>
static VariantMatcher matcherMarshall2(void (*Func)(), StringRef MatcherName,
                                       const SourceRange &NameRange,
                                       ArrayRef<ParserValue> Args,
                                       Diagnostics *Error) {
  using FuncType = ReturnType (*)(ArgType1, ArgType2);
  if (Args.size() != 2) {
    Error->addError(NameRange, Error->ET_RegistryWrongArgCount)
        << 1 << Args.size();
    return VariantMatcher();
  }
  if (!ArgTypeTraits<ArgType1>::is(Args[0].Value)) {
    Error->addError(Args[0].Range, Error->ET_RegistryWrongArgType)
        << MatcherName << 1;
    return VariantMatcher();
  }
  if (!ArgTypeTraits<ArgType2>::is(Args[1].Value)) {
    Error->addError(Args[1].Range, Error->ET_RegistryWrongArgType)
        << MatcherName << 1;
    return VariantMatcher();
  }
  ReturnType fnPointer = reinterpret_cast<FuncType>(Func)(
      ArgTypeTraits<ArgType1>::get(Args[0].Value),
      ArgTypeTraits<ArgType2>::get(Args[1].Value));
  return outvalueToVariantMatcher(*DynMatcher::constructDynMatcherFromMatcherFn(fnPointer));
}

/// \brief Variadic marshaller function.
template <typename ResultT, typename ArgT,
          ResultT (*Func)(ArrayRef<const ArgT *>)>
VariantMatcher
variadicMatcherDescriptor(StringRef MatcherName, const SourceRange &NameRange,
                          ArrayRef<ParserValue> Args, Diagnostics *Error) {
  ArgT **InnerArgs = new ArgT *[Args.size()]();

  bool HasError = false;
  for (size_t i = 0, e = Args.size(); i != e; ++i) {
    typedef ArgTypeTraits<ArgT> ArgTraits;
    const ParserValue &Arg = Args[i];
    const VariantValue &Value = Arg.Value;
    if (!ArgTraits::is(Value)) {
      Error->addError(Arg.Range, Error->ET_RegistryWrongArgType)
          << (i + 1) << ArgTraits::asString() << Value.getTypeAsString();
      HasError = true;
      break;
    }
    InnerArgs[i] = new ArgT(ArgTraits::get(Value));
  }

  VariantMatcher Out;
  if (!HasError) {
    llvm::errs() << "Culprit here: " << "\n";

    Out = outvalueToVariantMatcher(
        Func(ArrayRef<const ArgT *>(InnerArgs, Args.size())));
  }

  for (size_t i = 0, e = Args.size(); i != e; ++i) {
    delete InnerArgs[i];
  }
  delete[] InnerArgs;
  return Out;
}

/// \brief MatcherDescriptor that wraps multiple "overloads" of the same
///   matcher.
///
/// It will try every overload and generate appropriate errors for when none or
/// more than one overloads match the arguments.
class OverloadedMatcherDescriptor : public MatcherDescriptor {
public:
  OverloadedMatcherDescriptor(ArrayRef<MatcherDescriptor *> Callbacks)
      : Overloads(Callbacks) {}

  ~OverloadedMatcherDescriptor() override = default;

  virtual VariantMatcher create(const SourceRange &NameRange,
                                ArrayRef<ParserValue> Args,
                                Diagnostics *Error) const override {
    std::vector<VariantMatcher> Constructed;
    Diagnostics::OverloadContext Ctx(Error);
    for (size_t i = 0, e = Overloads.size(); i != e; ++i) {
      VariantMatcher SubMatcher = Overloads[i]->create(NameRange, Args, Error);
      if (!SubMatcher.isNull()) {
        Constructed.push_back(SubMatcher);
      }
    }

    if (Constructed.empty()) return VariantMatcher(); // No overload matched.
    // We ignore the errors if any matcher succeeded.
    Ctx.revertErrors();
    if (Constructed.size() > 1) {
      // More than one constructed. It is ambiguous.
      Error->addError(NameRange, Error->ET_RegistryAmbiguousOverload);
      return VariantMatcher();
    }
    return Constructed[0];
  }
private:
  std::vector<MatcherDescriptor *> Overloads;
};

/// \brief Variadic operator marshaller function.
class VariadicOperatorMatcherDescriptor : public MatcherDescriptor {
public:
  using VarOp = DynMatcher::VariadicOperator;
  VariadicOperatorMatcherDescriptor(unsigned MinCount, unsigned MaxCount,
                                    VarOp varOp, StringRef MatcherName)
      : MinCount(MinCount), MaxCount(MaxCount), varOp(varOp),
        MatcherName(MatcherName) {}

  VariantMatcher create(const SourceRange &NameRange,
                                ArrayRef<ParserValue> Args,
                                Diagnostics *Error) const override {
    if (Args.size() < MinCount || MaxCount < Args.size()) {
      // TODO
      // const std::string MaxStr =
      //     (MaxCount == UINT_MAX ? "" : Twine(MaxCount)).str();
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
auto makeMatcherAutoMarshall(ReturnType (*Func)(), StringRef MatcherName) {
  LLVM_DEBUG(DBGS() << "pre matcherMarshall0"
                    << "\n");
  llvm::errs() << "matcherMarshall0: " <<  ":func:" << (int*)Func << "\n";
  return new FixedArgCountMatcherDescriptor(matcherMarshall0<ReturnType>, reinterpret_cast<void (*)()>(Func), MatcherName);
}

// 1-arg overload
template <typename ReturnType, typename ArgType1>
auto makeMatcherAutoMarshall(ReturnType (*Func)(ArgType1),
                              StringRef MatcherName) {
LLVM_DEBUG(DBGS() << "pre matcherMarshall1"
                    << "\n");
  llvm::errs() << "matcherMarshall1: " <<  ":func:" << (int*)Func << "\n";
  return new FixedArgCountMatcherDescriptor(matcherMarshall1<ReturnType, ArgType1>, reinterpret_cast<void (*)()>(Func), MatcherName);
}

// 2-arg overload
template <typename ReturnType, typename ArgType1, typename ArgType2>
auto makeMatcherAutoMarshall(ReturnType (*Func)(ArgType1, ArgType2),
                              StringRef MatcherName) {
                                LLVM_DEBUG(DBGS() << "pre matcherMarshall2"
                    << "\n");
  llvm::errs() << "matcherMarshall2: " <<  ":func:" << (int*)Func << "\n";
  return new FixedArgCountMatcherDescriptor(matcherMarshall2<ReturnType, ArgType1, ArgType2>, reinterpret_cast<void (*)()>(Func), MatcherName);
}

// Variadic overload.
template <typename ResultT, typename ArgT, ResultT (*Func)(ArrayRef<const ArgT *>)>
auto makeMatcherAutoMarshall(VariadicFunction<ResultT, ArgT, Func> VarFunc, StringRef MatcherName) {
  llvm::errs() << "FreeFuncMatcherDescriptor: " <<  ":func:" << (int*)Func << "\n";
  return new FreeFuncMatcherDescriptor(&variadicMatcherDescriptor<ResultT, ArgT, Func>, MatcherName);
}

// Variadic operator overload.
template <unsigned MinCount, unsigned MaxCount>
auto makeMatcherAutoMarshall(VariadicOperatorMatcherFunc<MinCount, MaxCount> Func, StringRef MatcherName) {
  return new VariadicOperatorMatcherDescriptor(MinCount, MaxCount, Func.Op,
                                               MatcherName);
}

} // namespace internal
} // namespace matcher
} // namespace query
} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERS_MARSHALLERS_H