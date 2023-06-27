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
#include "MatchersInternal.h"
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
  static bool is(const VariantValue &Value) { return Value.isString(); }
  static const StringRef &get(const VariantValue &Value) {
    return Value.getString();
  }
};

template <>
struct ArgTypeTraits<DynMatcher> {
  static bool is(const VariantValue &Value) { return Value.isMatcher(); }
  static DynMatcher get(const VariantValue &Value) {
    return Value.getMatcher();
  }
};

template <>
struct ArgTypeTraits<bool> {
  static bool is(const VariantValue &Value) { return Value.isBoolean(); }
  static bool get(const VariantValue &Value) { return Value.getBoolean(); }
};

template <>
struct ArgTypeTraits<double> {
  static bool is(const VariantValue &Value) { return Value.isDouble(); }
  static double get(const VariantValue &Value) { return Value.getDouble(); }
};

template <>
struct ArgTypeTraits<unsigned> {
  static bool is(const VariantValue &Value) { return Value.isUnsigned(); }
  static unsigned get(const VariantValue &Value) { return Value.getUnsigned(); }
};

// Generic MatcherCreate interface.
// Provides a run() method that constructs the matcher from the provided
// arguments.
class MatcherCreateCallback {
public:
  virtual ~MatcherCreateCallback() = default;
  virtual DynMatcher *run(const SourceRange &NameRange,
                          const ArrayRef<ParserValue> &Args,
                          Diagnostics *Error) const = 0;
};

// Simple callback implementation. Marshaller and function are provided.
//
// Marshaller: Function to unpack the arguments and call Func.
// Func: Matcher construct function. This is the function that
// compile-time matcher expressions would use to create the matcher.
template <typename MarshallerType, typename FuncType>
class FixedArgCountMatcherCreateCallback : public MatcherCreateCallback {
public:
  FixedArgCountMatcherCreateCallback(MarshallerType Marshaller, FuncType Func,
                                     StringRef MatcherName)
      : Marshaller(Marshaller), Func(Func), MatcherName(MatcherName) {}

  DynMatcher *run(const SourceRange &NameRange,
                  const ArrayRef<ParserValue> &Args,
                  Diagnostics *Error) const override {
    return Marshaller(Func, MatcherName, NameRange, Args, Error);
  }

private:
  const MarshallerType Marshaller;
  const FuncType Func;
  const StringRef MatcherName;
};

// Variadic marshaller function.
template <typename T>
class VariadicMatcherCreateCallback : public MatcherCreateCallback {
public:
  explicit VariadicMatcherCreateCallback(StringRef MatcherName)
      : MatcherName(MatcherName.str()) {}

  typedef DynMatcher DerivedMatcherType;

  DynMatcher *run(const SourceRange &NameRange,
                  const ArrayRef<ParserValue> &Args,
                  Diagnostics *Error) const override {
    std::vector<DerivedMatcherType> References;
    std::vector<const DerivedMatcherType *> InnerArgs(Args.size());
    for (std::size_t i = 0, e = Args.size(); i != e; ++i) {

      if (!ArgTypeTraits<DerivedMatcherType>::is(Args[i].Value)) {
        Error->addError(Args[i].Range, Error->ET_RegistryWrongArgType)
            << MatcherName << i + 1;
        return nullptr;
      }
      References.push_back(
          ArgTypeTraits<DerivedMatcherType>::get(Args[i].Value));
      InnerArgs[i] = &References.back();
    }
    return new DynMatcher(new VariadicMatcher<T>(References));
  }

private:
  const std::string MatcherName;
};

// Helper function to perform template argument deduction.
template <typename MarshallerType, typename FuncType>
auto *createMarshallerCallback(MarshallerType Marshaller, FuncType Func,
                               StringRef MatcherName) {
  return new FixedArgCountMatcherCreateCallback<MarshallerType, FuncType>(
      Marshaller, Func, MatcherName);
}

// Metafunction to normalize argument types.
// We need to remove the const& out of the function parameters to be able to
// find values on VariantValue.
template <typename T>
struct remove_const_ref
    : public std::remove_const<typename std::remove_reference<T>::type> {};

// 0-arg marshaller function.
template <typename T, typename ReturnType>
DynMatcher *matcherMarshall0(ReturnType (*Func)(), StringRef MatcherName,
                             const SourceRange &NameRange,
                             ArrayRef<ParserValue> Args, Diagnostics *Error) {
  if (Args.size() != 0) {
    Error->addError(NameRange, Error->ET_RegistryWrongArgCount)
        << 0 << Args.size();
    return nullptr;
  }
  ReturnType matcherFn = Func();
  MatcherInterface<T> *singleMatcher =
      new SingleMatcher<T, ReturnType>(matcherFn);
  return new DynMatcher(singleMatcher);
}

// 1-arg marshaller function.
template <typename T, typename ReturnType, typename InArgType1>
DynMatcher *matcherMarshall1(ReturnType (*Func)(InArgType1),
                             StringRef MatcherName,
                             const SourceRange &NameRange,
                             ArrayRef<ParserValue> Args, Diagnostics *Error) {
  typedef typename remove_const_ref<InArgType1>::type ArgType1;
  // TODO: Extract this into a separate function.
  if (Args.size() != 1) {
    Error->addError(NameRange, Error->ET_RegistryWrongArgCount)
        << 1 << Args.size();
    return nullptr;
  }
  if (!ArgTypeTraits<ArgType1>::is(Args[0].Value)) {
    Error->addError(Args[0].Range, Error->ET_RegistryWrongArgType)
        << MatcherName << 1;
    return nullptr;
  }
  ReturnType matcherFn = Func(ArgTypeTraits<ArgType1>::get(Args[0].Value));
  MatcherInterface<T> *singleMatcher =
      new SingleMatcher<T, ReturnType>(matcherFn);
  return new DynMatcher(singleMatcher);
}

// 2-arg marshaller function.
template <typename T, typename ReturnType, typename InArgType1,
          typename InArgType2>
DynMatcher *matcherMarshall1(ReturnType (*Func)(InArgType1, InArgType2),
                             StringRef MatcherName,
                             const SourceRange &NameRange,
                             ArrayRef<ParserValue> Args, Diagnostics *Error) {
  typedef typename remove_const_ref<InArgType1>::type ArgType1;
  typedef typename remove_const_ref<InArgType2>::type ArgType2;
  // TODO: Extract this into a separate function.
  if (Args.size() != 2) {
    Error->addError(NameRange, Error->ET_RegistryWrongArgCount)
        << 1 << Args.size();
    return nullptr;
  }
  if (!ArgTypeTraits<ArgType1>::is(Args[0].Value)) {
    Error->addError(Args[0].Range, Error->ET_RegistryWrongArgType)
        << MatcherName << 1;
    return nullptr;
  }
  if (!ArgTypeTraits<ArgType2>::is(Args[1].Value)) {
    Error->addError(Args[1].Range, Error->ET_RegistryWrongArgType)
        << MatcherName << 1;
    return nullptr;
  }
  ReturnType matcherFn = Func(ArgTypeTraits<ArgType1>::get(Args[0].Value),
                              ArgTypeTraits<ArgType2>::get(Args[1].Value));
  MatcherInterface<T> *singleMatcher =
      new SingleMatcher<T, ReturnType>(matcherFn);
  return new DynMatcher(singleMatcher);
}

// Helper functions to select the appropriate marshaller functions.
// They detects the number of arguments, arguments types and return type.

// 0-arg overload
template <typename T, typename ReturnType>
auto *makeMatcherAutoMarshall(ReturnType (*Func)(), StringRef MatcherName) {
  return createMarshallerCallback(matcherMarshall0<T, ReturnType>, Func,
                                  MatcherName);
}

// 1-arg overload
template <typename T, typename ReturnType, typename ArgType1>
auto *makeMatcherAutoMarshall(ReturnType (*Func)(ArgType1),
                              StringRef MatcherName) {
  return createMarshallerCallback(matcherMarshall1<T, ReturnType, ArgType1>,
                                  Func, MatcherName);
}

// 2-arg overload
template <typename T, typename ReturnType, typename ArgType1, typename ArgType2>
auto *makeMatcherAutoMarshall(ReturnType (*Func)(ArgType1, ArgType2),
                              StringRef MatcherName) {
  return createMarshallerCallback(
      matcherMarshall1<T, ReturnType, ArgType1, ArgType2>, Func, MatcherName);
}

// Variadic overload.
template <typename T, typename MatcherType>
auto *makeMatcherAutoMarshall(MatcherType Func, StringRef MatcherName) {
  return new VariadicMatcherCreateCallback<T>(MatcherName);
}

} // namespace internal
} // namespace matcher
} // namespace query
} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERS_MARSHALLERS_H