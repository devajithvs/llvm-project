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

#include <list>
#include <string>
#include <vector>

#include "MatchersInternal.h"
#include "VariantValue.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/type_traits.h"

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
struct ArgTypeTraits<DynTypedMatcher> {
  static bool is(const VariantValue &Value) { return Value.isMatcher(); }
  static DynTypedMatcher get(const VariantValue &Value) { return Value.getMatcher(); }
};

// Generic MatcherCreate interface.
// Provides a run() method that constructs the matcher from the provided
// arguments.
class MatcherCreateCallback {
public:
  virtual ~MatcherCreateCallback() = default;
  virtual DynTypedMatcher *run(const SourceRange &NameRange, ArrayRef<ParserValue> Args,
                       Diagnostics *Error) const = 0;
};

// Simple callback implementation. Marshaller and function are provided.
//
// Marshaller: Function to unpack the arguments and call Func.
// Func: DynTypedMatcher construct function. This is the function that
// compile-time matcher expressions would use to create the matcher.
template <typename MarshallerType, typename FuncType>
class FixedArgCountMatcherCreateCallback : public MatcherCreateCallback {
public:
  FixedArgCountMatcherCreateCallback(MarshallerType Marshaller, FuncType Func,
                                     StringRef MatcherName)
      : Marshaller(Marshaller), Func(Func), MatcherName(MatcherName) {}

  DynTypedMatcher *run(const SourceRange &NameRange, ArrayRef<ParserValue> Args,
               Diagnostics *Error) const override {
    return Marshaller(Func, MatcherName, NameRange, Args, Error);
  }

private:
  const MarshallerType Marshaller;
  const FuncType Func;
  const StringRef MatcherName;
};

/// Variadic marshaller function.
class VariadicMatcherCreateCallback : public MatcherCreateCallback {
public:
  explicit VariadicMatcherCreateCallback(StringRef MatcherName)
      : MatcherName(MatcherName.str()) {}

  typedef DynTypedMatcher DerivedMatcherType;

  DynTypedMatcher *run(const SourceRange &NameRange, ArrayRef<ParserValue> Args,
               Diagnostics *Error) const override {
    std::vector<DerivedMatcherType> References;
    std::vector<const DerivedMatcherType *> InnerArgs(Args.size());
    for (size_t i = 0, e = Args.size(); i != e; ++i) {

      if (!ArgTypeTraits<DerivedMatcherType>::is(Args[i].Value)) {
        Error->addError(Args[i].Range, Error->ET_RegistryWrongArgType)
            << MatcherName << i + 1;
        return NULL;
      }
      References.push_back(
          ArgTypeTraits<DerivedMatcherType>::get(Args[i].Value));
      InnerArgs[i] = &References.back();
    }
    return new DynTypedMatcher(new VariadicMatcher(References));
  }

private:
  const std::string MatcherName;
};
// Helper function to perform template argument deduction.
template <typename MarshallerType, typename FuncType>
MatcherCreateCallback *createMarshallerCallback(MarshallerType Marshaller,
                                                FuncType Func,
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
template <typename ReturnType>
DynTypedMatcher *matcherMarshall0(ReturnType (*Func)(), StringRef MatcherName,
                          const SourceRange &NameRange,
                          ArrayRef<ParserValue> Args, Diagnostics *Error) {
  if (Args.size() != 0) {
    Error->addError(NameRange, Error->ET_RegistryWrongArgCount)
        << 0 << Args.size();
    return NULL;
  }
  ReturnType matcherFn = Func();
  MatcherInterface *singleMatcher = new SingleMatcher<ReturnType>(matcherFn);
  return new DynTypedMatcher(singleMatcher);
}

// 1-arg marshaller function.
template <typename ReturnType, typename InArgType1>
DynTypedMatcher *matcherMarshall1(ReturnType (*Func)(InArgType1), StringRef MatcherName,
                          const SourceRange &NameRange,
                          ArrayRef<ParserValue> Args, Diagnostics *Error) {
  typedef typename remove_const_ref<InArgType1>::type ArgType1;
  // TODO: Extract this into a separate function.
  if (Args.size() != 1) {
    Error->addError(NameRange, Error->ET_RegistryWrongArgCount)
        << 1 << Args.size();
    return NULL;
  }
  if (!ArgTypeTraits<ArgType1>::is(Args[0].Value)) {
    Error->addError(Args[0].Range, Error->ET_RegistryWrongArgType)
        << MatcherName << 1;
    return NULL;
  }
  ReturnType matcherFn = Func(ArgTypeTraits<ArgType1>::get(Args[0].Value));
  MatcherInterface *singleMatcher = new SingleMatcher<ReturnType>(matcherFn);
  return new DynTypedMatcher(singleMatcher);
}

/// TODO Variadic marshaller function.

// Helper functions to select the appropriate marshaller functions.
// They detects the number of arguments, arguments types and return type.

// 0-arg overload
template <typename ReturnType>
MatcherCreateCallback *makeMatcherAutoMarshall(ReturnType (*Func)(),
                                               StringRef MatcherName) {
  return createMarshallerCallback(matcherMarshall0<ReturnType>, Func,
                                  MatcherName);
}

// 1-arg overload
template <typename ReturnType, typename ArgType1>
MatcherCreateCallback *makeMatcherAutoMarshall(ReturnType (*Func)(ArgType1),
                                               StringRef MatcherName) {
  return createMarshallerCallback(matcherMarshall1<ReturnType, ArgType1>, Func,
                                  MatcherName);
}

/// Variadic overload.
template <typename MatcherType>
MatcherCreateCallback *makeMatcherAutoMarshall(MatcherType Func,
                                               StringRef MatcherName) {
  return new VariadicMatcherCreateCallback(MatcherName);
}

} // namespace internal
} // namespace matcher
} // namespace query
} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERS_MARSHALLERS_H