//===--- Marshallers.h - Generic matcher function marshallers -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Functions templates and classes to wrap matcher construct functions.
///
/// A collection of template function and classes that provide a generic
/// marshalling layer on top of matcher construct functions.
/// These are used by the registry to export all marshaller constructors with
/// the same generic interface.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_MATCHERS_DYNAMIC_MARSHALLERS_H
#define LLVM_CLANG_AST_MATCHERS_DYNAMIC_MARSHALLERS_H

#include <list>
#include <string>
#include <vector>

#include "mlir/IR/Matchers.h"
#include "MatchersInternal.h"
#include "VariantValue.h"
#include "clang/Basic/LLVM.h"
#include "llvm/Support/type_traits.h"

namespace mlir {
namespace query {
namespace matcher {

namespace internal {

/// \brief Helper template class to just from argument type to the right is/get
///   functions in VariantValue.
/// Used to verify and extract the matcher arguments below.
template <class T> struct ArgTypeTraits;
template <class T> struct ArgTypeTraits<const T &> : public ArgTypeTraits<T> {
};

template <> struct ArgTypeTraits<std::string> {
  static bool is(const VariantValue &Value) { return Value.isString(); }
  static const std::string &get(const VariantValue &Value) {
    return Value.getString();
  }
};

template <class T> struct ArgTypeTraits<Matcher > {
  static bool is(const VariantValue &Value) { return Value.isMatcher(); }
  static Matcher get(const VariantValue &Value) {
    return Value.getMatcher();
  }

};

/// \brief Generic MatcherCreate interface.
///
/// Provides a \c run() method that constructs the matcher from the provided
/// arguments.
class MatcherCreateCallback {
public:
  virtual ~MatcherCreateCallback() {}
  virtual Matcher *run( ArrayRef<ParserValue> Args) const = 0;
};

/// \brief Simple callback implementation. Marshaller and function are provided.
///
/// \param Marshaller Function to unpack the arguments and call \c Func
/// \param Func Matcher construct function. This is the function that
///   compile-time matcher expressions would use to create the matcher.
template <typename MarshallerType, typename FuncType>
class FixedArgCountMatcherCreateCallback : public MatcherCreateCallback {
public:
  FixedArgCountMatcherCreateCallback(MarshallerType Marshaller, FuncType Func,
                                     StringRef MatcherName)
      : Marshaller(Marshaller), Func(Func), MatcherName(MatcherName.str()) {}

  Matcher *run( ArrayRef<ParserValue> Args) const {
    return Marshaller(Func, MatcherName, Args);
  }

private:
  const MarshallerType Marshaller;
  const FuncType Func;
  const std::string MatcherName;
};

/// \brief Helper function to do template argument deduction.
template <typename MarshallerType, typename FuncType>
MatcherCreateCallback *
createMarshallerCallback(MarshallerType Marshaller, FuncType Func,
                         StringRef MatcherName) {
  return new FixedArgCountMatcherCreateCallback<MarshallerType, FuncType>(
      Marshaller, Func, MatcherName);
}

/// \brief Metafunction to normalize argument types.
///
/// We need to remove the const& out of the function parameters to be able to
/// find values on VariantValue.
template <typename T>
struct remove_const_ref :
    public llvm::remove_const<typename llvm::remove_reference<T>::type> {
};

/// \brief 0-arg marshaller function.
template <typename ReturnType>
Matcher *matcherMarshall0(ReturnType (*Func)(), StringRef MatcherName, ArrayRef<ParserValue> Args) {
  CHECK_ARG_COUNT(0);
  if (Args.size() != 0) {                                                  
    return NULL;                                                               
  }
  return Matcher(Func());
  //return Func().clone();
}

/// \brief 1-arg marshaller function.
template <typename ReturnType, typename InArgType1>
Matcher *matcherMarshall1(ReturnType (*Func)(InArgType1),
                                  StringRef MatcherName,
                                  ArrayRef<ParserValue> Args) {
  typedef typename remove_const_ref<InArgType1>::type ArgType1;
  if (Args.size() != 1) {
    return NULL;
  }
  if (!ArgTypeTraits<ArgType1>::is(Args[0].Value)) {
    return NULL;
  }
  return Matcher(Func(ArgTypeTraits<ArgType1>::get(Args[0].Value)));
  // TODO
  //.clone();
}

/// \brief TODO Variadic marshaller function.

/// Helper functions to select the appropriate marshaller functions.
/// They detects the number of arguments, arguments types and return type.

/// \brief 0-arg overload
template <typename ReturnType>
MatcherCreateCallback *makeMatcherAutoMarshall(ReturnType (*Func)(),
                                               StringRef MatcherName) {
  return createMarshallerCallback(matcherMarshall0<ReturnType>, Func,
                                  MatcherName);
}

/// \brief 1-arg overload
template <typename ReturnType, typename ArgType1>
MatcherCreateCallback *makeMatcherAutoMarshall(ReturnType (*Func)(ArgType1),
                                               StringRef MatcherName) {
  return createMarshallerCallback(matcherMarshall1<ReturnType, ArgType1>, Func,
                                  MatcherName);
}

}  // namespace internal
}  // namespace matcher
}  // namespace query
}  // namespace mlir

#endif  // LLVM_CLANG_AST_MATCHERS_DYNAMIC_MARSHALLERS_H