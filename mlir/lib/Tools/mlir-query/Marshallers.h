//===--- Marshallers.h - Generic matcher function marshallers -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains function templates and classes to wrap matcher construct
// functions. It provides a collection of template function and classes that
// present a generic marshalling layer on top of matcher construct functions.
// The registry uses these to export all marshaller constructors with a uniform
// interface. This mechanism takes inspiration from clang-query.
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

// Helper template class for jumping from argument type to the correct is/get
// functions in VariantValue. This is used for verifying and extracting the
// matcher arguments.
template <class T>
struct ArgTypeTraits;
template <class T>
struct ArgTypeTraits<const T &> : public ArgTypeTraits<T> {};

template <>
struct ArgTypeTraits<StringRef> {

  static bool hasCorrectType(const VariantValue &value) {
    return value.isString();
  }

  static const StringRef &get(const VariantValue &value) {
    return value.getString();
  }

  static ArgKind getKind() { return ArgKind(ArgKind::AK_String); }

  static std::optional<std::string> getBestGuess(const VariantValue &) {
    return std::nullopt;
  }
};

template <>
struct ArgTypeTraits<DynMatcher> {

  static bool hasCorrectType(const VariantValue &value) {
    return value.isMatcher();
  }

  static DynMatcher get(const VariantValue &value) {
    return *value.getMatcher().getDynMatcher();
  }

  static ArgKind getKind() { return ArgKind(ArgKind::AK_Matcher); }

  static std::optional<std::string> getBestGuess(const VariantValue &) {
    return std::nullopt;
  }
};

template <>
struct ArgTypeTraits<bool> {

  static bool hasCorrectType(const VariantValue &value) {
    return value.isBoolean();
  }

  static bool get(const VariantValue &value) { return value.getBoolean(); }

  static ArgKind getKind() { return ArgKind(ArgKind::AK_Boolean); }

  static std::optional<std::string> getBestGuess(const VariantValue &) {
    return std::nullopt;
  }
};

template <>
struct ArgTypeTraits<double> {

  static bool hasCorrectType(const VariantValue &value) {
    return value.isDouble();
  }

  static double get(const VariantValue &value) { return value.getDouble(); }

  static ArgKind getKind() { return ArgKind(ArgKind::AK_Double); }

  static std::optional<std::string> getBestGuess(const VariantValue &) {
    return std::nullopt;
  }
};

template <>
struct ArgTypeTraits<unsigned> {
  static bool hasCorrectType(const VariantValue &value) {
    return value.isUnsigned();
  }

  static unsigned get(const VariantValue &value) { return value.getUnsigned(); }

  static ArgKind getKind() { return ArgKind(ArgKind::AK_Unsigned); }

  static std::optional<std::string> getBestGuess(const VariantValue &) {
    return std::nullopt;
  }
};

// Interface for generic matcher descriptor.
// Offers a create() method that constructs the matcher from the provided
// arguments.
class MatcherDescriptor {
public:
  virtual ~MatcherDescriptor() = default;
  virtual VariantMatcher create(SourceRange nameRange,
                                const ArrayRef<ParserValue> args,
                                Diagnostics *error) const = 0;

  virtual bool isBuilderMatcher() const { return false; }

  virtual std::unique_ptr<MatcherDescriptor>
  buildMatcherCtor(SourceRange nameRange, ArrayRef<ParserValue> args,
                   Diagnostics *error) const {
    return {};
  }

  // If the matcher is variadic, it can take any number of arguments.
  virtual bool isVariadic() const = 0;

  // Returns the number of arguments accepted by the matcher if it's not
  // variadic.
  virtual unsigned getNumArgs() const = 0;

  // Append the set of argument types accepted for argument 'ArgNo' to
  // 'ArgKinds'.
  virtual void getArgKinds(unsigned argNo,
                           std::vector<ArgKind> &argKinds) const = 0;
};

class FixedArgCountMatcherDescriptor : public MatcherDescriptor {
public:
  using MarshallerType = VariantMatcher (*)(void (*func)(),
                                            StringRef matcherName,
                                            SourceRange nameRange,
                                            ArrayRef<ParserValue> args,
                                            Diagnostics *error);

  // Marshaller Function to unpack the arguments and call Func. Func is the Matcher construct function. This is the function that the matcher expressions would use to create the matcher.
  FixedArgCountMatcherDescriptor(MarshallerType marshaller, void (*func)(),
                                 StringRef matcherName,
                                 ArrayRef<ArgKind> argKinds)
      : marshaller(marshaller), func(func), matcherName(matcherName),
        argKinds(argKinds.begin(), argKinds.end()) {}

  VariantMatcher create(SourceRange nameRange, ArrayRef<ParserValue> args,
                        Diagnostics *error) const override {
    return marshaller(func, matcherName, nameRange, args, error);
  }

  bool isVariadic() const override { return false; }
  unsigned getNumArgs() const override { return argKinds.size(); }

  void getArgKinds(unsigned argNo, std::vector<ArgKind> &kinds) const override {
    kinds.push_back(argKinds[argNo]);
  }

private:
  const MarshallerType marshaller;
  void (*const func)();
  const StringRef matcherName;
  const std::vector<ArgKind> argKinds;
};

// Convert the return values of the functions into a VariantMatcher.
inline VariantMatcher outvalueToVariantMatcher(DynMatcher matcher) {
  return VariantMatcher::SingleMatcher(matcher);
}

/// Variadic marshaller function.
template <typename ResultT, typename ArgT,
          ResultT (*Func)(ArrayRef<const ArgT *>)>
VariantMatcher
variadicMatcherDescriptor(StringRef matcherName, SourceRange nameRange,
                          ArrayRef<ParserValue> args, Diagnostics *error) {
  SmallVector<ArgT *, 8> innerArgsPtr;
  innerArgsPtr.resize_for_overwrite(args.size());
  SmallVector<ArgT, 8> innerArgs;
  innerArgs.reserve(args.size());

  for (size_t i = 0, e = args.size(); i != e; ++i) {
    using ArgTraits = ArgTypeTraits<ArgT>;

    const ParserValue &arg = args[i];
    const VariantValue &value = arg.value;
    if (!ArgTraits::hasCorrectType(value)) {
      error->addError(arg.range, error->ET_RegistryWrongArgType)
          << (i + 1) << ArgTraits::getKind().asString()
          << value.getTypeAsString();
      return {};
    }
    if (!ArgTraits::hasCorrectValue(value)) {
      if (std::optional<std::string> bestGuess =
              ArgTraits::getBestGuess(value)) {
        error->addError(arg.range, error->ET_RegistryUnknownEnumWithReplace)
            << i + 1 << value.getString() << *bestGuess;
      } else if (value.isString()) {
        error->addError(arg.range, error->ET_RegistryValueNotFound)
            << value.getString();
      } else {
        // This isn't ideal, but it's better than reporting an empty string as
        // the error in this case.
        error->addError(arg.range, error->ET_RegistryWrongArgType)
            << (i + 1) << ArgTraits::getKind().asString()
            << value.getTypeAsString();
      }
      return {};
    }
    assert(innerArgs.size() < innerArgs.capacity());
    innerArgs.emplace_back(ArgTraits::get(value));
    innerArgsPtr[i] = &innerArgs[i];
  }
  return outvalueToVariantMatcher(
      *DynMatcher::constructDynMatcherFromMatcherFn(Func(innerArgsPtr)));
}

// Helper function to check if argument count matches expected count
inline bool checkArgCount(SourceRange nameRange, size_t expectedArgCount,
                          ArrayRef<ParserValue> args, Diagnostics *error) {
  if (args.size() != expectedArgCount) {
    error->addError(nameRange, error->ET_RegistryWrongArgCount)
        << expectedArgCount << args.size();
    return false;
  }
  return true;
}

// Helper function for checking argument type
template <typename ArgType, size_t Index>
inline bool checkArgTypeAtIndex(StringRef matcherName,
                                ArrayRef<ParserValue> args,
                                Diagnostics *error) {
  if (!ArgTypeTraits<ArgType>::hasCorrectType(args[Index].value)) {
    error->addError(args[Index].range, error->ET_RegistryWrongArgType)
        << matcherName << Index + 1;
    return false;
  }
  return true;
}

// Marshaller function for fixed number of arguments
template <typename ReturnType, typename... ArgTypes, size_t... Is>
static VariantMatcher
matcherMarshallFixedImpl(void (*func)(), StringRef matcherName,
                         SourceRange nameRange, ArrayRef<ParserValue> args,
                         Diagnostics *error, std::index_sequence<Is...>) {
  using FuncType = ReturnType (*)(ArgTypes...);
  if (!checkArgCount(nameRange, sizeof...(ArgTypes), args, error)) {
    return VariantMatcher();
  }

  if ((... && checkArgTypeAtIndex<ArgTypes, Is>(matcherName, args, error))) {
    ReturnType fnPointer = reinterpret_cast<FuncType>(func)(
        ArgTypeTraits<ArgTypes>::get(args[Is].value)...);
    return outvalueToVariantMatcher(
        *DynMatcher::constructDynMatcherFromMatcherFn(fnPointer));
  } else {
    return VariantMatcher();
  }
}

template <typename ReturnType, typename... ArgTypes>
static VariantMatcher
matcherMarshallFixed(void (*func)(), StringRef matcherName,
                     SourceRange nameRange, ArrayRef<ParserValue> args,
                     Diagnostics *error) {
  return matcherMarshallFixedImpl<ReturnType, ArgTypes...>(
      func, matcherName, nameRange, args, error,
      std::index_sequence_for<ArgTypes...>{});
}

// Variadic operator marshaller function.
class VariadicOperatorMatcherDescriptor : public MatcherDescriptor {
public:
  using VarOp = DynMatcher::VariadicOperator;
  VariadicOperatorMatcherDescriptor(unsigned minCount, unsigned maxCount,
                                    VarOp varOp, StringRef matcherName)
      : minCount(minCount), maxCount(maxCount), varOp(varOp),
        matcherName(matcherName) {}

  VariantMatcher create(SourceRange nameRange, ArrayRef<ParserValue> args,
                        Diagnostics *error) const override {
    if (args.size() < minCount || maxCount < args.size()) {
      error->addError(nameRange, error->ET_RegistryWrongArgCount)
          << args.size();
      return VariantMatcher();
    }

    std::vector<VariantMatcher> innerArgs;
    for (size_t i = 0, e = args.size(); i != e; ++i) {
      const ParserValue &arg = args[i];
      const VariantValue &value = arg.value;
      if (!value.isMatcher()) {
        error->addError(arg.range, error->ET_RegistryWrongArgType)
            << (i + 1) << "Matcher: " << value.getTypeAsString();
        return VariantMatcher();
      }
      innerArgs.push_back(value.getMatcher());
    }
    return VariantMatcher::VariadicOperatorMatcher(varOp, std::move(innerArgs));
  }

  bool isVariadic() const override { return true; }

  unsigned getNumArgs() const override { return 0; }

  void getArgKinds(unsigned argNo, std::vector<ArgKind> &kinds) const override {
    kinds.push_back(ArgKind(ArgKind::AK_Matcher));
  }

private:
  const unsigned minCount;
  const unsigned maxCount;
  const VarOp varOp;
  const StringRef matcherName;
};

// Helper functions to select the appropriate marshaller functions.
// They detect the number of arguments, arguments types, and return type.

// Fixed number of arguments overload
template <typename ReturnType, typename... ArgTypes>
std::unique_ptr<MatcherDescriptor>
makeMatcherAutoMarshall(ReturnType (*func)(ArgTypes...),
                        StringRef matcherName) {
  std::vector<ArgKind> argKinds = {ArgTypeTraits<ArgTypes>::getKind()...};
  return std::make_unique<FixedArgCountMatcherDescriptor>(
      matcherMarshallFixed<ReturnType, ArgTypes...>,
      reinterpret_cast<void (*)()>(func), matcherName, argKinds);
}

// Variadic operator overload.
template <unsigned MinCount, unsigned MaxCount>
std::unique_ptr<MatcherDescriptor>
makeMatcherAutoMarshall(VariadicOperatorMatcherFunc<MinCount, MaxCount> func,
                        StringRef matcherName) {
  return std::make_unique<VariadicOperatorMatcherDescriptor>(
      MinCount, MaxCount, func.varOp, matcherName);
}

} // namespace internal
} // namespace matcher
} // namespace query
} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERS_MARSHALLERS_H