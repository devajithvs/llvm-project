//===- MatchersInternal.h - Structural query framework --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the base layer of the matcher framework.
//
// Matchers are methods that return a Matcher which provides a method
// match(Operation *op)
//
// In general, matchers have two parts:
// 1. A function Matcher MatcherName(<arguments>) which returns a Matcher
// based on the arguments.
// 2. An implementation of a class derived from MatcherInterface.
//
// The matcher functions are defined in include/mlir/IR/Matchers.h.
// This file contains the wrapper classes needed to construct matchers for
// mlir-query.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_TOOLS_MLIRQUERY_MATCHERS_MATCHERSINTERNAL_H
#define MLIR_TOOLS_MLIRQUERY_MATCHERS_MATCHERSINTERNAL_H

#include "mlir/IR/Matchers.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"

#include "llvm/Support/Debug.h"
using llvm::dbgs;

#define DEBUG_TYPE "mlir-query"
#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
namespace query {
namespace matcher {

// Generic interface for matchers on an MLIR operation.
class MatcherInterface : public llvm::RefCountedBase<MatcherInterface> {
public:
  virtual ~MatcherInterface() = default;

  virtual bool match(Operation *op) = 0;
};

// TODO: Rename singleMatcher to MatcherFnImpl
// SingleMatcher takes a matcher function object and implements
// MatcherInterface.
template <typename MatcherFn>
class SingleMatcher : public MatcherInterface {
public:
  SingleMatcher(MatcherFn &matcherFn) : matcherFn(matcherFn) {
    llvm::errs() << "Setting matcher function: " << (int*)&matcherFn << ":impl: " << (int*) this << "\n";
  }
  bool match(Operation *op) override { 
    llvm::errs() << "singleMatcher working?\n";
    return matcherFn.match(op); }

private:
  MatcherFn matcherFn;
};

class DynMatcher;

// TODO: Use a polymorphic matcher instead for this usecase
// VariadicMatcher takes a vector of Matchers and returns true if any Matchers
// match the given operation.
using VariadicOperatorFunction = bool (*)(
    Operation *op,
    ArrayRef<DynMatcher> InnerMatchers);

template <VariadicOperatorFunction Func>
class VariadicMatcher : public MatcherInterface {
public:
  VariadicMatcher(std::vector<DynMatcher> matchers) : matchers(matchers) {}

  bool match(Operation *op) override {
    return Func(op, matchers);
  }

private:
  std::vector<DynMatcher> matchers;
};

static bool AllOfVariadicOperator(Operation *op, ArrayRef<DynMatcher> InnerMatchers);
static bool EachOfVariadicOperator(Operation *op, ArrayRef<DynMatcher> InnerMatchers);
static bool AnyOfVariadicOperator(Operation *op, ArrayRef<DynMatcher> InnerMatchers);

// Matcher wraps a MatcherInterface implementation and provides a match()
// method that redirects calls to the underlying implementation.
class DynMatcher {
public:
  // Takes ownership of the provided implementation pointer.
  DynMatcher(MatcherInterface *Implementation)
      : Implementation(Implementation), ExtractFunction(false) {
        llvm::errs() << "Implementation init: " << (int*)Implementation << "\n";

      }

    /// Construct from a variadic function.
  enum VariadicOperator {
    /// Matches nodes for which all provided matchers match.
    VO_AllOf,

    /// Matches nodes for which at least one of the provided matchers
    /// matches.
    VO_AnyOf,

    /// Matches nodes for which at least one of the provided matchers
    /// matches, but doesn't stop at the first match.
    VO_EachOf,

    /// Matches any node but executes all inner matchers to find result
    /// bindings.
    VO_Optionally,

    /// Matches nodes that do not match the provided matcher.
    ///
    /// Uses the variadic matcher interface, but fails if
    /// InnerMatchers.size() != 1.
    VO_UnaryNot
  };

  static DynMatcher *constructVariadic(VariadicOperator Op, std::vector<DynMatcher> InnerMatchers) {
    switch (Op) {
    case VO_AllOf:
      return new DynMatcher(new VariadicMatcher<AllOfVariadicOperator>(std::move(InnerMatchers)));
    case VO_AnyOf:
      return new DynMatcher(new VariadicMatcher<AnyOfVariadicOperator>(std::move(InnerMatchers)));
    case VO_EachOf:
      return new DynMatcher(new VariadicMatcher<EachOfVariadicOperator>(std::move(InnerMatchers)));
    }
    llvm_unreachable("Invalid Op value.");
  };
  
  template <typename MatcherFn>
  static DynMatcher *constructDynMatcherFromMatcherFn(MatcherFn &matcherFn) {
    auto impl = new SingleMatcher<MatcherFn>(matcherFn);
    llvm::errs() << "Implementation static: " << (int*)impl << "\n";

    return new DynMatcher(impl);
  };

  bool match(Operation *op) const { 
    llvm::errs() << "DynMatcher match working\n";
    llvm::errs() << "Implementation match: " << (int*)Implementation << "\n";
    return Implementation->match(op); 
  }

  DynMatcher *clone() const { return new DynMatcher(*this); }

  void setExtract(bool extractFunction) { ExtractFunction = extractFunction; };
  void setFunctionName(StringRef functionName) { FunctionName = functionName; };

  bool getExtract() const { return ExtractFunction; };
  StringRef getFunctionName() const { return FunctionName; };

private:
  MatcherInterface *Implementation;
  bool ExtractFunction;
  StringRef FunctionName;
};

/// VariadicOperatorMatcher related types.
/// @{

/// Polymorphic matcher object that uses a \c
/// DynMatcher::VariadicOperator operator.
///
/// Input matchers can have any type (including other polymorphic matcher
/// types), and the actual Matcher<T> is generated on demand with an implicit
/// conversion operator.
template <typename... Ps> class VariadicOperatorMatcher {
public:
  VariadicOperatorMatcher(DynMatcher::VariadicOperator VariadicOp, Ps &&... Params)
      : VariadicOp(VariadicOp), Params(std::forward<Ps>(Params)...) {}

  operator DynMatcher() const & {
    return &DynMatcher::constructVariadic(
               VariadicOp, getMatchers(std::index_sequence_for<Ps...>()));
  }

  operator DynMatcher() && {
    return &DynMatcher::constructVariadic(
               VariadicOp, getMatchers(std::index_sequence_for<Ps...>()));
  }

private:
  // Helper method to unpack the tuple into a vector.
  template <std::size_t... Is>
  std::vector<DynMatcher> getMatchers(std::index_sequence<Is...>) const & {
    return {DynMatcher(std::get<Is>(Params))...};
  }

  template <std::size_t... Is>
  std::vector<DynMatcher> getMatchers(std::index_sequence<Is...>) && {
    return {DynMatcher(std::get<Is>(std::move(Params)))...};
  }

  const DynMatcher::VariadicOperator VariadicOp;
  std::tuple<Ps...> Params;
};

/// Overloaded function object to generate VariadicOperatorMatcher
///   objects from arbitrary matchers.
template <unsigned MinCount, unsigned MaxCount>
struct VariadicOperatorMatcherFunc {
  DynMatcher::VariadicOperator Op;

  template <typename... Ms>
  VariadicOperatorMatcher<Ms...> operator()(Ms &&... Ps) const {
    static_assert(MinCount <= sizeof...(Ms) && sizeof...(Ms) <= MaxCount,
                  "invalid number of parameters for variadic matcher");
    return VariadicOperatorMatcher<Ms...>(Op, std::forward<Ms>(Ps)...);
  }
};


/// Variadic function object.
///
/// Most of the functions below that use VariadicFunction could be implemented
/// using plain C++11 variadic functions, but the function object allows us to
/// capture it on the dynamic matcher registry.
template <typename ResultT, typename ArgT,
          ResultT (*Func)(ArrayRef<const ArgT *>)>
struct VariadicFunction {
  ResultT operator()() const { return Func(std::nullopt); }

  template <typename... ArgsT>
  ResultT operator()(const ArgT &Arg1, const ArgsT &... Args) const {
    return Execute(Arg1, static_cast<const ArgT &>(Args)...);
  }

  // We also allow calls with an already created array, in case the caller
  // already had it.
  ResultT operator()(ArrayRef<ArgT> Args) const {
    return Func(llvm::to_vector<8>(llvm::make_pointer_range(Args)));
  }

private:
  // Trampoline function to allow for implicit conversions to take place
  // before we make the array.
  template <typename... ArgsT> ResultT Execute(const ArgsT &... Args) const {
    const ArgT *const ArgsArray[] = {&Args...};
    return Func(ArrayRef<const ArgT *>(ArgsArray, sizeof...(ArgsT)));
  }
};

static bool AllOfVariadicOperator(Operation *op, ArrayRef<DynMatcher> InnerMatchers) {
  // allOf leads to one matcher for each alternative in the first
  // matcher combined with each alternative in the second matcher.
  // Thus, we can reuse the same Builder.
  for (size_t i = 0, e = InnerMatchers.size(); i != e; ++i) {
    if (!InnerMatchers[i].match(op))
      return false;
  }
  return true;
}

static bool EachOfVariadicOperator(Operation *op, ArrayRef<DynMatcher> InnerMatchers) {
  bool Matched = false;
  for (size_t i = 0, e = InnerMatchers.size(); i != e; ++i) {
    if (InnerMatchers[i].match(op)) {
      Matched = true;
      // TODO
      // Add match to result
      // Result.addMatch
    }
  }
  // Set builder to result
  return Matched;
}

static bool AnyOfVariadicOperator(Operation *op, ArrayRef<DynMatcher> InnerMatchers) {
  for (size_t i = 0, e = InnerMatchers.size(); i != e; ++i) {
    if (InnerMatchers[i].match(op)) {
      // TODO: Add match to results. 
      return true;
    }
  }
  return false;
}

// \brief \c MatcherInterface<T> implementation for an variadic operator.
// class VariadicOperatorMatcherInterface : public MatcherInterface {
// public:
//   VariadicOperatorMatcherInterface(DynMatcher::VariadicOperator Func,
//                                    ArrayRef<DynMatcher> InnerMatchers)
//     : Func(Func), InnerMatchers(InnerMatchers) {}
//   bool match(Operation *op) override {
//     return Func(op, InnerMatchers);
//   }
// private:
//   const DynMatcher::VariadicOperator Func;
//   const std::vector<DynMatcher> InnerMatchers;
// };


// /// \brief Creates a DynMatcher that matches if all inner matchers match.
// DynMatcher makeAllOfComposite(ArrayRef<DynMatcher> InnerMatchers) {
//   std::vector<DynMatcher> DynMatchers;
//   for (size_t i = 0, e = InnerMatchers.size(); i != e; ++i) {
//     DynMatchers.push_back(InnerMatchers[i]);
//   }
//   return DynMatcher(new VariadicOperatorMatcherInterface(AllOfVariadicOperator, DynMatchers));
// }

// const VariadicOperatorMatcherFunc<
//     2, std::numeric_limits<unsigned>::max()>
//     eachOf = {DynMatcher::VO_EachOf};
// const VariadicOperatorMatcherFunc<
//     2, std::numeric_limits<unsigned>::max()>
//     anyOf = {DynMatcher::VO_AnyOf};
// const VariadicOperatorMatcherFunc<
//     2, std::numeric_limits<unsigned>::max()>
//     allOf = {DynMatcher::VO_AllOf};

// MatchFinder is used to find all operations that match a given matcher.
class MatchFinder {
public:
  // Returns all operations that match the given matcher.
  std::vector<Operation *> getMatches(Operation *root, DynMatcher matcher) {
    std::vector<Operation *> matches;
    llvm::errs() << "pre walk\n";
    matcher.match(root);

    root->walk([&](Operation *subOp) {
      if (matcher.match(subOp))
        matches.push_back(subOp);
    });
    llvm::errs() << "post walk\n";

    for (auto op: matches) {
      op->dump();
    }

    return matches;
  }
};

} // namespace matcher
} // namespace query
} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERS_MATCHERSINTERNAL_H
