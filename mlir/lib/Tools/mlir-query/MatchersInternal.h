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
class MatcherInterface
    : public llvm::ThreadSafeRefCountedBase<MatcherInterface> {
public:
  virtual ~MatcherInterface() = default;

  virtual bool match(Operation *op) = 0;
};

// MatcherFnImpl takes a matcher function object and implements
// MatcherInterface.
template <typename MatcherFn>
class MatcherFnImpl : public MatcherInterface {
public:
  MatcherFnImpl(MatcherFn &matcherFn) : matcherFn(matcherFn) {}
  bool match(Operation *op) override { return matcherFn.match(op); }

private:
  MatcherFn matcherFn;
};

class DynMatcher;

// VariadicMatcher takes a vector of Matchers and returns true if any Matchers
// match the given operation.
using VariadicOperatorFunction = bool (*)(Operation *op,
                                          ArrayRef<DynMatcher> innerMatchers);

template <VariadicOperatorFunction Func>
class VariadicMatcher : public MatcherInterface {
public:
  VariadicMatcher(std::vector<DynMatcher> matchers) : matchers(matchers) {}

  bool match(Operation *op) override { return Func(op, matchers); }

private:
  std::vector<DynMatcher> matchers;
};

class IdMatcher : public MatcherInterface {
public:
  IdMatcher(StringRef id,
            llvm::IntrusiveRefCntPtr<MatcherInterface> innerMatcher)
      : id(id), innerMatcher(std::move(innerMatcher)) {}

  bool match(Operation *op) override {
    bool result = innerMatcher->match(op);
    // TODO: Set binding
    // if (result) // setBinding(ID, op);
    return result;
  }

private:
  const std::string id;
  const llvm::IntrusiveRefCntPtr<MatcherInterface> innerMatcher;
};

static bool allOfVariadicOperator(Operation *op,
                                  ArrayRef<DynMatcher> innerMatchers);
static bool anyOfVariadicOperator(Operation *op,
                                  ArrayRef<DynMatcher> innerMatchers);

// Matcher wraps a MatcherInterface implementation and provides a match()
// method that redirects calls to the underlying implementation.
class DynMatcher {
public:
  // Takes ownership of the provided implementation pointer.
  DynMatcher(MatcherInterface *implementation)
      : implementation(implementation) {}

  /// Construct from a variadic function.
  enum VariadicOperator {
    /// Matches nodes for which all provided matchers match.
    VO_AllOf,

    /// Matches nodes for which at least one of the provided matchers
    /// matches.
    VO_AnyOf
  };

  /// \c MatcherIDType supports operator< and provides strict weak ordering.
  using MatcherIDType = uint64_t;
  MatcherIDType getID() const {
    /// FIXME: Document the requirements this imposes on matcher
    /// implementations (no new() implementation_ during a Matches()).
    return reinterpret_cast<uint64_t>(implementation.get());
  }

  static DynMatcher *constructVariadic(VariadicOperator Op,
                                       std::vector<DynMatcher> innerMatchers) {
    switch (Op) {
    case VO_AllOf:
      return new DynMatcher(
          new VariadicMatcher<allOfVariadicOperator>(std::move(innerMatchers)));
    case VO_AnyOf:
      return new DynMatcher(
          new VariadicMatcher<anyOfVariadicOperator>(std::move(innerMatchers)));
    }
    llvm_unreachable("Invalid Op value.");
  };

  template <typename MatcherFn>
  static DynMatcher *constructDynMatcherFromMatcherFn(MatcherFn &matcherFn) {
    auto impl = new MatcherFnImpl<MatcherFn>(matcherFn);
    return new DynMatcher(impl);
  };

  std::optional<DynMatcher> tryBind(StringRef ID) const {
    // if (!AllowBind)
    //   return std::nullopt;
    auto Result = *this;
    Result.implementation = new IdMatcher(ID, std::move(Result.implementation));
    return std::move(Result);
  }

  bool match(Operation *op) const { return implementation->match(op); }

  DynMatcher *clone() const { return new DynMatcher(*this); }

  void setFunctionName(StringRef fnName) {
    functionName = fnName.str();
  };

  bool isExtract() const { return !functionName.empty(); };
  StringRef getFunctionName() const { return functionName; };

private:
  llvm::IntrusiveRefCntPtr<MatcherInterface> implementation;
  std::string functionName;
};

/// VariadicOperatorMatcher related types.
template <typename... Ps>
class VariadicOperatorMatcher {
public:
  VariadicOperatorMatcher(DynMatcher::VariadicOperator varOp,
                          Ps &&...params)
      : varOp(varOp), params(std::forward<Ps>(params)...) {}

  operator DynMatcher() const & {
    return &DynMatcher::constructVariadic(
        varOp, getMatchers(std::index_sequence_for<Ps...>()));
  }

  operator DynMatcher() && {
    return &DynMatcher::constructVariadic(
        varOp, getMatchers(std::index_sequence_for<Ps...>()));
  }

private:
  // Helper method to unpack the tuple into a vector.
  template <std::size_t... Is>
  std::vector<DynMatcher> getMatchers(std::index_sequence<Is...>) const & {
    return {DynMatcher(std::get<Is>(params))...};
  }

  template <std::size_t... Is>
  std::vector<DynMatcher> getMatchers(std::index_sequence<Is...>) && {
    return {DynMatcher(std::get<Is>(std::move(params)))...};
  }

  const DynMatcher::VariadicOperator varOp;
  std::tuple<Ps...> params;
};

/// Overloaded function object to generate VariadicOperatorMatcher
///   objects from arbitrary matchers.
template <unsigned MinCount, unsigned MaxCount>
struct VariadicOperatorMatcherFunc {
  DynMatcher::VariadicOperator varOp;

  template <typename... Ms>
  VariadicOperatorMatcher<Ms...> operator()(Ms &&...Ps) const {
    static_assert(MinCount <= sizeof...(Ms) && sizeof...(Ms) <= MaxCount,
                  "invalid number of parameters for variadic matcher");
    return VariadicOperatorMatcher<Ms...>(varOp, std::forward<Ms>(Ps)...);
  }
};

static bool allOfVariadicOperator(Operation *op,
                                  ArrayRef<DynMatcher> innerMatchers) {
  // allOf leads to one matcher for each alternative in the first
  // matcher combined with each alternative in the second matcher.
  // Thus, we can reuse the same Builder.
  for (const DynMatcher &innerMatcher : innerMatchers) {
    if (!innerMatcher.match(op))
      return false;
  }
  return true;
}

static bool anyOfVariadicOperator(Operation *op,
                                  ArrayRef<DynMatcher> innerMatchers) {
  for (const DynMatcher &innerMatcher : innerMatchers) {
    if (innerMatcher.match(op)) {
      return true;
    }
  }
  return false;
}

// MatchFinder is used to find all operations that match a given matcher.
class MatchFinder {
public:
  // Returns all operations that match the given matcher.
  std::vector<Operation *> getMatches(Operation *root, DynMatcher matcher) {
    std::vector<Operation *> matches;

    root->walk([&](Operation *subOp) {
      if (matcher.match(subOp))
        matches.push_back(subOp);
    });

    return matches;
  }
};

} // namespace matcher
} // namespace query
} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERS_MATCHERSINTERNAL_H
