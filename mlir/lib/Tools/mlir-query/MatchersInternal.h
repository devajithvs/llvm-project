//===- MatchersInternal.h - Structural query framework ----------*- C++ -*-===//
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
// matches(Operation *op)
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

namespace mlir {
namespace query {
namespace matcher {

class MatcherInterface : public llvm::RefCountedBase<MatcherInterface> {
public:
  virtual ~MatcherInterface() = default;

  /// Returns true if 'op' can be matched.
  virtual bool matches(Operation *op) = 0;
};

typedef llvm::IntrusiveRefCntPtr<MatcherInterface> MatcherImplementation;

/// Matcher wraps a MatcherInterface implementation and provides a matches()
/// method that redirects calls to the underlying implementation.
class Matcher {
public:
  Matcher(MatcherInterface *Implementation) : Implementation(Implementation) {}

  /// Returns true if the matcher matches the given op.
  bool matches(Operation *op) const { return Implementation->matches(op); }

  Matcher *clone() const { return new Matcher(*this); }

private:
  MatcherImplementation Implementation;
};

/// SingleMatcher takes a matcher function object and implements
/// MatcherInterface.
template <typename T>
class SingleMatcher : public MatcherInterface {
public:
  SingleMatcher(T &matcherFn) : matcherFn(matcherFn) {}
  bool matches(Operation *op) override { return matcherFn.match(op); }

private:
  T matcherFn;
};

// static bool allofvariadicoperator(Operation *op,
// std::vector<Matcher> InnerMatchers) {
// return llvm::all_of(InnerMatchers, [&](const Matcher &InnerMatcher) {
// return InnerMatcher.matches(op);
// });
// }

/// VariadicMatcher takes a vector of Matchers and returns true if all Matchers
/// match the given operation.
class VariadicMatcher : public MatcherInterface {
public:
  VariadicMatcher(std::vector<Matcher> matchers) : matchers(matchers) {}

  bool matches(Operation *op) override {
    return llvm::all_of(
        matchers, [&](const Matcher &matcher) { return matcher.matches(op); });
  }

private:
  std::vector<Matcher> matchers;
};

/// MatchFinder is used to find all operations that match a given matcher.
class MatchFinder {
public:
  /// Returns all operations that match the given matcher.
  std::vector<Operation *> getMatches(Operation *op, const Matcher *matcher) {
    std::vector<Operation *> matches;
    op->walk([&](Operation *subOp) {
      if (matcher->matches(subOp))
        matches.push_back(subOp);
    });
    return matches;
  }
};

} // namespace matcher
} // namespace query
} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERS_MATCHERSINTERNAL_H
