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

template <typename NodeType>
class MatcherInterface : public llvm::RefCountedBase<MatcherInterface<NodeType>> {
public:
  virtual ~MatcherInterface() = default;

  /// Returns true if 'node' can be matched.
  virtual bool matches(NodeType node) = 0;
};

/// Matcher wraps a MatcherInterface implementation and provides a matches()
/// method that redirects calls to the underlying implementation.
template <typename NodeType>
class Matcher {
public:
  Matcher(MatcherInterface<NodeType> *Implementation) : Implementation(Implementation) {}

  /// Returns true if the matcher matches the given op.
  bool matches(NodeType node) const { return Implementation->matches(node); }

  Matcher *clone() const { return new Matcher<NodeType>(*this); }

private:
  llvm::IntrusiveRefCntPtr<MatcherInterface<NodeType>> Implementation;
};

/// A convenient helper for creating a Matcher<T> without specifying
/// the template type argument.
template <typename T>
inline Matcher<T> makeMatcher(MatcherInterface<T> *Implementation) {
  return Matcher<T>(Implementation);
}

/// SingleMatcher takes a matcher function object and implements
/// MatcherInterface.
template <typename T, typename NodeType>
class SingleMatcher : public MatcherInterface<NodeType> {
public:
  SingleMatcher(T &matcherFn) : matcherFn(matcherFn) {}
  bool matches(NodeType node) override { return matcherFn.match(node); }

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
template <typename NodeType>
class VariadicMatcher : public MatcherInterface<NodeType> {
public:
  VariadicMatcher(std::vector<Matcher<NodeType>> matchers) : matchers(matchers) {}

  bool matches(NodeType node) override {
    return llvm::all_of(
        matchers, [&](const Matcher<NodeType> &matcher) { return matcher.matches(node); });
  }

private:
  std::vector<Matcher<NodeType>> matchers;
};

/// MatchFinder is used to find all operations that match a given matcher.
class MatchFinder {
public:
  /// Returns all operations that match the given matcher.
  std::vector<Operation *> getMatches(Operation *rootOp, const Matcher<Operation*> *matcher) {
    std::vector<Operation *> matches;
    rootOp->walk([&](Operation *subOp) {
      if (matcher->matches(subOp))
        matches.push_back(subOp);
    });
    return matches;
  }

  /// Returns all values that match the given matcher.
  std::vector<Value> getMatches(Operation *rootOp, const Matcher<Value> *matcher) {
    std::vector<Value> matches;
    rootOp->walk([&](Operation *subOp) {
      for (Value value : subOp->getResults()) {
        if (matcher->matches(value))
          matches.push_back(value);
      }
      });
    return matches;
  }
};

} // namespace matcher
} // namespace query
} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERS_MATCHERSINTERNAL_H
