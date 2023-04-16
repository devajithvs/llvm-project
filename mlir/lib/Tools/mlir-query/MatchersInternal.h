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
#include "MLIRTypeTraits.h"

namespace mlir {
namespace query {
namespace matcher {

class DynMatcherInterface : public llvm::RefCountedBase<DynMatcherInterface> {
public:
  virtual ~DynMatcherInterface() = default;

  /// Returns true if 'op' can be matched.
  virtual bool matches(DynTypedNode &DynNode) = 0;
};

/// Matcher wraps a MatcherInterface implementation and provides a matches()
/// method that redirects calls to the underlying implementation.
class DynMatcher {
public:
  DynMatcher(DynMatcherInterface *Implementation) : Implementation(Implementation) {}

  /// Returns true if the matcher matches the given op.
  bool matches(DynTypedNode &DynNode) const { return Implementation->matches(DynNode); }

  DynMatcher *clone() const { return new DynMatcher(*this); }

private:
  llvm::IntrusiveRefCntPtr<DynMatcherInterface> Implementation;
};

/// SingleMatcher takes a matcher function object and implements
/// MatcherInterface.
template <typename T>
class SingleMatcher : public DynMatcherInterface {
public:
  SingleMatcher(T &matcherFn) : matcherFn(matcherFn) {}
  bool matches(DynTypedNode &DynNode) override {
    Operation* op = DynNode.getUnchecked<Operation>();
    return matcherFn.match(op);
  }

private:
  T matcherFn;
};

/// VariadicMatcher takes a vector of Matchers and returns true if all Matchers
/// match the given operation.
class VariadicMatcher : public DynMatcherInterface {
public:
  VariadicMatcher(std::vector<DynMatcher> matchers) : matchers(matchers) {}

  bool matches(DynTypedNode &DynNode) override {
    return llvm::all_of(
        matchers, [&](const DynMatcher &matcher) { return matcher.matches(DynNode); });
  }

private:
  std::vector<DynMatcher> matchers;
};

/// MatchFinder is used to find all operations that match a given matcher.
class MatchFinder {
public:
  /// Returns all operations that match the given matcher.
  std::vector<DynTypedNode> getMatches(Operation *op, const DynMatcher *matcher) {
    std::vector<DynTypedNode> matches;
    op->walk([&](Operation *subOp) {

      //const MLIRNodeKind node = MLIRNodeKind::getFromNode(*subOp);
      DynTypedNode node = DynTypedNode::create(*subOp);
      if (matcher->matches(node))
        matches.push_back(node);
    });
    return matches;
  }
};

} // namespace matcher
} // namespace query
} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERS_MATCHERSINTERNAL_H
