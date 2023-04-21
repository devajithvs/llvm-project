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

#include "MLIRTypeTraits.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"

#include "llvm/Support/Debug.h"
using llvm::dbgs;

#define DEBUG_TYPE "mlir-query"
#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
namespace query {
namespace matcher {

class DynMatcherInterface : public llvm::RefCountedBase<DynMatcherInterface> {
public:
  virtual ~DynMatcherInterface() = default;

  // Returns true if 'op' can be matched.
  virtual bool dynMatches(DynTypedNode &DynNode) = 0;
};

// Generic interface for matchers on an MLIR node of type T.
template <typename T>
class MatcherInterface : public DynMatcherInterface {
public:
  virtual bool matches(T Node) = 0;

  bool dynMatches(DynTypedNode &DynNode) override {
    return matches(DynNode.getUnchecked<T>());
  }
};

template <typename>
class Matcher;

// Matcher wraps a MatcherInterface implementation and provides a matches()
// method that redirects calls to the underlying implementation.
class DynMatcher {
public:
  // Takes ownership of the provided implementation pointer.
  template <typename T>
  DynMatcher(MatcherInterface<T> *Implementation)
      : SupportedKind(MLIRNodeKind::getFromNodeKind<T>()),
        RestrictKind(SupportedKind), Implementation(Implementation),
        ExtractFunction(false) {}

  bool matches(DynTypedNode &DynNode) const {
    return RestrictKind.isSame(DynNode.getNodeKind()) && Implementation->dynMatches(DynNode);
  }

  DynMatcher *clone() const { return new DynMatcher(*this); }

  void setExtract(bool extractFunction) { ExtractFunction = extractFunction; };

  bool getExtract() const { return ExtractFunction; };

private:
  MLIRNodeKind SupportedKind;

  // A potentially stricter node kind.
  // It allows to perform implicit and dynamic cast of matchers without
  // needing to change Implementation.
  MLIRNodeKind RestrictKind;
  llvm::IntrusiveRefCntPtr<DynMatcherInterface> Implementation;
  bool ExtractFunction;
};

// Wrapper of a MatcherInterface<T> *
template <typename T>
class Matcher {
public:
  // Takes ownership of the provided implementation pointer.
  Matcher(MatcherInterface<T> *Implementation)
      : Implementation(Implementation) {}

  // Forwards the call to the underlying MatcherInterface<T> pointer.
  bool matches(T Node) {
    return Implementation.matches(DynTypedNode::create(Node));
  }

private:
  DynMatcher Implementation;
};

// SingleMatcher takes a matcher function object and implements
// MatcherInterface.
template <typename T, typename MatcherFn>
class SingleMatcher : public MatcherInterface<T> {
public:
  SingleMatcher(MatcherFn &matcherFn) : matcherFn(matcherFn) {}
  bool matches(T Node) override { return matcherFn.match(Node); }

private:
  MatcherFn matcherFn;
};

// VariadicMatcher takes a vector of Matchers and returns true if all Matchers
// match the given operation.
template <typename T>
class VariadicMatcher : public MatcherInterface<T> {
public:
  VariadicMatcher(std::vector<DynMatcher> matchers) : matchers(matchers) {}

  bool matches(T Node) override {
    DynTypedNode DynNode = DynTypedNode::create(Node);
    return llvm::all_of(matchers, [&](const DynMatcher &matcher) {
      return matcher.matches(DynNode);
    });
  }

private:
  std::vector<DynMatcher> matchers;
};

// MatchFinder is used to find all operations that match a given matcher.
class MatchFinder {
public:
  // Returns all operations that match the given matcher.
  std::vector<DynTypedNode> getMatches(Operation *op,
                                       const DynMatcher *matcher) {
    std::vector<DynTypedNode> matches;
    op->walk([&](Operation *subOp) {
      DynTypedNode node = DynTypedNode::create(subOp);
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
