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
  virtual bool dynMatches(DynTypedNode &DynNode) = 0;
};

/// Generic interface for matchers on an AST node of type T.
///
/// Implement this if your matcher may need to inspect the children or
/// descendants of the node or bind matched nodes to names. If you are
/// writing a simple matcher that only inspects properties of the
/// current node and doesn't care about its children or descendants,
/// implement SingleNodeMatcherInterface instead.
template <typename T>
class MatcherInterface : public DynMatcherInterface {
public:
  /// Returns true if 'Node' can be matched.
  ///
  /// May bind 'Node' to an ID via 'Builder', or recurse into
  /// the AST via 'Finder'.
  virtual bool matches(T *Node) = 0;

  bool dynMatches(DynTypedNode &DynNode) override {
    return matches(DynNode.getUnchecked<T>());
  }
};

template <typename> class Matcher;

/// Matcher wraps a MatcherInterface implementation and provides a matches()
/// method that redirects calls to the underlying implementation.
class DynMatcher {
public:
  /// Takes ownership of the provided implementation pointer.
  template <typename T>
  DynMatcher(MatcherInterface<T> *Implementation) : SupportedKind(MLIRNodeKind::getFromNodeKind<T>()),
        RestrictKind(SupportedKind),Implementation(Implementation) {}

  /// Returns true if the matcher matches the given op.
  bool matches(DynTypedNode &DynNode) const { 
    if (RestrictKind.isSame(DynNode.getNodeKind()) &&
        Implementation->dynMatches(DynNode)) {
      return true;
    }
    return false; }

  DynMatcher *clone() const { return new DynMatcher(*this); }

private:
  MLIRNodeKind SupportedKind;

  /// A potentially stricter node kind.
  ///
  /// It allows to perform implicit and dynamic cast of matchers without
  /// needing to change \c Implementation.
  MLIRNodeKind RestrictKind;
  llvm::IntrusiveRefCntPtr<DynMatcherInterface> Implementation;
};

/// Wrapper of a MatcherInterface<T> *that allows copying.
///
/// A Matcher<Base> can be used anywhere a Matcher<Derived> is
/// required. This establishes an is-a relationship which is reverse
/// to the AST hierarchy. In other words, Matcher<T> is contravariant
/// with respect to T. The relationship is built via a type conversion
/// operator rather than a type hierarchy to be able to templatize the
/// type hierarchy instead of spelling it out.
template <typename T>
class Matcher {
public:
  /// Takes ownership of the provided implementation pointer.
  explicit Matcher(MatcherInterface<T> *Implementation)
      : Implementation(Implementation) {}
    
  /// Forwards the call to the underlying MatcherInterface<T> pointer.
  bool matches(T Node) {
    return Implementation.matches(DynTypedNode::create(Node));
  }
private:
  DynMatcher Implementation;
};

/// SingleMatcher takes a matcher function object and implements
/// MatcherInterface.
template <typename T, typename MatcherFn>
class SingleMatcher : public MatcherInterface<T> {
public:
  SingleMatcher(MatcherFn &matcherFn) : matcherFn(matcherFn) {}
  bool matches(T *Node) override {
    // TODO: Find a way to do this without manually checking the elementType
    // auto DynNode = DynTypedNode::create(Node);
    // return matcherFn.match(DynNode.template getUnchecked<T>());
    return matcherFn.match(Node);
  }

private:
  MatcherFn matcherFn;
};

/// VariadicMatcher takes a vector of Matchers and returns true if all Matchers
/// match the given operation.
class VariadicMatcher : public DynMatcherInterface {
public:
  VariadicMatcher(std::vector<DynMatcher> matchers) : matchers(matchers) {}

  bool dynMatches(DynTypedNode &DynNode) override {
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
