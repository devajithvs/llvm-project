//===- MatchersInternal.h - Structural query framework ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Implements the base layer of the matcher framework.
//
//  Matchers are methods that return a Matcher which provides a method
//  matches(Operation *op)
//
//  In general, matchers have two parts:
//  1. A function Matcher MatcherName(<arguments>) which returns a Matcher
//     based on the arguments.
//  2. An implementation of a class derived from MatcherInterface.
//
//  The matcher functions are defined in include/mlir/IR/Matchers.h.
//  This file contains the wrapper classes needed to construct matchers for
//  mlir-query.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_MATCHERS_MATCHERSINTERNAL_H
#define MLIR_TOOLS_MLIRQUERY_MATCHERS_MATCHERSINTERNAL_H

#include "mlir/IR/Matchers.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"

namespace mlir {
namespace query {
namespace matcher {

class MatcherInterface;
typedef llvm::IntrusiveRefCntPtr<MatcherInterface> MatcherImplementation;
class MatcherInterface : public llvm::RefCountedBase<MatcherInterface> {
public:
  virtual ~MatcherInterface() = default;

  /// \brief Returns true if 'op' can be matched.
  virtual bool matches(Operation *op) = 0;
};

/// It is constructed from a \c MatcherImplementation and redirects calls to
/// underlying implementation.
class Matcher {
public:
  Matcher(MatcherInterface *Implementation) : Implementation(Implementation) {}

  /// Returns true if the matcher matches the given \c op.
  bool matches(Operation *op) const { return Implementation->matches(op); }

  Matcher *clone() const { return new Matcher(*this); }

private:
  llvm::IntrusiveRefCntPtr<MatcherInterface> Implementation;
};

/// \brief Single matcher that takes the matcher as a template argument.
template <typename T>
class SingleMatcher : public MatcherInterface {
public:
  SingleMatcher(T &matcher) : Matcher(matcher) {}
  bool matches(Operation *op) override { return Matcher.match(op); }

  T Matcher;
};

class MatchFinder {
public:
  /// Contains all information for a given match.
  ///
  /// Every time a match is found, the MatchFinder will invoke the registered
  /// MatchCallback with a MatchResult containing information about the match.
  struct MatchResult {
    MatchResult(Operation *op);

    /// Contains the nodes bound on the current match.
    ///
    /// This allows user code to easily extract matched AST nodes.
    Operation *op;

    /// Utilities for interpreting the matched AST structures.
    /// @{
    const std::shared_ptr<llvm::SourceMgr> SourceMgr;
    /// @}
  };
  std::vector<Operation *> getMatches(Operation *f, const Matcher *matcher) {
    std::vector<Operation *> matches;
    f->walk([&matches, &matcher](Operation *op) {
      if (matcher->matches(op)) {
        matches.push_back(op);
      }
    });
    return matches;
  }
};

} // namespace matcher
} // namespace query
} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERS_MATCHERSINTERNAL_H
