//===- ExtraMatchers.h - Various common matchers --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides extra matchers that are very useful for mlir-query. The
// goal is to move this to include/mlir/IR/Matchers.h after the initial testing
// phase. The matchers in this file are:
//
// - operation(args...): Matches all of the matchers in the vector `matchers`.
//
// - argument(innerMatcher, index): Matches an operation argument that matches
// `innerMatcher` at the given `index`.
//
// - usedBy(innerMatcher, hops, inclusive): Matches an operation that is used by
// an operation that matches `innerMatcher` `hops` hops away. If `inclusive` is
// true, also matches operations up to `hops` away.
//
// - definedBy(innerMatcher, hops, inclusive): Matches an operation that is
// defined by an operation that matches `innerMatcher` `hops` hops away. If
// `inclusive` is true, also matches operations up to `hops` away.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_MATCHERS_EXTRAMATCHERS_H
#define MLIR_TOOLS_MLIRQUERY_MATCHERS_EXTRAMATCHERS_H

#include "MatchersInternal.h"

namespace mlir {

namespace query {

namespace extramatchers {

namespace detail {

// ArgumentMatcher matches the operand of an operation at a specific index.
template <typename Matcher>
struct ArgumentMatcher {
  ArgumentMatcher(Matcher innerMatcher, unsigned index)
      : innerMatcher(innerMatcher), index(index) {}

  bool match(Operation *op) const {
    if (op->getNumOperands() > index) {
      auto operand = op->getOperand(index);
      if (Operation *operandOp = operand.getDefiningOp()) {
        return innerMatcher.match(op);
      }
    }
    return false;
  }

  Matcher innerMatcher;
  unsigned index;
};

// This matcher checks if an operation is used by another operation that matches
// the given inner matcher. It allows specifying the number of hops to follow in
// the use-def chain, and whether the chain should be inclusive or not.
template <typename Matcher>
struct UsesMatcher {
  UsesMatcher(Matcher innerMatcher, unsigned hops, bool inclusive)
      : innerMatcher(innerMatcher), hops(hops), inclusive(inclusive) {}

  bool recursiveMatch(Operation *op, unsigned tempHops) const {
    if (tempHops == 0) {
      return innerMatcher.match(op);
    }
    if (inclusive) {
      return llvm::any_of(op->getOperands(), [&](Value operand) {
        if (Operation *operandOp = operand.getDefiningOp()) {
          return innerMatcher.match(operandOp) ||
                 recursiveMatch(operandOp, tempHops - 1);
        }
        return false;
      });
    } else {
      return llvm::any_of(op->getOperands(), [&](Value operand) {
        if (Operation *operandOp = operand.getDefiningOp()) {
          return recursiveMatch(operandOp, tempHops - 1);
        }
        return false;
      });
    }
  }

  bool match(Operation *op) const { return recursiveMatch(op, hops); }
  Matcher innerMatcher;
  unsigned hops;
  bool inclusive;
};

// This matcher checks if an operation is defined by another operation that
// matches the given inner matcher. It allows specifying the number of hops to
// follow in the def-use chain, and whether the chain should be inclusive or
// not.
template <typename Matcher>
struct DefinitionsMatcher {
  DefinitionsMatcher(Matcher innerMatcher, unsigned hops, bool inclusive)
      : innerMatcher(innerMatcher), hops(hops), inclusive(inclusive) {}

  bool recursiveMatch(Operation *op, unsigned tempHops) const {
    if (tempHops == 0) {
      return innerMatcher.match(op);
    }
    if (inclusive) {
      return llvm::any_of(op->getUsers(), [&](Operation *userOp) {
        return innerMatcher.match(userOp) ||
               recursiveMatch(userOp, tempHops - 1);
      });
    } else {
      return llvm::any_of(op->getUsers(), [&](Operation *userOp) {
        return recursiveMatch(userOp, tempHops - 1);
      });
    }
  }

  bool match(Operation *op) const { return recursiveMatch(op, hops); }
  Matcher innerMatcher;
  unsigned hops;
  bool inclusive;
};

} // namespace detail

const matcher::VariadicOperatorMatcherFunc<1,
                                           std::numeric_limits<unsigned>::max()>
    anyOf = {matcher::DynMatcher::VO_AnyOf};
const matcher::VariadicOperatorMatcherFunc<1,
                                           std::numeric_limits<unsigned>::max()>
    allOf = {matcher::DynMatcher::VO_AllOf};

inline detail::ArgumentMatcher<matcher::DynMatcher>
hasArgument(matcher::DynMatcher innerMatcher, unsigned argIndex) {
  return detail::ArgumentMatcher<matcher::DynMatcher>(innerMatcher, argIndex);
}

inline detail::UsesMatcher<matcher::DynMatcher>
uses(matcher::DynMatcher innerMatcher) {
  return detail::UsesMatcher<matcher::DynMatcher>(innerMatcher, 1, false);
}

inline detail::UsesMatcher<matcher::DynMatcher>
getUses(matcher::DynMatcher innerMatcher, unsigned hops) {
  return detail::UsesMatcher<matcher::DynMatcher>(innerMatcher, hops, false);
}

inline detail::UsesMatcher<matcher::DynMatcher>
getAllUses(matcher::DynMatcher innerMatcher, unsigned hops) {
  return detail::UsesMatcher<matcher::DynMatcher>(innerMatcher, hops, true);
}

inline detail::DefinitionsMatcher<matcher::DynMatcher>
definedBy(matcher::DynMatcher innerMatcher) {
  return detail::DefinitionsMatcher<matcher::DynMatcher>(innerMatcher, 1,
                                                         false);
}

inline detail::DefinitionsMatcher<matcher::DynMatcher>
getDefinitions(matcher::DynMatcher innerMatcher, unsigned hops) {
  return detail::DefinitionsMatcher<matcher::DynMatcher>(innerMatcher, hops,
                                                         false);
}

inline detail::DefinitionsMatcher<matcher::DynMatcher>
getAllDefinitions(matcher::DynMatcher innerMatcher, unsigned hops) {
  return detail::DefinitionsMatcher<matcher::DynMatcher>(innerMatcher, hops,
                                                         true);
}

} // namespace extramatchers

} // namespace query

} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERS_EXTRAMATCHERS_H
