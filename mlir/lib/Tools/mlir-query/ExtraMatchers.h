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
// - operation(args...): Matches an operation that matches all of the matchers
// in the vector `matchers`.
//
// - argument(innerMatcher, index): Matches an operation argument that matches
// `innerMatcher` at the given `index`.
//
// - usedBy(innerMatcher, hops, inclusive): Matches an operation that is used by
// an operation that matches `innerMatcher` `hops` hops away. If `inclusive` is
// true, also matches operations upto `hops` away.
//
// - definedBy(innerMatcher, hops, inclusive): Matches an operation that is
// defined by an operation that matches `innerMatcher` `hops` hops away. If
// `inclusive` is true, also matches operations upto `hops` away.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_MATCHERS_EXTRAMATCHERS_H
#define MLIR_TOOLS_MLIRQUERY_MATCHERS_EXTRAMATCHERS_H

#include "MatchersInternal.h"

namespace mlir {

namespace query {

namespace extramatcher {

namespace detail {

// TODO: Rename to AllOf
// OperationMatcher takes a vector of DynMatchers and returns true if all
// DynMatchers match the given operation.
struct OperationMatcher {
  OperationMatcher(std::vector<matcher::DynMatcher> matchers)
      : matchers(matchers) {}
  bool match(Operation *op) {
    matcher::DynTypedNode node = matcher::DynTypedNode::create(op);
    return llvm::all_of(matchers, [&](const matcher::DynMatcher &matcher) {
      return matcher.matches(node);
    });
  }
  std::vector<matcher::DynMatcher> matchers;
};

// ArgumentMatcher matches the operand of an operation at a specific index.
struct ArgumentMatcher {
  ArgumentMatcher(matcher::DynMatcher innerMatcher, unsigned index)
      : innerMatcher(innerMatcher), index(index) {}

  bool match(Operation *op) {
    if (op->getNumOperands() > index) {
      auto operand = op->getOperand(index);
      if (Operation *operandOp = operand.getDefiningOp()) {
        matcher::DynTypedNode node = matcher::DynTypedNode::create(operandOp);
        return innerMatcher.matches(node);
      }
    }
    return false;
  }

  matcher::DynMatcher innerMatcher;
  unsigned index;
};

// This matcher checks if an operation is used by another operation that matches
// the given inner matcher. It allows specifying the number of hops to follow in
// the use-def chain, and whether the chain should be inclusive or not.
struct UsedByMatcher {
  UsedByMatcher(matcher::DynMatcher innerMatcher, unsigned hops, bool inclusive)
      : innerMatcher(innerMatcher), hops(hops), inclusive(inclusive) {}

  bool recursiveMatch(Operation *op, unsigned tempHops) {
    if (tempHops == 0) {
      auto currentNode = matcher::DynTypedNode::create(op);
      return innerMatcher.matches(currentNode);
    }
    if (inclusive) {
      return llvm::any_of(op->getOperands(), [&](Value operand) {
        if (Operation *operandOp = operand.getDefiningOp()) {
          matcher::DynTypedNode node = matcher::DynTypedNode::create(operandOp);
          return innerMatcher.matches(node) ||
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
  bool match(Operation *op) { return recursiveMatch(op, hops); }
  matcher::DynMatcher innerMatcher;
  unsigned hops;
  bool inclusive;
};

// This matcher checks if an operation is defined by another operation that
// matches the given inner matcher. It allows specifying the number of hops to
// follow in the def-use chain, and whether the chain should be inclusive or
// not.
struct DefinedByMatcher {
  DefinedByMatcher(matcher::DynMatcher innerMatcher, unsigned hops,
                   bool inclusive)
      : innerMatcher(innerMatcher), hops(hops), inclusive(inclusive) {}

  bool recursiveMatch(Operation *op, unsigned tempHops) {
    if (tempHops == 0) {
      auto currentNode = matcher::DynTypedNode::create(op);
      return innerMatcher.matches(currentNode);
    }
    if (inclusive) {
      return llvm::any_of(op->getUsers(), [&](Operation *userOp) {
        auto userNode = matcher::DynTypedNode::create(userOp);
        return innerMatcher.matches(userNode) ||
               recursiveMatch(userOp, tempHops - 1);
      });
    } else {
      return llvm::any_of(op->getUsers(), [&](Operation *userOp) {
        return recursiveMatch(userOp, tempHops - 1);
      });
    }
  }

  bool match(Operation *op) { return recursiveMatch(op, hops); }
  matcher::DynMatcher innerMatcher;
  unsigned hops;
  bool inclusive;
};

} // namespace detail

// TODO: Rename to allOf()
inline detail::OperationMatcher operation(matcher::DynMatcher args...) {
  std::vector<matcher::DynMatcher> matchers({args});
  return detail::OperationMatcher(matchers);
}

inline detail::ArgumentMatcher hasArgument(matcher::DynMatcher innerMatcher,
                                           unsigned argIndex) {
  return detail::ArgumentMatcher(innerMatcher, argIndex);
}

inline detail::UsedByMatcher usedBy(matcher::DynMatcher innerMatcher) {
  return detail::UsedByMatcher(innerMatcher, 1, false);
}

inline detail::UsedByMatcher getUses(matcher::DynMatcher innerMatcher,
                                     unsigned hops) {
  return detail::UsedByMatcher(innerMatcher, hops, false);
}

inline detail::UsedByMatcher getAllUses(matcher::DynMatcher innerMatcher,
                                        unsigned hops) {
  return detail::UsedByMatcher(innerMatcher, hops, true);
}

inline detail::DefinedByMatcher definedBy(matcher::DynMatcher innerMatcher) {
  return detail::DefinedByMatcher(innerMatcher, 1, false);
}

inline detail::DefinedByMatcher getDefinitions(matcher::DynMatcher innerMatcher,
                                               unsigned hops) {
  return detail::DefinedByMatcher(innerMatcher, hops, false);
}

inline detail::DefinedByMatcher
getAllDefinitions(matcher::DynMatcher innerMatcher, unsigned hops) {
  return detail::DefinedByMatcher(innerMatcher, hops, true);
}

} // namespace extramatcher

} // namespace query

} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERS_EXTRAMATCHERS_H
