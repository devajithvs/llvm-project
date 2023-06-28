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

// AllOf takes a vector of DynMatchers and returns true if all the DynMatchers
// match the given operation.
struct AllOfMatcher {
  AllOfMatcher(std::vector<matcher::DynMatcher> matchers)
      : matchers(matchers) {}
  bool match(Operation *op) {
    return llvm::all_of(matchers, [&](const matcher::DynMatcher &matcher) {
      return matcher.match(op);
    });
  }
  std::vector<matcher::DynMatcher> matchers;
};

// AnyOf takes a vector of DynMatchers and returns true if any of the
// DynMatchers match the given operation.
struct AnyOfMatcher {
  AnyOfMatcher(std::vector<matcher::DynMatcher> matchers)
      : matchers(matchers) {}
  bool match(Operation *op) {
    return llvm::any_of(matchers, [&](const matcher::DynMatcher &matcher) {
      return matcher.match(op);
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
        return innerMatcher.match(op);
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
struct UsesMatcher {
  UsesMatcher(matcher::DynMatcher innerMatcher, unsigned hops, bool inclusive)
      : innerMatcher(innerMatcher), hops(hops), inclusive(inclusive) {}

  bool recursiveMatch(Operation *op, unsigned tempHops) {
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
  bool match(Operation *op) { return recursiveMatch(op, hops); }
  matcher::DynMatcher innerMatcher;
  unsigned hops;
  bool inclusive;
};

// This matcher checks if an operation is defined by another operation that
// matches the given inner matcher. It allows specifying the number of hops to
// follow in the def-use chain, and whether the chain should be inclusive or
// not.
struct DefinitionsMatcher {
  DefinitionsMatcher(matcher::DynMatcher innerMatcher, unsigned hops,
                     bool inclusive)
      : innerMatcher(innerMatcher), hops(hops), inclusive(inclusive) {}

  bool recursiveMatch(Operation *op, unsigned tempHops) {
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

  bool match(Operation *op) { return recursiveMatch(op, hops); }
  matcher::DynMatcher innerMatcher;
  unsigned hops;
  bool inclusive;
};

} // namespace detail

inline detail::AllOfMatcher allOf(matcher::DynMatcher args...) {
  std::vector<matcher::DynMatcher> matchers({args});
  return detail::AllOfMatcher(matchers);
}

inline detail::AnyOfMatcher anyOf(matcher::DynMatcher args...) {
  std::vector<matcher::DynMatcher> matchers({args});
  return detail::AnyOfMatcher(matchers);
}

inline detail::ArgumentMatcher hasArgument(matcher::DynMatcher innerMatcher,
                                           unsigned argIndex) {
  return detail::ArgumentMatcher(innerMatcher, argIndex);
}

inline detail::UsesMatcher uses(matcher::DynMatcher innerMatcher) {
  return detail::UsesMatcher(innerMatcher, 1, false);
}

inline detail::UsesMatcher getUses(matcher::DynMatcher innerMatcher,
                                   unsigned hops) {
  return detail::UsesMatcher(innerMatcher, hops, false);
}

inline detail::UsesMatcher getAllUses(matcher::DynMatcher innerMatcher,
                                      unsigned hops) {
  return detail::UsesMatcher(innerMatcher, hops, true);
}

inline detail::DefinitionsMatcher definedBy(matcher::DynMatcher innerMatcher) {
  return detail::DefinitionsMatcher(innerMatcher, 1, false);
}

inline detail::DefinitionsMatcher
getDefinitions(matcher::DynMatcher innerMatcher, unsigned hops) {
  return detail::DefinitionsMatcher(innerMatcher, hops, false);
}

inline detail::DefinitionsMatcher
getAllDefinitions(matcher::DynMatcher innerMatcher, unsigned hops) {
  return detail::DefinitionsMatcher(innerMatcher, hops, true);
}

} // namespace extramatcher

} // namespace query

} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERS_EXTRAMATCHERS_H
