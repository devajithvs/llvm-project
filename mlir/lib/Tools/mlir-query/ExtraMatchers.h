//===- Matchers.h - Various common matchers ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides extra matchers that are very useful for mlir-query
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_EXTRAMATCHERS_H
#define MLIR_IR_EXTRAMATCHERS_H

#include "MatchersInternal.h"

namespace mlir {

namespace query {

namespace extramatcher {

namespace detail {
/// VariadicMatcher takes a vector of DynMatchers and returns true if all
/// DynMatchers match the given operation.
struct OperationMatcher {
  OperationMatcher(std::vector<matcher::DynMatcher> matchers)
      : matchers(matchers) {}
  bool matches(Operation *op) {
    matcher::DynTypedNode node = matcher::DynTypedNode::create(op);
    return llvm::all_of(matchers, [&](const matcher::DynMatcher &matcher) {
      return matcher.matches(node);
    });
  }
  std::vector<matcher::DynMatcher> matchers;
};

struct DefinedByMatcher {
  DefinedByMatcher(matcher::DynMatcher innerMatcher)
      : innerMatcher(innerMatcher) {}
  bool match(Operation *op) {
    LLVM_DEBUG(dbgs() << "\nTrying to match\n");
    return llvm::any_of(op->getOperands(), [&](Value operand) {
      if (Operation *operandOp = operand.getDefiningOp()) {
        matcher::DynTypedNode node = matcher::DynTypedNode::create(operandOp);
        return innerMatcher.matches(node);
      }
      return false;
    });
  }
  matcher::DynMatcher innerMatcher;
};

struct UsedByMatcher {
  UsedByMatcher(matcher::DynMatcher innerMatcher)
      : innerMatcher(innerMatcher) {}
  bool match(Operation *op) {
    return llvm::any_of(op->getUsers(), [&](Operation *userOp) {
      matcher::DynTypedNode node = matcher::DynTypedNode::create(userOp);
      return innerMatcher.matches(node);
    });
  }
  matcher::DynMatcher innerMatcher;
};

} // namespace detail

inline detail::OperationMatcher operation(matcher::DynMatcher args...) {
  std::vector<matcher::DynMatcher> matchers({args});
  return detail::OperationMatcher(matchers);
}

inline detail::DefinedByMatcher definedBy(matcher::DynMatcher innerMatcher) {
  return detail::DefinedByMatcher(innerMatcher);
}

inline detail::UsedByMatcher usedBy(matcher::DynMatcher innerMatcher) {
  return detail::UsedByMatcher(innerMatcher);
}

inline detail::UsedByMatcher usedBy(matcher::DynMatcher innerMatcher, StringRef hops) {
  return detail::UsedByMatcher(innerMatcher);
}

} // namespace extramatcher

} // namespace query

} // namespace mlir

#endif // MLIR_IR_EXTRAMATCHERS_H
