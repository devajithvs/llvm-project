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
/// VariadicMatcher takes a vector of Matchers and returns true if all Matchers
/// match the given operation.
struct OperationMatcher {
  OperationMatcher(std::vector<matcher::DynTypedMatcher> matchers)
      : matchers(matchers) {}
  bool matches(Operation *op) {
    return llvm::all_of(matchers, [&](const matcher::DynTypedMatcher &matcher) {
      return matcher.matches(op);
    });
  }
  std::vector<matcher::DynTypedMatcher> matchers;
};
} // namespace detail

inline detail::OperationMatcher operation(matcher::DynTypedMatcher args...) {
  std::vector<matcher::DynTypedMatcher> matchers({args});
  return detail::OperationMatcher(matchers);
}

} // namespace extramatcher

} // namespace query

} // namespace mlir

#endif // MLIR_IR_EXTRAMATCHERS_H
