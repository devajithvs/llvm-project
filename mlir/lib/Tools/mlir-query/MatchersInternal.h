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
// match(Operation *op)
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

#include "llvm/Support/Debug.h"
using llvm::dbgs;

#define DEBUG_TYPE "mlir-query"
#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
namespace query {
namespace matcher {

// Generic interface for matchers on an MLIR operation.
class MatcherInterface : public llvm::RefCountedBase<MatcherInterface> {
public:
  virtual ~MatcherInterface() = default;

  virtual bool match(Operation *op) = 0;
};

// Matcher wraps a MatcherInterface implementation and provides a match()
// method that redirects calls to the underlying implementation.
class DynMatcher {
public:
  // Takes ownership of the provided implementation pointer.
  DynMatcher(MatcherInterface *Implementation)
      : Implementation(Implementation), ExtractFunction(false) {}

  bool match(Operation *op) const { return Implementation->match(op); }

  DynMatcher *clone() const { return new DynMatcher(*this); }

  void setExtract(bool extractFunction) { ExtractFunction = extractFunction; };
  void setFunctionName(StringRef functionName) { FunctionName = functionName; };

  bool getExtract() const { return ExtractFunction; };
  StringRef getFunctionName() const { return FunctionName; };

private:
  llvm::IntrusiveRefCntPtr<MatcherInterface> Implementation;
  bool ExtractFunction;
  StringRef FunctionName;
};

// SingleMatcher takes a matcher function object and implements
// MatcherInterface.
template <typename MatcherFn>
class SingleMatcher : public MatcherInterface {
public:
  SingleMatcher(MatcherFn &matcherFn) : matcherFn(matcherFn) {}
  bool match(Operation *op) override { return matcherFn.match(op); }

private:
  MatcherFn matcherFn;
};

// TODO: Use a polymorphic matcher instead for this usecase
// VariadicMatcher takes a vector of Matchers and returns true if any Matchers
// match the given operation.
class VariadicMatcher : public MatcherInterface {
public:
  VariadicMatcher(std::vector<DynMatcher> matchers) : matchers(matchers) {}

  bool match(Operation *op) override {
    return llvm::any_of(
        matchers, [&](const DynMatcher &matcher) { return matcher.match(op); });
  }

private:
  std::vector<DynMatcher> matchers;
};

// MatchFinder is used to find all operations that match a given matcher.
class MatchFinder {
public:
  // Returns all operations that match the given matcher.
  std::vector<Operation *> getMatches(Operation *root,
                                      const DynMatcher *matcher) {
    std::vector<Operation *> matches;
    root->walk([&](Operation *subOp) {
      if (matcher->match(subOp))
        matches.push_back(subOp);
    });
    return matches;
  }
};

} // namespace matcher
} // namespace query
} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERS_MATCHERSINTERNAL_H
