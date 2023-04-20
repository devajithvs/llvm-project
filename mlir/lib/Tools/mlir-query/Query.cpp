//===---- Query.cpp - mlir-query query -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Query.h"
#include "QuerySession.h"
#include "llvm/Support/raw_ostream.h"
#include <bits/stdc++.h>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/Support/Debug.h"
using llvm::dbgs;

#define DEBUG_TYPE "mlir-query"
#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;

namespace mlir {
namespace query {

Query::~Query() {}

bool InvalidQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  OS << ErrStr << "\n";
  return false;
}

bool NoOpQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  return true;
}

bool HelpQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  OS << "Available commands:\n\n"
        "  match MATCHER, m MATCHER      "
        "Match the mlir against the given matcher.\n"
        "  set bind-root (true|false)    "
        "Set whether to bind the root matcher to \"root\".\n\n";
  return true;
}

// This could be done better but is not worth the variadic template trouble.
std::vector<matcher::DynTypedNode>
getMatches(Operation *rootOp, const matcher::DynMatcher *matcher) {
  auto matchFinder = matcher::MatchFinder();
  return matchFinder.getMatches(rootOp, matcher);
}

// TODO: Only supports operation node type.
Operation *extractFunction(std::vector<matcher::DynTypedNode> &nodes,
                           OpBuilder builder) {
  std::vector<Operation *> slice;
  std::vector<Value> values;

  bool hasReturn = false;
  TypeRange resultType = std::nullopt;

  for (matcher::DynTypedNode node : nodes) {
    if (Operation *op = *node.get<Operation *>()) {
      slice.push_back(op);
      if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
        resultType = returnOp.getOperands().getTypes();
        hasReturn = true;
      } else {
        // Extract all values that might potentially be needed as func
        // arguments.
        for (Value value : op->getOperands()) {
          values.push_back(value);
        }
      }
    }
  }

  auto loc = builder.getUnknownLoc();
  func::FuncOp funcOp = func::FuncOp::create(
      loc, "extracted",
      builder.getFunctionType(ValueRange(values), resultType));

  loc = funcOp.getLoc();
  builder.setInsertionPointToEnd(funcOp.addEntryBlock());
  builder.setInsertionPointToEnd(&funcOp.getBody().front());

  IRMapping mapper;
  for (const auto &arg : llvm::enumerate(values))
    mapper.map(arg.value(), funcOp.getArgument(arg.index()));
  for (Operation *slicedOp : slice)
    builder.clone(*slicedOp, mapper);

  // Remove func arguments that are not used.
  int currentIndex = 0;
  for (auto value : funcOp.getArguments()) {
    if (value.getUses().empty()) {
      funcOp.eraseArgument(currentIndex);
    } else {
      currentIndex++;
    }
  }
  if (!hasReturn)
    builder.create<func::ReturnOp>(loc);

  return funcOp;
}

bool MatchQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {

  Operation *rootOp = QS.Op;

  if (!matcher) {
    return false;
  }
  auto matches = getMatches(rootOp, matcher);

  if (matcher->getExtract()) {
    MLIRContext context;
    context.loadDialect<func::FuncDialect>();
    OpBuilder builder(&context);
    Operation *function = extractFunction(matches, builder);
    OS << "\n\n" << *function << "\n\n\n";
  } else {
    unsigned MatchCount = 0;
    for (auto node : matches) {
      if (Operation *op = *node.get<Operation *>()) {
        auto opLoc = op->getLoc().cast<FileLineColLoc>();
        OS << "\nMatch #" << ++MatchCount << ":\n\n";
        OS << opLoc.getFilename().getValue() << ":" << opLoc.getLine() << ":"
           << opLoc.getColumn() << ": note: \"root\" binds here\n"
           << *op << "\n";
      }
    }
    OS << "\n"
       << MatchCount << (MatchCount == 1 ? " match.\n\n" : " matches.\n\n");
  }

  return true;
}

const QueryKind SetQueryKind<bool>::value;
const QueryKind SetQueryKind<OutputKind>::value;

} // namespace query
} // namespace mlir