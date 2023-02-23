//===---- Query.cpp - mlir-query query -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Query.h"
#include "QuerySession.h"
// #include "clang/ASTMatchers/ASTMatchFinder.h"
// #include "clang/Frontend/ASTUnit.h"
// #include "clang/Frontend/TextDiagnostic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionInterfaces.h"

#include "mlir/IR/Matchers.h"

#include "llvm/Support/Debug.h"
using llvm::dbgs;

#define DEBUG_TYPE "mlir-query"

#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")

// using namespace clang::ast_matchers;
// using namespace clang::ast_matchers::dynamic;
using namespace mlir;

// // This could be done better but is not worth the variadic template trouble.
template <typename Matcher>
static unsigned countMatches(FunctionOpInterface f, Matcher &matcher) {
  unsigned count = 0;
  f.walk([&count, &matcher](Operation *op) {
    if (matcher.match(op)){
      llvm::outs() << "matched " << *op << "\n";
      ++count;
    }
  });
  return count;
}

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
        "Match the loaded ASTs against the given matcher.\n"
        "  set bind-root (true|false)    "
        "Set whether to bind the root matcher to \"root\".\n"
        "  set output (diag|print|dump)  "
        "Set whether to print bindings as diagnostics,\n"
        "                                "
        "AST pretty prints or AST dumps.\n\n";
  return true;
}

bool MatchQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  unsigned MatchCount = 0;

  // // std::vector<Operation*> Matches;
  // std::vector<Value> Matches;

  // Operation *rootOp = QS.Op;
  // rootOp->walk([&](mlir::Operation *op) {
  //   for (auto operand: op->getOperands()) {
  //     if (matchPattern(operand, m_Constant())) {
  //       Matches.push_back(operand);
  //       // MatchCount++;
  //     }
  //   }
  // });

  // for (auto op: Matches) {
  //     OS << "\nMatch #" << ++MatchCount << ":\n\n";
  //     op.print(OS);
  //     // op->print(OS);
  //     // OS << op->getName() << ":\n\n";
  // }

  // std::vector<Operation*> Matches;
  Operation *rootOp = QS.Op;
  // rootOp->walk([&](mlir::Operation *op) {
  //     if (matchPattern(op, m_Op<op->getName()>())) {
  //       Matches.push_back(operand);
  //       // MatchCount++;
  //     }
  // });

  // for (auto op: Matches) {
  //     OS << "\nMatch #" << ++MatchCount << ":\n\n";
  //     op->print(OS);
  // }

  // OperationName(StringRef name, MLIRContext *context)
  auto operation_name = OperationName(StringRef("arith.addf"), rootOp->getContext());
  OS << "Operation name " << operation_name.getImpl() << " times\n";

  auto matcher = m_Op<operation_name.getTypeID()>();
  // auto matcher = m_Op<arith::AddFOp>();

  OS << "Pattern add(*) matched " << countMatches(rootOp, matcher) << " times\n";



  LLVM_DEBUG(DBGS() << "Query running" << "\n");
  OS << MatchCount << (MatchCount == 1 ? " match.\n" : " matches.\n");
  return true;
}

const QueryKind SetQueryKind<bool>::value;
const QueryKind SetQueryKind<OutputKind>::value;

} // namespace mlir
} // namespace clang