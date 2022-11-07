//===- MlirQueryMain.h - MLIR Query main ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the general framework of the MLIR query tool. It
// parses the command line arguments, parses the MLIR file and outputs the query
// results.
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-query/MlirQueryMain.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Reducer/Passes.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/ParseUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Query Parser
//===----------------------------------------------------------------------===//

LogicalResult mlir::mlirQueryMain(int argc, char **argv,
                                  llvm::StringRef toolName) {
  // Override the default '-h' and use the default PrintHelpMessage() which
  // won't print options in categories.
  static llvm::cl::opt<bool> help("h", llvm::cl::desc("Alias for -help"),
                                  llvm::cl::Hidden);

  static llvm::cl::OptionCategory mlirQueryCategory("mlir-query options");

  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::cat(mlirQueryCategory));

  static llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename for the queried result"),
      llvm::cl::init("-"), llvm::cl::cat(mlirQueryCategory));

  llvm::cl::HideUnrelatedOptions(mlirQueryCategory);

  llvm::InitLLVM y(argc, argv);

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR test case query tool.\n");

  if (help) {
    llvm::cl::PrintHelpMessage();
    return success();
  }

  std::string errorMessage;

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output)
    return failure();

  // Operation *op = getOperation();
  // resetIndent();
  // printOperation(op);

  return success();
}
