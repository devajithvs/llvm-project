//===- MlirQueryMain.h - MLIR Query main ------------------------*- C++ -*-===//
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
#include "Query.h"
#include "QueryParser.h"
#include "QuerySession.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/ParseUtilities.h"
#include "llvm/LineEditor/LineEditor.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

#include "llvm/Support/Debug.h"
using llvm::dbgs;

#define DEBUG_TYPE "mlir-query"
#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;
using namespace mlir::query;
using namespace llvm;

//===----------------------------------------------------------------------===//
// Query Parser
//===----------------------------------------------------------------------===//

LogicalResult mlir::mlirQueryMain(int argc, char **argv, MLIRContext &context) {
  // Override the default '-h' and use the default PrintHelpMessage() which
  // won't print options in categories.
  static llvm::cl::opt<bool> help("h", llvm::cl::desc("Alias for -help"),
                                  llvm::cl::Hidden);

  static llvm::cl::OptionCategory mlirQueryCategory("mlir-query options");

  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::cat(mlirQueryCategory));

  static llvm::cl::opt<bool> noImplicitModule{
      "no-implicit-module",
      llvm::cl::desc(
          "Disable implicit addition of a top-level module op during parsing"),
      llvm::cl::init(false)};

  llvm::cl::HideUnrelatedOptions(mlirQueryCategory);

  llvm::InitLLVM y(argc, argv);

  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR test case query tool.\n");

  if (help) {
    llvm::cl::PrintHelpMessage();
    return success();
  }

  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(file), SMLoc());

  // Parse the input MLIR file.
  OwningOpRef<Operation *> opRef =
      parseSourceFileForTool(sourceMgr, &context, !noImplicitModule);
  if (!opRef)
    return failure();

  QuerySession QS(opRef.get(), sourceMgr);
  LineEditor LE("mlir-query");
  LE.setListCompleter([&QS](StringRef Line, size_t Pos) {
    return QueryParser::complete(Line, Pos, QS);
  });
  while (llvm::Optional<std::string> Line = LE.readLine()) {
    QueryRef Q = QueryParser::parse(*Line, QS);
    Q->run(llvm::outs(), QS);
LLVM_DEBUG(DBGS() << "unningtoken" <<  "\n");
    llvm::outs().flush();
LLVM_DEBUG(DBGS() << "Flush it and terminate?" <<  "\n");
    if (QS.Terminate)
      break;
  }

  return success();
}
