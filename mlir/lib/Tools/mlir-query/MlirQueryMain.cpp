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

#include "llvm/LineEditor/LineEditor.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/WithColor.h"
#include <string>

using namespace mlir;

// Parse and verify the input MLIR file. Returns null on error.
OwningOpRef<Operation *> loadModule(MLIRContext &context,
                                    StringRef inputFilename,
                                    bool insertImplictModule) {
  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  return parseSourceFileForTool(sourceMgr, &context, insertImplictModule);
}

void runOperation(Operation *op) {
  llvm::outs() << "Visiting op '" << op->getName() << "' with "
                << op->getNumOperands() << " operands:\n";

  // Print information about the producer of each of the operands.
  for (Value operand : op->getOperands()) {
    if (Operation *producer = operand.getDefiningOp()) {
      llvm::outs() << "  - Operand produced by operation '"
                    << producer->getName() << "'\n";
    } else {
      // If there is no defining op, the Value is necessarily a Block
      // argument.
      auto blockArg = operand.cast<BlockArgument>();
      llvm::outs() << "  - Operand produced by Block argument, number "
                    << blockArg.getArgNumber() << "\n";
    }
  }

  // Print information about the user of each of the result.
  llvm::outs() << "Has " << op->getNumResults() << " results:\n";
  for (const auto &indexedResult : llvm::enumerate(op->getResults())) {
    Value result = indexedResult.value();
    llvm::outs() << "  - Result " << indexedResult.index();
    if (result.use_empty()) {
      llvm::outs() << " has no uses\n";
      continue;
    }
    if (result.hasOneUse()) {
      llvm::outs() << " has a single use: ";
    } else {
      llvm::outs() << " has "
                    << std::distance(result.getUses().begin(),
                                    result.getUses().end())
                    << " uses:\n";
    }
    for (Operation *userOp : result.getUsers()) {
      llvm::outs() << "    - " << userOp->getName() << "\n";
    }
  }
}

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

  static llvm::cl::list<std::string> Commands("c", llvm::cl::desc("Specify command to run"),
                                              llvm::cl::value_desc("command"),
                                              llvm::cl::cat(mlirQueryCategory));

  static llvm::cl::list<std::string> CommandFiles("f",
                                                  llvm::cl::desc("Read commands from file"),
                                                  llvm::cl::value_desc("file"),
                                                  llvm::cl::cat(mlirQueryCategory));

  static llvm::cl::opt<std::string> PreloadFile(
      "preload",
      llvm::cl::desc("Preload commands from file and start interactive mode"),
      llvm::cl::value_desc("file"), llvm::cl::cat(mlirQueryCategory));

  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR test case query tool.\n");

  if (!Commands.empty() && !CommandFiles.empty()) {
    llvm::errs() << argv[0] << ": cannot specify both -c and -f\n";
    return failure();
  }

  if ((!Commands.empty() || !CommandFiles.empty()) && !PreloadFile.empty()) {
    llvm::errs() << argv[0]
                 << ": cannot specify both -c or -f with --preload\n";
    return failure();
  }

  // static llvm::cl::opt<std::string> inputFilename(
  //     llvm::cl::Positional, llvm::cl::desc("<input file>"),
  //     llvm::cl::cat(mlirQueryCategory));

  // static llvm::cl::opt<std::string> outputFilename(
  //     "o", llvm::cl::desc("Output filename for the queried result"),
  //     llvm::cl::init("-"), llvm::cl::cat(mlirQueryCategory));
  
  // static llvm::cl::opt<bool> noImplicitModule{
  //     "no-implicit-module",
  //     llvm::cl::desc(
  //         "Disable implicit addition of a top-level module op during parsing"),
  //     llvm::cl::init(false)};
  
  // static llvm::cl::opt<bool> allowUnregisteredDialects(
  //     "allow-unregistered-dialect",
  //     llvm::cl::desc("Allow operation with no registered dialects"),
  //     llvm::cl::init(false));

  // static llvm::cl::opt<bool> splitInputFile(
  //     "split-input-file",
  //     llvm::cl::desc("Split the input file into pieces and "
  //                    "process each chunk independently"),
  //     llvm::cl::init(false));

  // static llvm::cl::opt<bool> verifyDiagnostics(
  //     "verify-diagnostics",
  //     llvm::cl::desc("Check that emitted diagnostics match "
  //                    "expected-* lines on the corresponding line"),
  //     llvm::cl::init(false));

  // llvm::cl::HideUnrelatedOptions(mlirQueryCategory);

  // llvm::InitLLVM y(argc, argv);

  // llvm::cl::ParseCommandLineOptions(argc, argv,
  //                                   "MLIR test case query tool.\n");

  // if (help) {
  //   llvm::cl::PrintHelpMessage();
  //   return success();
  // }

  // std::string errorMessage;

  // auto output = openOutputFile(outputFilename, &errorMessage);
  // if (!output)
  //   return failure();
  
  // MLIRContext context;
  // context.allowUnregisteredDialects(allowUnregisteredDialects);
  // context.printOpOnDiagnostic(!verifyDiagnostics);
  // OwningOpRef<Operation *> opRef =
  //     loadModule(context, inputFilename, !noImplicitModule);
  // if (!opRef)
  //   return failure();

  // // Operation *op = getOperation();
  // // resetIndent();
  // // printOperation(op);

  // PassManager pm(&context, opRef.get()->getName().getStringRef());
  // OwningOpRef<Operation *> op = opRef.get()->clone();

  // if (failed(pm.run(op.get())))
  //   return failure();
  // runOperation(op.get());
  
  // op.get()->print(output->os());
  // output->keep();

  return success();
}
