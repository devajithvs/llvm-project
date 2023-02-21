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

#include "Query.h"
#include "QuerySession.h"
#include "QueryParser.h"

#include "mlir/Tools/mlir-query/MlirQueryMain.h"
#include "mlir/Reducer/Passes.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/InitLLVM.h"

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

  llvm::cl::HideUnrelatedOptions(mlirQueryCategory);

  llvm::InitLLVM y(argc, argv);

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR test case query tool.\n");

  if (help) {
    llvm::cl::PrintHelpMessage();
    return success();
  }

  // QuerySession QS;
  // LineEditor LE("clang-query");
  // LE.setListCompleter([&QS](StringRef Line, size_t Pos) {
  //   return QueryParser::complete(Line, Pos, QS);
  // });
  // while (llvm::Optional<std::string> Line = LE.readLine()) {
  //   QueryRef Q = QueryParser::parse(*Line, QS);
  //   Q->run(llvm::outs(), QS);
  //   llvm::outs().flush();
  //   if (QS.Terminate)
  //     break;
  // }
  


  // Operation *op = getOperation();
  // resetIndent();
  // printOperation(op);

  return success();
}
