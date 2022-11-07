//===- MlirQueryMain.h - MLIR Query main -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-query for when built as standalone
// binary.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_MLIRQUERYMAIN_H
#define MLIR_TOOLS_MLIRQUERY_MLIRQUERYMAIN_H

#include "Query.h"
#include "QueryParser.h"
#include "QuerySession.h"
#include "mlir/Support/LogicalResult.h"
namespace mlir {

class MLIRContext;
/// This is the entry point for the implementation
/// of tools like `mlir-query`. The query to perform is parsed from
/// the command line. The `toolName` argument is used for the header displayed
/// by `--help`.
LogicalResult mlirQueryMain(int argc, char **argv, MLIRContext &context);
} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MLIRQUERYMAIN_H
