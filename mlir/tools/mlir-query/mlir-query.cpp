//===- mlir-query.cpp - MLIR Query Driver -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that queries a file from/to MLIR using one
// of the registered queries.
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-query/MlirQueryMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  return failed(mlirQueryMain(argc, argv, "MLIR Query Testing Tool"));
}
