//===--- QuerySession.h - mlir-query ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_QUERYSESSION_H
#define MLIR_TOOLS_MLIRQUERY_QUERYSESSION_H

#include "llvm/ADT/StringMap.h"

namespace mlir::query {

class RegistryMaps;
// Represents the state for a particular mlir-query session.
class QuerySession {
public:
  QuerySession(Operation *rootOp, llvm::SourceMgr &sourceMgr, unsigned bufferId,
               const matcher::RegistryMaps &registryData)
      : rootOp(rootOp), sourceMgr(sourceMgr), bufferId(bufferId),
        registryData(registryData) {}

  Operation *getRootOp() { return rootOp; }
  llvm::SourceMgr &getSourceManager() const { return sourceMgr; }
  unsigned getBufferId() { return bufferId; }
  const matcher::RegistryMaps &getRegistryData() const { return registryData; }

  llvm::StringMap<matcher::VariantValue> namedValues;
  bool terminate = false;

private:
  Operation *rootOp;
  llvm::SourceMgr &sourceMgr;
  unsigned bufferId;
  const matcher::RegistryMaps &registryData;
};

} // namespace mlir::query

#endif // MLIR_TOOLS_MLIRQUERY_QUERYSESSION_H
