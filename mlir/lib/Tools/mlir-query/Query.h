//===--- Query.h - mlir-query ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_QUERY_H
#define MLIR_TOOLS_MLIRQUERY_QUERY_H

#include "Matcher/VariantValue.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/Twine.h"
#include <string>

namespace mlir {
namespace query {

enum OutputKind { OK_Diag, OK_Print, OK_Dump };

enum QueryKind {
  QK_Invalid,
  QK_NoOp,
  QK_Help,
  QK_Match,
  QK_SetBool,
  QK_SetOutputKind
};

class QuerySession;

struct Query : llvm::RefCountedBase<Query> {
  Query(QueryKind kind) : kind(kind) {}
  virtual ~Query();

  // Perform the query on QS and print output to OS.
  // Return false if an error occurs, otherwise return true.
  virtual bool run(llvm::raw_ostream &OS, QuerySession &QS) const = 0;

  llvm::StringRef remainingContent;
  const QueryKind kind;
};

typedef llvm::IntrusiveRefCntPtr<Query> QueryRef;

// Any query which resulted in a parse error. The error message is in ErrStr.
struct InvalidQuery : Query {
  InvalidQuery(const llvm::Twine &errStr)
      : Query(QK_Invalid), errStr(errStr.str()) {}
  bool run(llvm::raw_ostream &OS, QuerySession &QS) const override;

  std::string errStr;

  static bool classof(const Query *Q) { return Q->kind == QK_Invalid; }
};

// No-op query (i.e. a blank line).
struct NoOpQuery : Query {
  NoOpQuery() : Query(QK_NoOp) {}
  bool run(llvm::raw_ostream &OS, QuerySession &QS) const override;

  static bool classof(const Query *Q) { return Q->kind == QK_NoOp; }
};

// Query for "help".
struct HelpQuery : Query {
  HelpQuery() : Query(QK_Help) {}
  bool run(llvm::raw_ostream &OS, QuerySession &QS) const override;

  static bool classof(const Query *Q) { return Q->kind == QK_Help; }
};

// Query for "match MATCHER".
struct MatchQuery : Query {
  MatchQuery(StringRef source, const matcher::DynMatcher &matcher)
      : Query(QK_Match), matcher(matcher), source(source) {}
  bool run(llvm::raw_ostream &OS, QuerySession &QS) const override;

  const matcher::DynMatcher matcher;

  StringRef source;

  static bool classof(const Query *Q) { return Q->kind == QK_Match; }
};

template <typename T>
struct SetQueryKind {};

template <>
struct SetQueryKind<bool> {
  static const QueryKind value = QK_SetBool;
};

template <>
struct SetQueryKind<OutputKind> {
  static const QueryKind value = QK_SetOutputKind;
};

// Query for "set VAR VALUE".
template <typename T>
struct SetQuery : Query {
  SetQuery(T QuerySession::*var, T value)
      : Query(SetQueryKind<T>::value), var(var), value(value) {}
  bool run(llvm::raw_ostream &OS, QuerySession &QS) const override {
    QS.*var = value;
    return true;
  }

  static bool classof(const Query *Q) {
    return Q->kind == SetQueryKind<T>::value;
  }

  T QuerySession::*var;
  T value;
};

} // namespace query
} // namespace mlir

#endif
