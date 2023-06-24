//===--- Query.h - mlir-query ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_QUERY_H
#define MLIR_TOOLS_MLIRQUERY_QUERY_H

#include "Parser.h"
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
  Query(QueryKind Kind) : Kind(Kind) {}
  virtual ~Query();

  // Perform the query on QS and print output to OS.
  // Return false if an error occurs, otherwise return true.
  virtual bool run(llvm::raw_ostream &OS, QuerySession &QS) const = 0;

  llvm::StringRef RemainingContent;
  const QueryKind Kind;
};

typedef llvm::IntrusiveRefCntPtr<Query> QueryRef;

// Any query which resulted in a parse error. The error message is in ErrStr.
struct InvalidQuery : Query {
  InvalidQuery(const llvm::Twine &ErrStr)
      : Query(QK_Invalid), ErrStr(ErrStr.str()) {}
  bool run(llvm::raw_ostream &OS, QuerySession &QS) const override;

  std::string ErrStr;

  static bool classof(const Query *Q) { return Q->Kind == QK_Invalid; }
};

// No-op query (i.e. a blank line).
struct NoOpQuery : Query {
  NoOpQuery() : Query(QK_NoOp) {}
  bool run(llvm::raw_ostream &OS, QuerySession &QS) const override;

  static bool classof(const Query *Q) { return Q->Kind == QK_NoOp; }
};

// Query for "help".
struct HelpQuery : Query {
  HelpQuery() : Query(QK_Help) {}
  bool run(llvm::raw_ostream &OS, QuerySession &QS) const override;

  static bool classof(const Query *Q) { return Q->Kind == QK_Help; }
};

// Query for "match MATCHER".
struct MatchQuery : Query {
  MatchQuery(const matcher::DynMatcher *Matcher)
      : Query(QK_Match), matcher(Matcher) {}
  bool run(llvm::raw_ostream &OS, QuerySession &QS) const override;

  const matcher::DynMatcher *matcher;

  static bool classof(const Query *Q) { return Q->Kind == QK_Match; }
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
  SetQuery(T QuerySession::*Var, T Value)
      : Query(SetQueryKind<T>::value), Var(Var), Value(Value) {}
  bool run(llvm::raw_ostream &OS, QuerySession &QS) const override {
    QS.*Var = Value;
    return true;
  }

  static bool classof(const Query *Q) {
    return Q->Kind == SetQueryKind<T>::value;
  }

  T QuerySession::*Var;
  T Value;
};

} // namespace query
} // namespace mlir

#endif
