//===--- Query.h - mlir-query ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_QUERY_H
#define MLIR_TOOLS_MLIRQUERY_QUERY_H

#include <string>
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Twine.h"

#include "mlir/IR/Matchers.h"
namespace mlir {
namespace query {

enum OutputKind {
  OK_Diag,
  OK_Print,
  OK_Dump
};

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

  /// Perform the query on \p QS and print output to \p OS.
  ///
  /// \return false if an error occurs, otherwise return true.
  virtual bool run(llvm::raw_ostream &OS, QuerySession &QS) const = 0;

  llvm::StringRef RemainingContent;
  const QueryKind Kind;
};

typedef llvm::IntrusiveRefCntPtr<Query> QueryRef;

/// Any query which resulted in a parse error.  The error message is in ErrStr.
struct InvalidQuery : Query {
  InvalidQuery(const llvm::Twine &ErrStr) : Query(QK_Invalid), ErrStr(ErrStr.str()) {}
  bool run(llvm::raw_ostream &OS, QuerySession &QS) const override;

  std::string ErrStr;

  static bool classof(const Query *Q) { return Q->Kind == QK_Invalid; }
};

/// No-op query (i.e. a blank line).
struct NoOpQuery : Query {
  NoOpQuery() : Query(QK_NoOp) {}
  bool run(llvm::raw_ostream &OS, QuerySession &QS) const override;

  static bool classof(const Query *Q) { return Q->Kind == QK_NoOp; }
};

/// Query for "help".
struct HelpQuery : Query {
  HelpQuery() : Query(QK_Help) {}
  bool run(llvm::raw_ostream &OS, QuerySession &QS) const override;

  static bool classof(const Query *Q) { return Q->Kind == QK_Help; }
};

/*
struct name_op_matcher  {
  StringRef opName;
  name_op_matcher(StringRef opN) : opName(opN) {}

  bool match(Operation *op) { return op->getName().getStringRef() == opName; }
};

struct attr_op_matcher  {
  StringRef opAttr;
  attr_op_matcher(StringRef opN) : opAttr(opN) {}

  bool match(Operation *op) { return op->hasAttr(opAttr); }
};
*/


/// Query for "match MATCHER".
template <typename T>
struct MatchQuery : Query {
  MatchQuery(StringRef Source,
             const T &Matcher)
      : Query(QK_Match), Matcher(Matcher), Source(Source) {}
  bool run(llvm::raw_ostream &OS, QuerySession &QS) const override;

  T Matcher;

  StringRef Source;

  static bool classof(const Query *Q) { return Q->Kind == QK_Match; }
};

template <typename T> struct SetQueryKind {};

template <> struct SetQueryKind<bool> {
  static const QueryKind value = QK_SetBool;
};

template <> struct SetQueryKind<OutputKind> {
  static const QueryKind value = QK_SetOutputKind;
};

/// Query for "set VAR VALUE".
template <typename T> struct SetQuery : Query {
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
