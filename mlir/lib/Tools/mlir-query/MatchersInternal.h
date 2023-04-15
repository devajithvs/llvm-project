//===- MatchersInternal.h - Structural query framework ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the base layer of the matcher framework.
//
// Matchers are methods that return a Matcher which provides a method
// matches(Operation *op)
//
// In general, matchers have two parts:
// 1. A function Matcher MatcherName(<arguments>) which returns a Matcher
// based on the arguments.
// 2. An implementation of a class derived from MatcherInterface.
//
// The matcher functions are defined in include/mlir/IR/Matchers.h.
// This file contains the wrapper classes needed to construct matchers for
// mlir-query.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_MATCHERS_MATCHERSINTERNAL_H
#define MLIR_TOOLS_MLIRQUERY_MATCHERS_MATCHERSINTERNAL_H

#include "mlir/IR/Matchers.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"

namespace mlir {
namespace query {
namespace matcher {

// Kind identifier.
///
//// It can be constructed from any node kind and allows for runtime type
/// hierarchy checks.
/// Use getFromNodeKind<T>() to construct them.
class MLIRNodeKind {
public:
  /// Empty identifier. It matches nothing.
  constexpr MLIRNodeKind() : KindId(NKI_None) {}

  /// Construct an identifier for T.
  template <class T> static constexpr MLIRNodeKind getFromNodeKind() {
    return MLIRNodeKind(KindToKindId<T>::Id);
  }

  /// \{
  /// Construct an identifier for the dynamic type of the node
  static MLIRNodeKind getFromNode(const Operation &O);
  static MLIRNodeKind getFromNode(const Value &V);
  /// \}

  /// Returns \c true if \c this and \c Other represent the same kind.
  constexpr bool isSame(MLIRNodeKind Other) const {
    return KindId != NKI_None && KindId == Other.KindId;
  }

  /// Returns \c true only for the default \c MLIRNodeKind()
  constexpr bool isNone() const { return KindId == NKI_None; }

  /// String representation of the kind.
  StringRef asStringRef() const;

private:
  /// Kind ids.
  ///
  /// Includes all possible base and derived kinds.
  enum NodeKindId {
    NKI_None,
    NKI_Attr,
    NKI_Value,
    NKI_Operation
  };

  /// Use getFromNodeKind<T>() to construct the kind.
  constexpr MLIRNodeKind(NodeKindId KindId) : KindId(KindId) {}

  /// Helper meta-function to convert a kind T to its enum value.
  ///
  /// This struct is specialized below for all known kinds.
  template <class T> struct KindToKindId {
    static const NodeKindId Id = NKI_None;
  };
  template <class T>
  struct KindToKindId<const T> : KindToKindId<T> {};

  NodeKindId KindId;
};

template <> struct MLIRNodeKind::KindToKindId<Operation> {
  static const NodeKindId Id = NKI_Operation;
};

/// A dynamically typed MLIR node container.
///
/// Stores an MLIR node in a type safe way. This allows writing code that
/// works with different kinds of MLIR nodes, despite the fact that they don't
/// have a common base class.
///
/// Use \c create(Node) to create a \c DynTypedNode from an MLIR node,
/// and \c get<T>() to retrieve the node as type T if the types match.
///
/// See \c MLIRNodeKind for which node base types are currently supported;
/// You can create DynTypedNodes for all nodes in the inheritance hierarchy of
/// the supported base types.
class DynTypedNode {
public:
  /// Creates a \c DynTypedNode from \c Node.
  template <typename T>
  static DynTypedNode create(T &Node) {
    return BaseConverter<T>::create(Node);
  }

  /// Retrieve the stored node as type \c T.
  ///
  /// Returns NULL if the stored node does not have a type that is
  /// convertible to \c T.
  ///
  /// For types that have identity via their pointer in the AST
  /// (like \c Stmt, \c Decl, \c Type and \c NestedNameSpecifier) the returned
  /// pointer points to the referenced AST node.
  /// For other types (like \c QualType) the value is stored directly
  /// in the \c DynTypedNode, and the returned pointer points at
  /// the storage inside DynTypedNode. For those nodes, do not
  /// use the pointer outside the scope of the DynTypedNode.
  template <typename T> T *get() {
    return BaseConverter<T>::get(NodeKind, &Storage);
  }

  /// Retrieve the stored node as type \c T.
  ///
  /// Similar to \c get(), but asserts that the type is what we are expecting.
  template <typename T>
  T *getUnchecked() const {
    return BaseConverter<T>::getUnchecked(NodeKind, &Storage);
  }

private:
  /// Takes care of converting from and to \c T.
  template <typename T, typename EnablerT = void> struct BaseConverter;

  /// Converter that stores T* (by pointer).
  template <typename T> struct PtrConverter {
    static T *get(MLIRNodeKind NodeKind, const void *Storage) {
      if (MLIRNodeKind::getFromNodeKind<T>().isSame(NodeKind))
        return getUnchecked(NodeKind, Storage);
      return nullptr;
    }
    static T *getUnchecked(MLIRNodeKind NodeKind,  const void *Storage) {
      assert(MLIRNodeKind::getFromNodeKind<T>().isSame(NodeKind));
      return static_cast<T *>( *reinterpret_cast<void *const *>(Storage));
    }
    static DynTypedNode create(T &Node) {
      DynTypedNode Result;
      Result.NodeKind = MLIRNodeKind::getFromNodeKind<T>();
      new (&Result.Storage) void *(&Node);
      return Result;
    }
  };

  /// Converter that stores T (by value).
  template <typename T> struct ValueConverter {
    static const T *get(MLIRNodeKind NodeKind, const void *Storage) {
      if (MLIRNodeKind::getFromNodeKind<T>().isSame(NodeKind))
        return reinterpret_cast<const T *>(Storage);
      return nullptr;
    }
    static const T &getUnchecked(MLIRNodeKind NodeKind, const void *Storage) {
      assert(MLIRNodeKind::getFromNodeKind<T>().isSame(NodeKind));
      return *reinterpret_cast<const T *>(Storage);
    }
    static DynTypedNode create(const T &Node) {
      DynTypedNode Result;
      Result.NodeKind = MLIRNodeKind::getFromNodeKind<T>();
      new (&Result.Storage) T(Node);
      return Result;
    }
  };

  /// Converter that stores nodes by value. It must be possible to dynamically
  /// cast the stored node within a type hierarchy without breaking (especially
  /// through slicing).
  template <typename T, typename BaseT,
            typename = std::enable_if_t<(sizeof(T) == sizeof(BaseT))>>
  struct DynCastValueConverter {
    static const T *get(MLIRNodeKind NodeKind, const void *Storage) {
      if (MLIRNodeKind::getFromNodeKind<T>().isBaseOf(NodeKind))
        return &getUnchecked(NodeKind, Storage);
      return nullptr;
    }
    static const T &getUnchecked(MLIRNodeKind NodeKind, const void *Storage) {
      assert(MLIRNodeKind::getFromNodeKind<T>().isBaseOf(NodeKind));
      return *static_cast<const T *>(reinterpret_cast<const BaseT *>(Storage));
    }
    static DynTypedNode create(const T &Node) {
      DynTypedNode Result;
      Result.NodeKind = MLIRNodeKind::getFromNode(Node);
      new (&Result.Storage) T(Node);
      return Result;
    }
  };
  MLIRNodeKind NodeKind;
  /// Stores the data of the node.
  /// Note that we can store Operation and Value by pointer as they are
  /// guaranteed to be unique pointers pointing to dedicated storage in the MLIR.
  llvm::AlignedCharArrayUnion<const void *, Operation, Value> Storage;
};

template <>
struct DynTypedNode::BaseConverter<
    Operation, void> : public PtrConverter<Operation> {};

class DynMatcherInterface : public llvm::RefCountedBase<DynMatcherInterface> {
public:
  virtual ~DynMatcherInterface() = default;

  /// Returns true if 'op' can be matched.
  virtual bool matches(DynTypedNode &DynNode) = 0;
};

/// Matcher wraps a MatcherInterface implementation and provides a matches()
/// method that redirects calls to the underlying implementation.
class DynMatcher {
public:
  DynMatcher(DynMatcherInterface *Implementation) : Implementation(Implementation) {}

  /// Returns true if the matcher matches the given op.
  bool matches(DynTypedNode &DynNode) const { return Implementation->matches(DynNode); }

  DynMatcher *clone() const { return new DynMatcher(*this); }

private:
  llvm::IntrusiveRefCntPtr<DynMatcherInterface> Implementation;
};

/// SingleMatcher takes a matcher function object and implements
/// MatcherInterface.
template <typename T>
class SingleMatcher : public DynMatcherInterface {
public:
  SingleMatcher(T &matcherFn) : matcherFn(matcherFn) {}
  bool matches(DynTypedNode &DynNode) override {
    Operation* op = DynNode.getUnchecked<Operation>();
    return matcherFn.match(op);
  }

private:
  T matcherFn;
};

/// VariadicMatcher takes a vector of Matchers and returns true if all Matchers
/// match the given operation.
class VariadicMatcher : public DynMatcherInterface {
public:
  VariadicMatcher(std::vector<DynMatcher> matchers) : matchers(matchers) {}

  bool matches(DynTypedNode &DynNode) override {
    return llvm::all_of(
        matchers, [&](const DynMatcher &matcher) { return matcher.matches(DynNode); });
  }

private:
  std::vector<DynMatcher> matchers;
};

/// MatchFinder is used to find all operations that match a given matcher.
class MatchFinder {
public:
  /// Returns all operations that match the given matcher.
  std::vector<DynTypedNode> getMatches(Operation *op, const DynMatcher *matcher) {
    std::vector<DynTypedNode> matches;
    op->walk([&](Operation *subOp) {

      //const MLIRNodeKind node = MLIRNodeKind::getFromNode(*subOp);
      DynTypedNode node = DynTypedNode::create(*subOp);
      if (matcher->matches(node))
        matches.push_back(node);
    });
    return matches;
  }
};

} // namespace matcher
} // namespace query
} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERS_MATCHERSINTERNAL_H
