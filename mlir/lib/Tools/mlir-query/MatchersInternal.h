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

  /// Returns \c true if \c this is a base kind of (or same as) \c Other.
  /// \param Distance If non-null, used to return the distance between \c this
  /// and \c Other in the class hierarchy.
  bool isBaseOf(MLIRNodeKind Other, unsigned *Distance = nullptr) const;

  /// String representation of the kind.
  StringRef asStringRef() const;

  /// Strict weak ordering for MLIRNodeKind.
  constexpr bool operator<(const MLIRNodeKind &Other) const {
    return KindId < Other.KindId;
  }

  /// Return the most derived type between \p Kind1 and \p Kind2.
  ///
  /// Return MLIRNodeKind() if they are not related.
  static MLIRNodeKind getMostDerivedType(MLIRNodeKind Kind1, MLIRNodeKind Kind2);

  /// Return the most derived common ancestor between Kind1 and Kind2.
  ///
  /// Return MLIRNodeKind() if they are not related.
  static MLIRNodeKind getMostDerivedCommonAncestor(MLIRNodeKind Kind1,
                                                  MLIRNodeKind Kind2);

  MLIRNodeKind getCladeKind() const;

  /// Hooks for using MLIRNodeKind as a key in a DenseMap.
  struct DenseMapInfo {
    // MLIRNodeKind() is a good empty key because it is represented as a 0.
    static inline MLIRNodeKind getEmptyKey() { return MLIRNodeKind(); }

    static unsigned getHashValue(const MLIRNodeKind &Val) { return Val.KindId; }
    static bool isEqual(const MLIRNodeKind &LHS, const MLIRNodeKind &RHS) {
      return LHS.KindId == RHS.KindId;
    }
  };

private:
  /// Kind ids.
  ///
  /// Includes all possible base and derived kinds.
  enum NodeKindId {
    NKI_None,
    NKI_Operation,
    NKI_Value,
  };

  /// Use getFromNodeKind<T>() to construct the kind.
  constexpr MLIRNodeKind(NodeKindId KindId) : KindId(KindId) {}

  /// Returns \c true if \c Base is a base kind of (or same as) \c
  ///   Derived.
  /// \param Distance If non-null, used to return the distance between \c Base
  /// and \c Derived in the class hierarchy.
  static bool isBaseOf(NodeKindId Base, NodeKindId Derived, unsigned *Distance);

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
  static DynTypedNode create(const T &Node) {
    return BaseConverter<T>::create(Node);
  }

  /// Retrieve the stored node as type \c T.
  ///
  /// Returns NULL if the stored node does not have a type that is
  /// convertible to \c T.
  ///
  /// For types that have identity via their pointer in the MLIR
  /// (like \c Stmt, \c Decl, \c Type and \c NestedNameSpecifier) the returned
  /// pointer points to the referenced MLIR node.
  /// For other types (like \c QualType) the value is stored directly
  /// in the \c DynTypedNode, and the returned pointer points at
  /// the storage inside DynTypedNode. For those nodes, do not
  /// use the pointer outside the scope of the DynTypedNode.
  template <typename T> const T *get() const {
    return BaseConverter<T>::get(NodeKind, &Storage);
  }

  /// Retrieve the stored node as type \c T.
  ///
  /// Similar to \c get(), but asserts that the type is what we are expecting.
  template <typename T>
  const T &getUnchecked() const {
    return BaseConverter<T>::getUnchecked(NodeKind, &Storage);
  }

  MLIRNodeKind getNodeKind() const { return NodeKind; }

private:
  /// Takes care of converting from and to \c T.
  template <typename T, typename EnablerT = void> struct BaseConverter;

  /// Converter that uses dyn_cast<T> from a stored BaseT*.
  template <typename T, typename BaseT> struct DynCastPtrConverter {
    static const T *get(MLIRNodeKind NodeKind, const void *Storage) {
      if (MLIRNodeKind::getFromNodeKind<T>().isBaseOf(NodeKind))
        return &getUnchecked(NodeKind, Storage);
      return nullptr;
    }
    static const T &getUnchecked(MLIRNodeKind NodeKind, const void *Storage) {
      assert(MLIRNodeKind::getFromNodeKind<T>().isBaseOf(NodeKind));
      return *cast<T>(static_cast<const BaseT *>(
          *reinterpret_cast<const void *const *>(Storage)));
    }
    static DynTypedNode create(const BaseT &Node) {
      DynTypedNode Result;
      Result.NodeKind = MLIRNodeKind::getFromNode(Node);
      new (&Result.Storage) const void *(&Node);
      return Result;
    }
  };

  /// Converter that stores T* (by pointer).
  template <typename T> struct PtrConverter {
    static const T *get(MLIRNodeKind NodeKind, const void *Storage) {
      if (MLIRNodeKind::getFromNodeKind<T>().isSame(NodeKind))
        return &getUnchecked(NodeKind, Storage);
      return nullptr;
    }
    static const T &getUnchecked(MLIRNodeKind NodeKind, const void *Storage) {
      assert(MLIRNodeKind::getFromNodeKind<T>().isSame(NodeKind));
      return *static_cast<const T *>(
          *reinterpret_cast<const void *const *>(Storage));
    }
    static DynTypedNode create(const T &Node) {
      DynTypedNode Result;
      Result.NodeKind = MLIRNodeKind::getFromNodeKind<T>();
      new (&Result.Storage) const void *(&Node);
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
  ///
  /// Note that we can store \c Decls, \c Stmts, \c Types,
  /// \c NestedNameSpecifiers and \c CXXCtorInitializer by pointer as they are
  /// guaranteed to be unique pointers pointing to dedicated storage in the MLIR.
  /// \c QualTypes, \c NestedNameSpecifierLocs, \c TypeLocs,
  /// \c TemplateArguments and \c TemplateArgumentLocs on the other hand do not
  /// have storage or unique pointers and thus need to be stored by value.
  llvm::AlignedCharArrayUnion<const void *>
      Storage;
};

template <typename T>
struct DynTypedNode::BaseConverter<
    T, std::enable_if_t<std::is_base_of<Decl, T>::value>>
    : public DynCastPtrConverter<T, Decl> {};

template <typename T>
struct DynTypedNode::BaseConverter<
    T, std::enable_if_t<std::is_base_of<Stmt, T>::value>>
    : public DynCastPtrConverter<T, Stmt> {};

template <typename T>
struct DynTypedNode::BaseConverter<
    T, std::enable_if_t<std::is_base_of<Type, T>::value>>
    : public DynCastPtrConverter<T, Type> {};

template <typename T>
struct DynTypedNode::BaseConverter<
    T, std::enable_if_t<std::is_base_of<OMPClause, T>::value>>
    : public DynCastPtrConverter<T, OMPClause> {};

template <typename T>
struct DynTypedNode::BaseConverter<
    T, std::enable_if_t<std::is_base_of<Attr, T>::value>>
    : public DynCastPtrConverter<T, Attr> {};

template <>
struct DynTypedNode::BaseConverter<
    NestedNameSpecifier, void> : public PtrConverter<NestedNameSpecifier> {};

template <>
struct DynTypedNode::BaseConverter<
    CXXCtorInitializer, void> : public PtrConverter<CXXCtorInitializer> {};

template <>
struct DynTypedNode::BaseConverter<
    TemplateArgument, void> : public ValueConverter<TemplateArgument> {};

template <>
struct DynTypedNode::BaseConverter<TemplateArgumentLoc, void>
    : public ValueConverter<TemplateArgumentLoc> {};

template <>
struct DynTypedNode::BaseConverter<LambdaCapture, void>
    : public ValueConverter<LambdaCapture> {};

template <>
struct DynTypedNode::BaseConverter<
    TemplateName, void> : public ValueConverter<TemplateName> {};

template <>
struct DynTypedNode::BaseConverter<
    NestedNameSpecifierLoc,
    void> : public ValueConverter<NestedNameSpecifierLoc> {};

template <>
struct DynTypedNode::BaseConverter<QualType,
                                   void> : public ValueConverter<QualType> {};

template <typename T>
struct DynTypedNode::BaseConverter<
    T, std::enable_if_t<std::is_base_of<TypeLoc, T>::value>>
    : public DynCastValueConverter<T, TypeLoc> {};

template <>
struct DynTypedNode::BaseConverter<CXXBaseSpecifier, void>
    : public PtrConverter<CXXBaseSpecifier> {};

template <>
struct DynTypedNode::BaseConverter<ObjCProtocolLoc, void>
    : public ValueConverter<ObjCProtocolLoc> {};

/// Generic interface for all matchers.
///
/// Used by the implementation of Matcher<T> and DynTypedMatcher.
/// In general, implement MatcherInterface<T> or SingleNodeMatcherInterface<T>
/// instead.
class DynMatcherInterface
    : public llvm::ThreadSafeRefCountedBase<DynMatcherInterface> {
public:
  virtual ~DynMatcherInterface() = default;

  /// Returns true if \p DynNode can be matched.
  virtual bool dynMatches(const DynTypedNode &DynNode) const = 0;

};

/// Generic interface for matchers on an MLIR node of type T.
///
/// Implement this if your matcher may need to inspect the children or
/// descendants of the node or bind matched nodes to names. If you are
/// writing a simple matcher that only inspects properties of the
/// current node and doesn't care about its children or descendants,
/// implement SingleNodeMatcherInterface instead.
template <typename T>
class MatcherInterface : public DynMatcherInterface {
public:
  /// Returns true if 'Node' can be matched.
  ///
  /// May bind 'Node' to an ID via 'Builder', or recurse into
  /// the MLIR via 'Finder'.
  virtual bool matches(const T &Node) = 0;

  bool dynMatches(const DynTypedNode &DynNode) const override {
    return matches(DynNode.getUnchecked<T>());
  }
};

/// Matcher wraps a MatcherInterface implementation and provides a matches()
/// method that redirects calls to the underlying implementation.
template <typename T>
class Matcher {
public:
  Matcher(MatcherInterface<T> *Implementation) : Implementation(Implementation) {}

  /// Returns true if the matcher matches the given op.
  bool matches(const T &Node) const { return Implementation->matches(Node); }

  Matcher *clone() const { return new Matcher<T>(*this); }

private:
  llvm::IntrusiveRefCntPtr<MatcherInterface<T>> Implementation;
};


/// Matcher that works on a \c DynTypedNode.
///
/// It is constructed from a \c MatcherTemplate<T> object and redirects most calls to
/// underlying matcher.
/// It checks whether the \c DynTypedNode is convertible into the type of the
/// underlying matcher and then do the actual match on the actual node, or
/// return false if it is not convertible.
class DynTypedMatcher {
public:
  template <typename T>
  DynTypedMatcher(MatcherInterface<T> *Implementation) : SupportedKind(MLIRNodeKind::getFromNodeKind<T>()), Implementation(Implementation) {}

  /// Returns true if the matcher matches the given op.
  bool matches(const DynTypedNode &DynNode) const { return Implementation->matches(DynNode); }

  DynTypedMatcher *clone() const { return new DynTypedMatcher(*this); }

  /// Check whether this matcher could ever match a node of kind \p Kind.
  /// \return \c false if this matcher will never match such a node. Otherwise,
  /// return \c true.
  bool canMatchNodesOfKind(MLIRNodeKind Kind) const;

  /// Construct a \c MatcherTemplate<T> interface around the dynamic matcher.
  ///
  /// This method asserts that \c canConvertTo() is \c true. Callers
  /// should call \c canConvertTo() first to make sure that \c this is
  /// compatible with T.
  template <typename T>
  Matcher<T> convertTo() const {
    //assert(canConvertTo<T>());
    return unconditionalConvertTo<T>();
  }

private:
  MLIRNodeKind SupportedKind;
  llvm::IntrusiveRefCntPtr<DynMatcherInterface> Implementation;
};

template <typename T>
inline MatcherTemplate<T> DynTypedMatcher::unconditionalConvertTo() const {
  return MatcherTemplate<T>(*this);
}

/// SingleMatcher takes a matcher function object and implements
/// MatcherInterface.
template <typename MatcherFn, typename T>
class SingleMatcher : public MatcherInterface<T> {
public:
  SingleMatcher(MatcherFn &matcherFn) : matcherFn(matcherFn) {}
  bool matches(const T &Node) override { return matcherFn.match(Node); }

private:
  MatcherFn matcherFn;
};

// static bool allofvariadicoperator(Operation *op,
// std::vector<Matcher> InnerMatchers) {
// return llvm::all_of(InnerMatchers, [&](const Matcher &InnerMatcher) {
// return InnerMatcher.matches(op);
// });
// }

/// VariadicMatcher takes a vector of Matchers and returns true if all Matchers
/// match the given operation.
template <typename T>
class VariadicMatcher : public MatcherInterface<T> {
public:
  VariadicMatcher(std::vector<Matcher<T>> matchers) : matchers(matchers) {}

  bool matches(const T &Node) override {
    return llvm::all_of(
        matchers, [&](const Matcher<T> &matcher) { return matcher.matches(Node); });
  }

private:
  std::vector<Matcher<T>> matchers;
};

/// MatchFinder is used to find all operations that match a given matcher.
class MatchFinder {
public:
  /// Returns all operations that match the given matcher.
  std::vector<Operation *> getMatches(Operation *rootOp, const DynTypedMatcher *matcher) {
    std::vector<Operation *> matches;
    rootOp->walk([&](Operation *subOp) {
      if (matcher->matches(subOp))
        matches.push_back(subOp);
    });
    return matches;
  }

  /// Returns all values that match the given matcher.
  std::vector<Value> getMatchesValue(Operation *rootOp, const DynTypedMatcher *matcher) {
    std::vector<Value> matches;
    rootOp->walk([&](Operation *subOp) {
      for (Value value : subOp->getResults()) {
        if (matcher->matches(value))
          matches.push_back(value);
      }
      });
    return matches;
  }
};

} // namespace matcher
} // namespace query
} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERS_MATCHERSINTERNAL_H
