//===--- MLIRTypeTraits.h -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Provides a dynamic type identifier and a dynamically typed node container
//  that can be used to store an MLIR base node at runtime in the same storage
//  in a type safe way.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_MATCHERS_MLIRTYPETRAITS_H
#define MLIR_TOOLS_MLIRQUERY_MATCHERS_MLIRTYPETRAITS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace query {
namespace matcher {

// MLIRNodeKind can be constructed from any node kind and allows for runtime type
// hierarchy checks. Use getFromNodeKind<T>() to construct them.
class MLIRNodeKind {
public:
  // Empty identifier. It matches nothing.
  constexpr MLIRNodeKind() : KindId(NKI_None) {}

  // Construct an identifier for T.
  template <class T>
  static constexpr MLIRNodeKind getFromNodeKind() {
    return MLIRNodeKind(KindToKindId<T>::Id);
  }

  // Returns true if this and Other represent the same kind.
  constexpr bool isSame(MLIRNodeKind Other) const {
    return KindId != NKI_None && KindId == Other.KindId;
  }

  // Returns true only for the default MLIRNodeKind()
  constexpr bool isNone() const { return KindId == NKI_None; }

private:
  // Kind ids.
  // Includes all possible base and derived kinds.
  enum NodeKindId { NKI_None, NKI_Attr, NKI_Value, NKI_Operation };

  // Use getFromNodeKind<T>() to construct the kind.
  constexpr MLIRNodeKind(NodeKindId KindId) : KindId(KindId) {}

  // Helper meta-function to convert a kind T to its enum value.
  // This struct is specialized below for all known kinds.
  template <class T>
  struct KindToKindId {
    static const NodeKindId Id = NKI_None;
  };
  template <class T>
  struct KindToKindId<const T> : KindToKindId<T> {};

  NodeKindId KindId;
};

template <>
struct MLIRNodeKind::KindToKindId<Operation *> {
  static const NodeKindId Id = NKI_Operation;
};

// A dynamically typed MLIR node container.
//
// Stores an MLIR node in a type safe way. This allows writing code that
// works with different kinds of MLIR nodes, despite the fact that they don't
// have a common base class.
//
// Use create(Node) to create a DynTypedNode from an MLIR node,
// and get<T>() to retrieve the node as type T if the types match.
//
// See MLIRNodeKind for which node base types are currently supported;
// You can create DynTypedNodes for all nodes in the inheritance hierarchy of
// the supported base types.
class DynTypedNode {
public:
  // Creates a DynTypedNode from Node.
  template <typename T>
  static DynTypedNode create(T &Node) {
    return BaseConverter<T>::create(Node);
  }

  // Retrieve the stored node as type T.
  // Returns NULL if the stored node does not have a type that is
  // convertible to T.
  template <typename T>
  T *get() {
    return BaseConverter<T>::get(NodeKind, &Storage);
  }

  // Retrieve the stored node as type T.
  // Similar to get(), but asserts that the type is what we are expecting.
  template <typename T>
  T &getUnchecked() const {
    return BaseConverter<T>::getUnchecked(NodeKind, &Storage);
  }

  MLIRNodeKind getNodeKind() const { return NodeKind; }

private:
  // Takes care of converting from and to T.
  template <typename T, typename EnablerT = void>
  struct BaseConverter;

  // Converter that stores T (by value).
  template <typename T>
  struct ValueConverter {
    static T *get(MLIRNodeKind NodeKind, const void *Storage) {
      if (MLIRNodeKind::getFromNodeKind<T>().isSame(NodeKind))
        return const_cast<T *>(reinterpret_cast<const T *>(Storage));
      return nullptr;
    }
    static T &getUnchecked(MLIRNodeKind NodeKind, const void *Storage) {
      assert(MLIRNodeKind::getFromNodeKind<T>().isSame(NodeKind));
      return *const_cast<T *>(reinterpret_cast<const T *>(Storage));
    }
    static DynTypedNode create(T &Node) {
      DynTypedNode Result;
      Result.NodeKind = MLIRNodeKind::getFromNodeKind<T>();
      new (&Result.Storage) T(Node);
      return Result;
    }
  };

  MLIRNodeKind NodeKind;
  // Stores the data of the node.
  // Note that we can store Operation and Value by pointer as they are
  // guaranteed to be unique pointers pointing to dedicated storage in the
  // MLIR.
  llvm::AlignedCharArrayUnion<Operation *, Value> Storage;
};

template <>
struct DynTypedNode::BaseConverter<Operation *, void>
    : public ValueConverter<Operation *> {};

template <>
struct DynTypedNode::BaseConverter<Value, void> : public ValueConverter<Value> {
};

} // namespace matcher
} // namespace query
} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERS_MLIRTYPETRAITS_H
