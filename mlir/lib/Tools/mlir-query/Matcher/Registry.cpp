//===- MatcherRegistry.cpp - Matcher registry -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registry map populated at static initialization time.
//
//===----------------------------------------------------------------------===//

#include "Registry.h"
#include "ExtraMatchers.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ManagedStatic.h"
#include <set>
#include <utility>

#include "llvm/Support/Debug.h"
using llvm::dbgs;

#define DEBUG_TYPE "mlir-query"
#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
namespace query {
namespace matcher {
namespace {

using internal::MatcherDescriptor;
using ConstructorMap =
    llvm::StringMap<std::unique_ptr<const MatcherDescriptor>>;

using constantFnType = detail::constant_op_matcher();
using attrFnType = detail::AttrOpMatcher(StringRef);
using opFnType = detail::NameOpMatcher(StringRef);

class RegistryMaps {
public:
  RegistryMaps();
  ~RegistryMaps();

  const ConstructorMap &constructors() const { return constructorMap; }

private:
  void registerMatcher(StringRef matcherName,
                       std::unique_ptr<MatcherDescriptor> callback);
  ConstructorMap constructorMap;
};

} // namespace

void RegistryMaps::registerMatcher(
    StringRef matcherName, std::unique_ptr<MatcherDescriptor> callback) {
  assert(!constructorMap.contains(matcherName));
  constructorMap[matcherName] = std::move(callback);
}

// Generate a registry map with all the known matchers.
RegistryMaps::RegistryMaps() {

  using internal::makeMatcherAutoMarshall;

  // Define a template function to register operation matchers
  auto registerOpMatcher = [&](const std::string &name, auto matcher) {
    registerMatcher(name, makeMatcherAutoMarshall(matcher, name));
  };

  // Register matchers using the template function
  registerOpMatcher("allOf", extramatchers::allOf);
  registerOpMatcher("anyOf", extramatchers::anyOf);
  registerOpMatcher("hasArgument", extramatchers::hasArgument);
  registerOpMatcher("definedBy", extramatchers::definedBy);
  registerOpMatcher("getDefinitions", extramatchers::getDefinitions);
  registerOpMatcher("getAllDefinitions", extramatchers::getAllDefinitions);
  registerOpMatcher("uses", extramatchers::uses);
  registerOpMatcher("getUses", extramatchers::getUses);
  registerOpMatcher("getAllUses", extramatchers::getAllUses);
  registerOpMatcher("isConstantOp", static_cast<constantFnType *>(m_Constant));
  registerOpMatcher("hasOpAttrName", static_cast<attrFnType *>(m_Attr));
  registerOpMatcher("hasOpName", static_cast<opFnType *>(m_Op));
  registerOpMatcher("m_PosZeroFloat", m_PosZeroFloat);
  registerOpMatcher("m_NegZeroFloat", m_NegZeroFloat);
  registerOpMatcher("m_AnyZeroFloat", m_AnyZeroFloat);
  registerOpMatcher("m_OneFloat", m_OneFloat);
  registerOpMatcher("m_PosInfFloat", m_PosInfFloat);
  registerOpMatcher("m_NegInfFloat", m_NegInfFloat);
  registerOpMatcher("m_Zero", m_Zero);
  registerOpMatcher("m_NonZero", m_NonZero);
  registerOpMatcher("m_One", m_One);
}

RegistryMaps::~RegistryMaps() = default;

static llvm::ManagedStatic<RegistryMaps> registryData;

internal::MatcherDescriptorPtr::MatcherDescriptorPtr(MatcherDescriptor *ptr)
    : ptr(ptr) {}

internal::MatcherDescriptorPtr::~MatcherDescriptorPtr() { delete ptr; }

bool Registry::isBuilderMatcher(MatcherCtor ctor) {
  return ctor->isBuilderMatcher();
}

internal::MatcherDescriptorPtr
Registry::buildMatcherCtor(MatcherCtor ctor, SourceRange nameRange,
                           ArrayRef<ParserValue> args, Diagnostics *error) {
  return internal::MatcherDescriptorPtr(
      ctor->buildMatcherCtor(nameRange, args, error).release());
}

std::optional<MatcherCtor> Registry::lookupMatcherCtor(StringRef matcherName) {
  auto it = registryData->constructors().find(matcherName);
  return it == registryData->constructors().end() ? std::optional<MatcherCtor>()
                                                  : it->second.get();
}

std::vector<ArgKind> Registry::getAcceptedCompletionTypes(
    ArrayRef<std::pair<MatcherCtor, unsigned>> context) {
  // Starting with the above seed of acceptable top-level matcher types, compute
  // the acceptable type set for the argument indicated by each context element.
  std::set<ArgKind> typeSet;
  typeSet.insert(ArgKind(ArgKind::AK_Matcher));
  for (const auto &ctxEntry : context) {
    MatcherCtor ctor = ctxEntry.first;
    unsigned argNumber = ctxEntry.second;
    std::vector<ArgKind> nextTypeSet;
    if ((ctor->isVariadic() || argNumber < ctor->getNumArgs()))
      ctor->getArgKinds(argNumber, nextTypeSet);
    typeSet.insert(nextTypeSet.begin(), nextTypeSet.end());
  }
  return std::vector<ArgKind>(typeSet.begin(), typeSet.end());
}

std::vector<MatcherCompletion>
Registry::getMatcherCompletions(ArrayRef<ArgKind> acceptedTypes) {
  std::vector<MatcherCompletion> completions;

  // Search the registry for acceptable matchers.
  for (const auto &m : registryData->constructors()) {
    const MatcherDescriptor &matcher = *m.getValue();
    StringRef name = m.getKey();

    unsigned numArgs = matcher.isVariadic() ? 1 : matcher.getNumArgs();
    std::vector<std::vector<ArgKind>> argKinds(numArgs);
    for (const ArgKind &kind : acceptedTypes) {
      if (kind.getArgKind() != kind.AK_Matcher) {
        continue;
      }

      for (unsigned arg = 0; arg != numArgs; ++arg)
        matcher.getArgKinds(arg, argKinds[arg]);
    }

    std::string decl;
    llvm::raw_string_ostream OS(decl);

    std::string typedText = std::string(name);

    OS << "Matcher: " << name << "(";
    for (const std::vector<ArgKind> &arg : argKinds) {
      if (&arg != &argKinds[0])
        OS << ", ";

      bool firstArgKind = true;
      // Two steps. First all non-matchers, then matchers only.
      for (const ArgKind &argKind : arg) {
        if (!firstArgKind)
          OS << "|";
        firstArgKind = false;
        OS << argKind.asString();
      }
    }

    if (matcher.isVariadic())
      OS << "...";
    OS << ")";

    typedText += "(";
    if (argKinds.empty())
      typedText += ")";
    else if (argKinds[0][0].getArgKind() == ArgKind::AK_String)
      typedText += "\"";

    completions.emplace_back(typedText, OS.str());
  }

  return completions;
}

// static
VariantMatcher Registry::constructMatcher(MatcherCtor ctor,
                                          SourceRange nameRange,
                                          ArrayRef<ParserValue> args,
                                          Diagnostics *error) {
  return ctor->create(nameRange, args, error);
}

// static
VariantMatcher Registry::constructFunctionMatcher(MatcherCtor ctor,
                                                  SourceRange nameRange,
                                                  StringRef FunctionName,
                                                  ArrayRef<ParserValue> args,
                                                  Diagnostics *error) {

  VariantMatcher out = constructMatcher(ctor, nameRange, args, error);
  if (out.isNull())
    return out;

  std::optional<DynMatcher> result = out.getSingleMatcher();

  if (result.has_value()) {
    result->setFunctionName(FunctionName);
    if (result.has_value()) {
      return VariantMatcher::SingleMatcher(*result);
    }
  }
  error->addError(nameRange, error->ET_RegistryNotBindable);
  return out;
}

// static
VariantMatcher Registry::constructBoundMatcher(MatcherCtor ctor,
                                               SourceRange nameRange,
                                               StringRef bindId,
                                               ArrayRef<ParserValue> args,
                                               Diagnostics *error) {
  VariantMatcher out = constructMatcher(ctor, nameRange, args, error);
  if (out.isNull())
    return out;

  std::optional<DynMatcher> result = out.getSingleMatcher();
  if (result.has_value()) {
    // TODO: FIXME
    // std::optional<DynMatcher> Bound = result->tryBind(BindID);
    // if (Bound.has_value()) {
    return VariantMatcher::SingleMatcher(*result);
    // }
  }
  error->addError(nameRange, error->ET_RegistryNotBindable);
  return VariantMatcher();
}

} // namespace matcher
} // namespace query
} // namespace mlir