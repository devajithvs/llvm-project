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

#include "MatcherRegistry.h"
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

  const ConstructorMap &constructors() const { return Constructors; }

private:
  void registerMatcher(StringRef MatcherName,
                       std::unique_ptr<MatcherDescriptor> Callback);
  ConstructorMap Constructors;
};

} // namespace

void RegistryMaps::registerMatcher(
    StringRef MatcherName, std::unique_ptr<MatcherDescriptor> Callback) {
  assert(!Constructors.contains(MatcherName));
  Constructors[MatcherName] = std::move(Callback);
}

// Generate a registry map with all the known matchers.
RegistryMaps::RegistryMaps() {

  // TODO: This list is not complete. It only has non-templated matchers,
  // which are the simplest to add to the system. Templated matchers require
  // more supporting code that was omitted from the first revision for
  // simplicitly of code review.
  using internal::makeMatcherAutoMarshall;

  // Define a template function to register operation matchers
  auto registerOpMatcher = [&](const std::string &name, auto matcher) {
    registerMatcher(name, makeMatcherAutoMarshall(matcher, name));
  };

  // Register matchers using the template function
  registerOpMatcher("allOf", extramatcher::allOf);
  registerOpMatcher("anyOf", extramatcher::anyOf);
  registerOpMatcher("hasArgument", extramatcher::hasArgument);
  registerOpMatcher("definedBy", extramatcher::definedBy);
  registerOpMatcher("getDefinitions", extramatcher::getDefinitions);
  registerOpMatcher("getAllDefinitions", extramatcher::getAllDefinitions);
  registerOpMatcher("uses", extramatcher::uses);
  registerOpMatcher("getUses", extramatcher::getUses);
  registerOpMatcher("getAllUses", extramatcher::getAllUses);
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

static llvm::ManagedStatic<RegistryMaps> RegistryData;

internal::MatcherDescriptorPtr::MatcherDescriptorPtr(MatcherDescriptor *Ptr)
    : Ptr(Ptr) {}

internal::MatcherDescriptorPtr::~MatcherDescriptorPtr() { delete Ptr; }

bool Registry::isBuilderMatcher(MatcherCtor Ctor) {
  return Ctor->isBuilderMatcher();
}

internal::MatcherDescriptorPtr
Registry::buildMatcherCtor(MatcherCtor Ctor, SourceRange NameRange,
                           ArrayRef<ParserValue> Args, Diagnostics *Error) {
  return internal::MatcherDescriptorPtr(
      Ctor->buildMatcherCtor(NameRange, Args, Error).release());
}

std::optional<MatcherCtor> Registry::lookupMatcherCtor(StringRef MatcherName) {
  auto it = RegistryData->constructors().find(MatcherName);
  return it == RegistryData->constructors().end() ? std::optional<MatcherCtor>()
                                                  : it->second.get();
}

std::vector<ArgKind> Registry::getAcceptedCompletionTypes(
    ArrayRef<std::pair<MatcherCtor, unsigned>> Context) {
  // Starting with the above seed of acceptable top-level matcher types, compute
  // the acceptable type set for the argument indicated by each context element.
  std::set<ArgKind> typeSet;
  typeSet.insert(ArgKind(ArgKind::AK_Matcher));
  for (const auto &CtxEntry : Context) {
    MatcherCtor Ctor = CtxEntry.first;
    unsigned ArgNumber = CtxEntry.second;
    std::vector<ArgKind> NextTypeSet;
    if ((Ctor->isVariadic() || ArgNumber < Ctor->getNumArgs()))
      Ctor->getArgKinds(ArgNumber, NextTypeSet);
    typeSet.insert(NextTypeSet.begin(), NextTypeSet.end());
  }
  return std::vector<ArgKind>(typeSet.begin(), typeSet.end());
}

std::vector<MatcherCompletion>
Registry::getMatcherCompletions(ArrayRef<ArgKind> AcceptedTypes) {
  std::vector<MatcherCompletion> Completions;

  // Search the registry for acceptable matchers.
  for (const auto &M : RegistryData->constructors()) {
    const MatcherDescriptor &Matcher = *M.getValue();
    StringRef Name = M.getKey();

    unsigned NumArgs = Matcher.isVariadic() ? 1 : Matcher.getNumArgs();
    std::vector<std::vector<ArgKind>> ArgsKinds(NumArgs);
    for (const ArgKind &Kind : AcceptedTypes) {
      if (Kind.getArgKind() != Kind.AK_Matcher) {
        continue;
      }

      for (unsigned Arg = 0; Arg != NumArgs; ++Arg)
        Matcher.getArgKinds(Arg, ArgsKinds[Arg]);
    }

    std::string Decl;
    llvm::raw_string_ostream OS(Decl);

    std::string TypedText = std::string(Name);

    OS << "Matcher: " << Name << "(";
    for (const std::vector<ArgKind> &Arg : ArgsKinds) {
      if (&Arg != &ArgsKinds[0])
        OS << ", ";

      bool FirstArgKind = true;
      // Two steps. First all non-matchers, then matchers only.
      for (const ArgKind &AK : Arg) {
        if (!FirstArgKind)
          OS << "|";
        FirstArgKind = false;
        OS << AK.asString();
      }
    }

    if (Matcher.isVariadic())
      OS << "...";
    OS << ")";

    TypedText += "(";
    if (ArgsKinds.empty())
      TypedText += ")";
    else if (ArgsKinds[0][0].getArgKind() == ArgKind::AK_String)
      TypedText += "\"";

    Completions.emplace_back(TypedText, OS.str());
  }

  return Completions;
}

// static
VariantMatcher Registry::constructMatcher(MatcherCtor Ctor,
                                          SourceRange NameRange,
                                          ArrayRef<ParserValue> Args,
                                          Diagnostics *Error) {
  return Ctor->create(NameRange, Args, Error);
}

// static
VariantMatcher Registry::constructMatcherWrapper(
    MatcherCtor Ctor, SourceRange NameRange, bool ExtractFunction,
    StringRef FunctionName, ArrayRef<ParserValue> Args, Diagnostics *Error) {

  LLVM_DEBUG(DBGS() << "pre constructMatcher"
                    << "\n");
  VariantMatcher Out = constructMatcher(Ctor, NameRange, Args, Error);
  LLVM_DEBUG(DBGS() << "post constructMatcher"
                    << "\n");
  if (Out.isNull())
    return Out;

  LLVM_DEBUG(DBGS() << "pre getSingleMatcher"
                    << "\n");
  std::optional<DynMatcher> Result = Out.getSingleMatcher();
  LLVM_DEBUG(DBGS() << "post getSingleMatcher"
                    << "\n");
  if (Result.has_value()) {
    Result->setExtract(ExtractFunction);
    Result->setFunctionName(FunctionName);
    if (Result.has_value()) {
      return VariantMatcher::SingleMatcher(*Result);
    }
  }
  Error->addError(NameRange, Error->ET_RegistryNotBindable);
  return Out;
}

// static
VariantMatcher Registry::constructBoundMatcher(MatcherCtor Ctor,
                                               SourceRange NameRange,
                                               StringRef BindID,
                                               ArrayRef<ParserValue> Args,
                                               Diagnostics *Error) {
  VariantMatcher Out = constructMatcher(Ctor, NameRange, Args, Error);
  if (Out.isNull())
    return Out;

  std::optional<DynMatcher> Result = Out.getSingleMatcher();
  if (Result.has_value()) {
    // FIXME
    // std::optional<DynMatcher> Bound = Result->tryBind(BindID);
    // if (Bound.has_value()) {
    return VariantMatcher::SingleMatcher(*Result);
    // }
  }
  Error->addError(NameRange, Error->ET_RegistryNotBindable);
  return VariantMatcher();
}

} // namespace matcher
} // namespace query
} // namespace mlir