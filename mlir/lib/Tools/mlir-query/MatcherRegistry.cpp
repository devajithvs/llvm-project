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
using ConstructorMap = llvm::StringMap<const MatcherDescriptor *>;

using constantFnType = detail::constant_op_matcher();
using attrFnType = detail::AttrOpMatcher(StringRef);
using opFnType = detail::NameOpMatcher(StringRef);

class RegistryMaps {
public:
  RegistryMaps();
  ~RegistryMaps();

  const ConstructorMap &constructors() const { return Constructors; }

private:
  void registerMatcher(StringRef MatcherName, MatcherDescriptor *Callback);
  ConstructorMap Constructors;
};

void RegistryMaps::registerMatcher(StringRef MatcherName,
                                   MatcherDescriptor *Callback) {
  assert(Constructors.find(MatcherName) == Constructors.end());
  Constructors[MatcherName] = Callback;
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

RegistryMaps::~RegistryMaps() {
  for (ConstructorMap::iterator it = Constructors.begin(),
                                end = Constructors.end();
       it != end; ++it) {
        // TODO
    // delete it->second;
  }
}

static llvm::ManagedStatic<RegistryMaps> RegistryData;

} // anonymous namespace

std::optional<MatcherCtor>
Registry::lookupMatcherCtor(StringRef MatcherName, const SourceRange &NameRange,
                            Diagnostics *Error) {
  llvm::errs() << "registry lookupMatcherCtor" << "\n";
  ConstructorMap::const_iterator it =
      RegistryData->constructors().find(MatcherName);
  llvm::errs() << "registry lookupMatcherCtor found" << "\n";
  if (it == RegistryData->constructors().end()) {
    Error->addError(NameRange, Error->ET_RegistryMatcherNotFound) << MatcherName;
    return std::optional<MatcherCtor>();
  }
  llvm::errs() << "registry lookupMatcherCtor: " << it->first() << ": " << (int*)it->second << "\n";
  return it->second;
}

// static
VariantMatcher Registry::constructMatcher(MatcherCtor Ctor,
                                       const SourceRange &NameRange,
                                       ArrayRef<ParserValue> Args,
                                       Diagnostics *Error) {
  llvm::errs() << "Ctor->create()"  << "\n";
  return Ctor->create(NameRange, Args, Error);
}

// static
VariantMatcher Registry::constructMatcherWrapper(
    MatcherCtor Ctor, const SourceRange &NameRange, bool ExtractFunction,
    StringRef FunctionName, ArrayRef<ParserValue> Args, Diagnostics *Error) {
  
  LLVM_DEBUG(DBGS() << "pre constructMatcher"
                    << "\n");
  VariantMatcher Out = constructMatcher(Ctor, NameRange, Args, Error);
  LLVM_DEBUG(DBGS() << "post constructMatcher"
                    << "\n");
  if (Out.isNull()) return Out;
  
  LLVM_DEBUG(DBGS() << "pre getSingleMatcher"
                    << "\n");
  llvm::errs() << "getSingleMatcher\n";
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
  llvm::errs() << "constructMatcherWrapper failed"  << "\n";

  Error->addError(NameRange, Error->ET_RegistryNotBindable);
  return Out;
}

// static
VariantMatcher Registry::constructBoundMatcher(MatcherCtor Ctor,
                                               const SourceRange &NameRange,
                                               StringRef BindID,
                                               ArrayRef<ParserValue> Args,
                                               Diagnostics *Error) {
  VariantMatcher Out = constructMatcher(Ctor, NameRange, Args, Error);
  if (Out.isNull()) return Out;

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