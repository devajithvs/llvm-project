//===- Registry.cpp - Matcher registry ------------------------------------===//
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
#include <utility>

#include "llvm/Support/Debug.h"
using llvm::dbgs;

#define DEBUG_TYPE "mlir-query"
#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
namespace query {
namespace matcher {
namespace {

using internal::MatcherCreateCallback;
using ConstructorMap = llvm::StringMap<const MatcherCreateCallback *>;

using constantFnType = detail::constant_op_matcher();
using attrFnType = detail::AttrOpMatcher(StringRef);
using opFnType = detail::NameOpMatcher(StringRef);

class RegistryMaps {
public:
  RegistryMaps();
  ~RegistryMaps();

  const ConstructorMap &constructors() const { return Constructors; }

private:
  void registerMatcher(StringRef MatcherName, MatcherCreateCallback *Callback);
  ConstructorMap Constructors;
};

void RegistryMaps::registerMatcher(StringRef MatcherName,
                                   MatcherCreateCallback *Callback) {
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
    registerMatcher(name, makeMatcherAutoMarshall<Operation *>(matcher, name));
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
    delete it->second;
  }
}

static llvm::ManagedStatic<RegistryMaps> RegistryData;

} // anonymous namespace

// static
DynMatcher *Registry::constructMatcher(StringRef MatcherName,
                                       const SourceRange &NameRange,
                                       ArrayRef<ParserValue> Args,
                                       Diagnostics *Error) {
  ConstructorMap::const_iterator it =
      RegistryData->constructors().find(MatcherName);
  if (it == RegistryData->constructors().end()) {
    Error->addError(NameRange, Error->ET_RegistryMatcherNotFound)
        << MatcherName;
    return nullptr;
  }

  return it->second->run(NameRange, Args, Error);
}

// static
DynMatcher *Registry::constructMatcherWrapper(
    StringRef MatcherName, const SourceRange &NameRange, bool ExtractFunction,
    StringRef FunctionName, ArrayRef<ParserValue> Args, Diagnostics *Error) {

  DynMatcher *Out = constructMatcher(MatcherName, NameRange, Args, Error);
  if (!Out)
    return Out;
  Out->setExtract(ExtractFunction);
  Out->setFunctionName(FunctionName);
  return Out;
}

} // namespace matcher
} // namespace query
} // namespace mlir