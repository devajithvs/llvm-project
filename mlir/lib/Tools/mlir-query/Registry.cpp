//===--- Registry.cpp - Matcher registry ------------------===//
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

/// Generate a registry map with all the known matchers.
RegistryMaps::RegistryMaps() {

  // TODO: This list is not complete. It only has non-templated matchers,
  // which are the simplest to add to the system. Templated matchers require
  // more supporting code that was omitted from the first revision for
  // simplicitly of code review.
  using internal::makeMatcherAutoMarshall;
  // clang-format off
  registerMatcher("operation", makeMatcherAutoMarshall(extramatcher::operation, "operation"));
  registerMatcher("isConstant", makeMatcherAutoMarshall((constantFnType *)m_Constant, "m_Constant"));
  registerMatcher("hasAttr", makeMatcherAutoMarshall((attrFnType *)m_Attr, "m_Attr"));
  registerMatcher("hasName", makeMatcherAutoMarshall((opFnType *)m_Op, "m_Op"));
  registerMatcher("m_AnyZeroFloat", makeMatcherAutoMarshall(m_AnyZeroFloat, "m_AnyZeroFloat"));
  registerMatcher("m_PosZeroFloat", makeMatcherAutoMarshall(m_PosZeroFloat, "m_PosZeroFloat"));
  registerMatcher("m_NegZeroFloat", makeMatcherAutoMarshall(m_NegZeroFloat, "m_NegZeroFloat"));
  registerMatcher("m_OneFloat", makeMatcherAutoMarshall(m_OneFloat, "m_OneFloat"));
  registerMatcher("m_PosInfFloat", makeMatcherAutoMarshall(m_PosInfFloat, "m_PosInfFloat"));
  registerMatcher("m_NegInfFloat", makeMatcherAutoMarshall(m_NegInfFloat, "m_NegInfFloat"));
  registerMatcher("m_Zero", makeMatcherAutoMarshall(m_Zero, "m_Zero"));
  registerMatcher("m_NonZero", makeMatcherAutoMarshall(m_NonZero, "m_NonZero"));
  registerMatcher("m_One", makeMatcherAutoMarshall(m_One, "m_One"));
  registerMatcher("m_OneFloat", makeMatcherAutoMarshall(m_OneFloat, "m_OneFloat"));
  registerMatcher("m_OneFloat", makeMatcherAutoMarshall(m_OneFloat, "m_OneFloat"));
  registerMatcher("m_OneFloat", makeMatcherAutoMarshall(m_OneFloat, "m_OneFloat"));
  // clang-format on
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
Matcher *Registry::constructMatcher(StringRef MatcherName,
                                    const SourceRange &NameRange,
                                    ArrayRef<ParserValue> Args,
                                    Diagnostics *Error) {
  ConstructorMap::const_iterator it =
      RegistryData->constructors().find(MatcherName);
  if (it == RegistryData->constructors().end()) {
    Error->addError(NameRange, Error->ET_RegistryMatcherNotFound)
        << MatcherName;
    return NULL;
  }

  return it->second->run(NameRange, Args, Error);
}

} // namespace matcher
} // namespace query
} // namespace mlir