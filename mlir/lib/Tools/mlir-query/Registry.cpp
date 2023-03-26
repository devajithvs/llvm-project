//===--- Registry.cpp - Matcher registry ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Registry map populated at static initialization time.
///
//===----------------------------------------------------------------------===//


#include <utility>

#include "Registry.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ManagedStatic.h"

#include "llvm/Support/Debug.h"
using llvm::dbgs;

#define DEBUG_TYPE "mlir-query"

#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")
namespace mlir {
namespace query {
namespace matcher {
namespace {

using internal::MatcherCreateCallback;

typedef llvm::StringMap<const MatcherCreateCallback *> ConstructorMap;
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

/// \brief Generate a registry map with all the known matchers.
RegistryMaps::RegistryMaps() {
  // TODO: This list is not complete. It only has non-overloaded matchers,
  // which are the simplest to add to the system. Overloaded matchers require
  // more supporting code that was omitted from the first revision for
  // simplicitly of code review.
  registerMatcher("m_Name", internal::makeMatcherAutoMarshall(m_Name, "m_Name"));
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
Matcher *Registry::constructMatcher(StringRef MatcherName, ArrayRef<ParserValue> Args) {
  LLVM_DEBUG(DBGS() << "Running constructMatcher" << "\n");
  ConstructorMap::const_iterator it =
      RegistryData->constructors().find(MatcherName);
  if (it == RegistryData->constructors().end()) {
    return NULL;
  }

  LLVM_DEBUG(DBGS() << "Running constructMatcher" << "\n");
  LLVM_DEBUG(DBGS() << it->second->run(Args) << "\n");
  return it->second->run(Args);
}

}  // namespace matcher
}  // namespace query
}  // namespace mlir