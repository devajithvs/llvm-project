
#ifndef MLIR_TOOLS_MLIRQUERY_MATCHER_REGISTRYMAP_H
#define MLIR_TOOLS_MLIRQUERY_MATCHER_REGISTRYMAP_H

#include "Marshallers.h"
#include "llvm/ADT/StringMap.h"
#include <string>
namespace mlir::query::matcher {

using ConstructorMap =
    llvm::StringMap<std::unique_ptr<const internal::MatcherDescriptor>>;

class Registry {
public:
  Registry() = default;
  ~Registry() = default;

  const ConstructorMap &constructors() const { return constructorMap; }

  template <typename MatcherType>
  void registerMatcher(const std::string &name, MatcherType matcher) {
    registerMatcherDescriptor(name,
                              internal::makeMatcherAutoMarshall(matcher, name));
  }

private:
  void registerMatcherDescriptor(
      llvm::StringRef matcherName,
      std::unique_ptr<internal::MatcherDescriptor> callback);

  ConstructorMap constructorMap;
};
} // namespace mlir::query::matcher

#endif