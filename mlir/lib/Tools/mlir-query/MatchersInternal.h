#include "mlir/IR/Matchers.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"

namespace mlir {
namespace query {
namespace matcher {

class MatcherInterface;
typedef llvm::IntrusiveRefCntPtr<MatcherInterface> MatcherImplementation;
class MatcherInterface : public llvm::RefCountedBase<MatcherInterface> {
public:
  virtual ~MatcherInterface() = default;

  /// \brief Returns true if 'op' can be matched.
  virtual bool matches( Operation *op)  = 0;

  /// \brief Makes a copy of this matcher object.
  virtual MatcherImplementation clone() const = 0;
};


/// The kind provided to the constructor overrides any kind that may be
/// specified by the `InnerMatcher`.
class TraversalMatcherImpl : public MatcherInterface {
public:
  TraversalMatcherImpl(
      llvm::IntrusiveRefCntPtr<MatcherInterface> InnerMatcher)
      : InnerMatcher(std::move(InnerMatcher)) {}

  bool matches(Operation *op) override {
    return this->InnerMatcher->matches(op);
  }

private:
  llvm::IntrusiveRefCntPtr<MatcherInterface> InnerMatcher;
};

class SingleMatcherInterface : public MatcherInterface {
public:
  /// \brief Returns true if 'op' can be matched.
  virtual bool matches( Operation *op) override = 0;

  virtual MatcherImplementation clone() const override = 0;

};
//typedef llvm::IntrusiveRefCntPtr<SingleMatcherInterface> SingleMatcherImplementation;

/// \brief Single matcher that takes the matcher as a template argument.
template <typename T>
class SingleMatcher : public SingleMatcherInterface {
public:
  SingleMatcher(T &matcher)
      : Matcher(matcher) {}
  bool matches( Operation *op) override {
    return Matcher.match(op);
  }

  /// \brief Makes a copy of this matcher object.
  MatcherImplementation clone() const override { return new SingleMatcher<T>(*this); }

  T Matcher;
};


class MatchFinder {
public:
    std::vector<Operation *> getMatches(Operation *f, const MatcherImplementation &Implementation) {
        std::vector<Operation *> matches;
        f->walk([&matches, &Implementation](Operation *op) {
            if (Implementation->matches(op)) {
            matches.push_back(op);
            }
        });
        return matches;
    }
private:
  llvm::IntrusiveRefCntPtr<MatcherInterface> Implementation;
};

} // namespace matcher
} // namespace query
} // namespace mlir
