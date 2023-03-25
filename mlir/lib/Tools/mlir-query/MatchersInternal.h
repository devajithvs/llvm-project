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
};


/// Matcher that works on a \c DynTypedNode.
///
/// It is constructed from a \c Matcher<T> object and redirects most calls to
/// underlying matcher.
/// It checks whether the \c DynTypedNode is convertible into the type of the
/// underlying matcher and then do the actual match on the actual node, or
/// return false if it is not convertible.
class Matcher {
public:
  Matcher(MatcherInterface *Implementation)
      : Implementation(Implementation) {}

  /// Returns true if the matcher matches the given \c op.
  bool matches(Operation *op) const {
    return Implementation->matches(op);
  }

  Matcher *clone() const { return new Matcher(*this); }
private:
  llvm::IntrusiveRefCntPtr<MatcherInterface> Implementation;
};

class SingleMatcherInterface : public MatcherInterface {
public:
  /// \brief Returns true if 'op' can be matched.
  virtual bool matches( Operation *op) override = 0;

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
