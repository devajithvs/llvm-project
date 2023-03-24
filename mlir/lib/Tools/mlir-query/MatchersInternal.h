#include "mlir/IR/Matchers.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"

namespace mlir {
namespace query {
namespace matcher {

class SingleMatcherInterface : public llvm::RefCountedBase<SingleMatcherInterface> {
public:
  virtual ~SingleMatcherInterface() {}

  /// \brief Returns true if 'op' can be matched.
  virtual bool matches( Operation *op)  = 0;
};

typedef llvm::IntrusiveRefCntPtr<SingleMatcherInterface> SingleMatcherImplementation;

/// \brief Single matcher that takes the matcher as a template argument.
template <typename T>
class SingleMatcher : public SingleMatcherInterface {
public:
  SingleMatcher(T &matcher)
      : Matcher(matcher) {}
  bool matches( Operation *op)  override;
  T Matcher;
};

template <typename T>
bool SingleMatcher<T>::matches( Operation *op)  {
  return Matcher.match(op);
}

} // namespace matcher
} // namespace query
} // namespace mlir
