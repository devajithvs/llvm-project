#ifndef MLIR_TOOLS_MLIRQUERY_SOURCELOCATION_H
#define MLIR_TOOLS_MLIRQUERY_SOURCELOCATION_H

namespace mlir::query::matcher::internal {

// Represents the line and column numbers in a source query.
struct SourceLocation {
  unsigned line{};
  unsigned column{};
};

// Represents a range in a source query, defined by its start and end locations.
struct SourceRange {
  SourceLocation start{};
  SourceLocation end{};
};
} // namespace mlir::query::matcher::internal

#endif