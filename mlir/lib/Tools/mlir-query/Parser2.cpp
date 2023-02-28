//===- Parser.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Parser.h"
#include "Lexer.h"
#include <string>
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mlir::query;

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

namespace {
class Parser {
public:
//   Parser(ast::Context &ctx, llvm::SourceMgr &sourceMgr,
//          bool enableDocumentation, CodeCompleteContext *codeCompleteContext)
//       : ctx(ctx) {}

  /// Try to parse a new module. Returns nullptr in the case of failure.
  FailureOr<Operation *> parseModule();

private:
  /// The current context of the parser. It allows for the parser to know a bit
  /// about the construct it is nested within during parsing. This is used
  /// specifically to provide additional verification during parsing, e.g. to
  /// prevent using rewrites within a match context, matcher constraints within
  /// a rewrite section, etc.
  enum class ParserContext {
    /// The parser is in the global context.
    Global,
    /// The parser is currently within a Constraint, which disallows all types
    /// of rewrites (e.g. `erase`, `replace`, calls to Rewrites, etc.).
    Constraint,
    /// The parser is currently within the matcher portion of a Pattern, which
    /// is allows a terminal operation rewrite statement but no other rewrite
    /// transformations.
    PatternMatch,
    /// The parser is currently within a Rewrite, which disallows calls to
    /// constraints, requires operation expressions to have names, etc.
    Rewrite,
  };

  /// The current specification context of an operations result type. This
  /// indicates how the result types of an operation may be inferred.
  enum class OpResultTypeContext {
    /// The result types of the operation are not known to be inferred.
    Explicit,
    /// The result types of the operation are inferred from the root input of a
    /// `replace` statement.
    Replacement,
    /// The result types of the operation are inferred by using the
    /// `InferTypeOpInterface` interface provided by the operation.
    Interface,
  };

  //===--------------------------------------------------------------------===//
  // Parsing
  //===--------------------------------------------------------------------===//

//   /// Push a new decl scope onto the lexer.
//   ast::DeclScope *pushDeclScope() {
//     ast::DeclScope *newScope =
//         new (scopeAllocator.Allocate()) ast::DeclScope(curDeclScope);
//     return (curDeclScope = newScope);
//   }
//   void pushDeclScope(ast::DeclScope *scope) { curDeclScope = scope; }

//   /// Pop the last decl scope from the lexer.
//   void popDeclScope() { curDeclScope = curDeclScope->getParentScope(); }

  /// Parse the body of an AST module.
  LogicalResult parseModuleBody(SmallVectorImpl<Operation *> &decls);
};
} // namespace

FailureOr<Operation *> Parser::parseModule() {
//   SMLoc moduleLoc = curToken.getStartLoc();
//   pushDeclScope();

//   // Parse the top-level decls of the module.
//   SmallVector<ast::Decl *> decls;
//   if (failed(parseModuleBody(decls)))
//     return popDeclScope(), failure();

//   popDeclScope();
//   return ast::Module::create(ctx, moduleLoc, decls);
  return failure();
}

LogicalResult Parser::parseModuleBody(SmallVectorImpl<Operation *> &decls) {
//   while (curToken.isNot(Token::eof)) {
//     if (curToken.is(Token::directive)) {
//       if (failed(parseDirective(decls)))
//         return failure();
//       continue;
//     }

//     FailureOr<ast::Decl *> decl = parseTopLevelDecl();
//     if (failed(decl))
//       return failure();
//     decls.push_back(*decl);
//   }
  return success();
}

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

FailureOr<mlir::detail::name_op_matcher>
mlir::query::parseQuery(StringRef &MatcherCode) {
  Parser parser;
  return parser.parseModule();
}
