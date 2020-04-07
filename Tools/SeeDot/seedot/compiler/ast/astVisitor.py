# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''
ASTVisitor contains the dynamic dispatcher used by printAST.py
'''

import seedot.compiler.ast.ast as AST


class ASTVisitor:

    def visit(self, node, *args):
        if isinstance(node, AST.Int):
            return self.visitInt(node, *args)
        elif isinstance(node, AST.Float):
            return self.visitFloat(node, *args)
        elif isinstance(node, AST.ID):
            return self.visitId(node, *args)
        elif isinstance(node, AST.Decl):
            return self.visitDecl(node, *args)
        elif isinstance(node, AST.Init):
            return self.visitInit(node, *args)
        elif isinstance(node, AST.Transp):
            return self.visitTransp(node, *args)
        elif isinstance(node, AST.Reshape):
            return self.visitReshape(node, *args)
        elif isinstance(node, AST.Maxpool):
            return self.visitMaxpool(node, *args)
        elif isinstance(node, AST.Index):
            return self.visitIndex(node, *args)
        elif isinstance(node, AST.FuncCall):
            return self.visitFuncCall(node, *args)
        elif isinstance(node, AST.Uop):
            return self.visitUop(node, *args)
        elif isinstance(node, AST.Bop1):
            return self.visitBop1(node, *args)
        elif isinstance(node, AST.Bop2):
            return self.visitBop2(node, *args)
        elif isinstance(node, AST.Func):
            return self.visitFunc(node, *args)
        elif isinstance(node, AST.Sum):
            return self.visitSum(node, *args)
        elif isinstance(node, AST.Loop):
            return self.visitLoop(node, *args)
        elif isinstance(node, AST.Cond):
            return self.visitCond(node, *args)
        elif isinstance(node, AST.Let):
            return self.visitLet(node, *args)
        else:
            assert False
