# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''
PrintAST can be used to print the generated AST for a given SeeDot program.
'''

import numpy as np
import os

from seedot.compiler.antlr.seedotParser import seedotParser as SeeDotParser

import seedot.compiler.ast.ast as AST
from seedot.compiler.ast.astVisitor import ASTVisitor


class EvaluateAST(ASTVisitor):

    def visitInt(self, node: AST.Int, env):
        return node.value

    def visitFloat(self, node: AST.Float, env):
        raise NotImplementedError

    def visitId(self, node: AST.ID, env):
        return env[node.name]

    def visitDecl(self, node: AST.Decl, env):
        pass

    def visitInit(self, node: AST.Init, env):
        raise NotImplementedError

    def visitTransp(self, node: AST.Transp, env):
        raise NotImplementedError

    def visitReshape(self, node: AST.Reshape, env):
        raise NotImplementedError

    def visitMaxpool(self, node: AST.Maxpool, env):
        raise NotImplementedError

    def visitIndex(self, node: AST.Index, env):
        raise NotImplementedError

    def visitFuncCall(self, node: AST.Index, env):
        raise NotImplementedError

    def visitUop(self, node: AST.Uop, env):
        raise NotImplementedError

    def visitBop1(self, node: AST.Bop1, env):
        if node.op == SeeDotParser.MUL:
            a = self.visit(node.expr1, env)
            b = self.visit(node.expr2, env)

            if isinstance(node.expr2, AST.Int):
                c = np.multiply(a, b)
            else:
                c = np.matmul(a, b)
            return c
        elif node.op == SeeDotParser.DIV:
            a = self.visit(node.expr1, env)
            b = self.visit(node.expr2, env)
            c = np.floor_divide(a, b)
            return c
        else:
            raise NotImplementedError

    def visitBop2(self, node: AST.Bop2, env):
        assert node.op == SeeDotParser.ADD
        a = self.visit(node.expr1, env)
        b = self.visit(node.expr2, env)
        c = np.add(a, b)
        return c

    def visitFunc(self, node: AST.Func, env):
        if node.op == SeeDotParser.ROUND:
            a = self.visit(node.expr, env)
            a = np.int8(a)
            #print(a)
            return a
        elif node.op == SeeDotParser.UNROUND:
            a = self.visit(node.expr, env)
            a = np.float(a)
            #print(a)
            return a
        else:
            raise NotImplementedError

    def visitSum(self, node: AST.Sum, env):
        raise NotImplementedError

    def visitLoop(self, node: AST.Loop, env):
        raise NotImplementedError

    def visitCond(self, node: AST.Cond, env):
        raise NotImplementedError

    def visitLet(self, node: AST.Let, env):
        if isinstance(node.decl, AST.Decl):
            if node.name != 'X':
                var = np.load(os.path.join("seedot", "compiler", "input", node.name + ".npy"))
                env[node.name] = var
        else:
            res = self.visit(node.decl, env)
            env[node.name] = res

        return self.visit(node.expr, env)
