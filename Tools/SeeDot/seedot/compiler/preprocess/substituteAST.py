# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''
unroller can be used to unroll loop/sum 
'''

from seedot.compiler.antlr.seedotParser import seedotParser as SeeDotParser

import seedot.compiler.ast.ast as AST
from seedot.compiler.ast.astVisitor import ASTVisitor
from seedot.writer import Writer

import os

class SubstituteAST(ASTVisitor):

    def __init__(self, indexToAppend, doNotSubstitute=[]):
        self.indexToAppend = indexToAppend
        self.doNotSubstitute = doNotSubstitute
        self.declaredVars = []

    def visitInt(self, node: AST.Int):
        return AST.Int(node.value)

    def visitFloat(self, node: AST.Float):
        return AST.Int(node.value)

    def visitId(self, node: AST.ID):
        if node.name in self.declaredVars:
            return AST.ID(node.name + "_" + str(self.indexToAppend))
        else:
            return AST.ID(node.name)

    def visitDecl(self, node: AST.Decl):
        return AST.Decl(node.shape, node.range)

    def visitInit(self, node: AST.Init):
        return AST.Init(node.shape, node.value)

    def visitTransp(self, node: AST.Transp):
        return AST.Transp(self.visit(node.expr))

    def visitReshape(self, node: AST.Reshape):
        return AST.Reshape(self.visit(node.expr), node.shape, node.order)

    def visitMaxpool(self, node: AST.Maxpool):
        return AST.Maxpool(self.visit(node.expr), node.dim)

    def visitIndex(self, node: AST.Index):
        return AST.Index(self.visit(node.expr), self.visit(node.index))

    def visitFuncCall(self, node: AST.Index):
        exprList = []
        for expr in node.exprList:
            exprList.append(self.visit(expr))
        return AST.FuncCall(node.name, exprList)

    def visitUop(self, node: AST.Uop):
        return AST.Uop(node.op, self.visit(node.expr))

    def visitBop1(self, node: AST.Bop1):
        return AST.Bop1(self.visit(node.expr1), node.op, self.visit(node.expr2))

    def visitBop2(self, node: AST.Bop2):
        return AST.Bop2(self.visit(node.expr1), node.op, self.visit(node.expr2))

    def visitFunc(self, node: AST.Func):
        return AST.Func(node.op, self.visit(node.expr))

    def visitSum(self, node: AST.Sum):
        return AST.Sum(node.name + ("_" + str(self.indexToAppend) if node.name in self.declaredVars else ""), node.start, node.end, self.visit(node.expr))

    def visitSumUnroll(self, node: AST.SumUnroll):
        return AST.SumUnroll(node.name + ("_" + str(self.indexToAppend) if node.name in self.declaredVars else ""), node.start, node.end, node.unrollFactor, self.visit(node.expr))

    def visitLoop(self, node: AST.Loop):
        return AST.Loop(node.name + ("_" + str(self.indexToAppend) if node.name in self.declaredVars else ""), node.start, node.end, self.visit(node.mutableVar), self.visit(node.expr))

    def visitLoopUnroll(self, node: AST.LoopUnroll):
        return AST.LoopUnroll(node.name + ("_" + str(self.indexToAppend) if node.name in self.declaredVars else ""), node.start, node.end, self.visit(node.mutableVar), node.unrollFactor, self.visit(node.expr))

    def visitCond(self, node: AST.Cond):
        return AST.Cond(self.visit(node.expr), node.num, self.visit(node.trueBlock), self.visit(node.falseBlock))

    def visitLet(self, node: AST.Let):
        declVisit = self.visit(node.decl)
        if node.name not in self.doNotSubstitute:
            self.declaredVars.append(node.name)
        return AST.Let(node.name + ("_" + str(self.indexToAppend) if node.name in self.declaredVars else ""), declVisit, self.visit(node.expr))
