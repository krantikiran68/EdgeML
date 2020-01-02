# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''
unroller can be used to unroll loop/sum 
'''

from seedot.compiler.antlr.seedotParser import seedotParser as SeeDotParser

import seedot.compiler.ast.ast as AST
from seedot.compiler.ast.astVisitor import ASTVisitor
from seedot.writer import Writer

import seedot.compiler.preprocess.substituteAST as subs

import os

indent = "  "

class Unroller(ASTVisitor):

    def __init__(self, outputFile):
        self.outputFile = Writer(outputFile + ".sd")
        self.newline = True
        self.curDepth = 0
        self.depthToVar = {}
        self.depthToBreadth = {}

    def visitInt(self, node: AST.Int):
        self.outputFile.printf(str(node.value), indent=self.isNewLine())

    def visitFloat(self, node: AST.Float):
        self.outputFile.printf(str(node.value), indent=self.isNewLine())

    def visitId(self, node: AST.ID):
        self.outputFile.printf(node.name, indent=self.isNewLine())

    def visitDecl(self, node: AST.Decl):
        self.outputFile.printf("(", indent=self.isNewLine())
        for i in range(len(node.shape)):
            self.outputFile.printf(str(node.shape[i]), indent=self.isNewLine())
            if i + 1 < len(node.shape):
                self.outputFile.printf(",", indent=self.isNewLine())
        self.outputFile.printf(") in [", indent=self.isNewLine())
        for i in range(len(node.range)):
            self.outputFile.printf(str(node.range[i]), indent=self.isNewLine())
            if i + 1 < len(node.range):
                self.outputFile.printf(",", indent=self.isNewLine())
        self.outputFile.printf("]", indent=self.isNewLine())

    def visitInit(self, node: AST.Init):
        self.outputFile.printf("init([", indent=self.isNewLine())
        for i in range(len(node.shape)):
            self.outputFile.printf(str(node.shape[i]), indent=self.isNewLine())
            if i + 1 < len(node.shape):
                self.outputFile.printf(",", indent=self.isNewLine())
        self.outputFile.printf("], %f)" %(node.value), indent=self.isNewLine())

    def visitTransp(self, node: AST.Transp):
        self.visit(node.expr)
        self.outputFile.printf("^T ", indent=self.isNewLine())

    def visitReshape(self, node: AST.Reshape):
        self.outputFile.printf("reshape(", indent=self.isNewLine())
        self.visit(node.expr)
        self.outputFile.printf(") ", indent=self.isNewLine())

    def visitMaxpool(self, node: AST.Maxpool):
        self.outputFile.printf("maxpool(", indent=self.isNewLine())
        self.visit(node.expr)
        self.outputFile.printf(") ", indent=self.isNewLine())

    def visitIndex(self, node: AST.Index):
        self.visit(node.expr)
        self.outputFile.printf("[", indent=self.isNewLine())
        self.visit(node.index)
        self.outputFile.printf("]", indent=self.isNewLine())

    def visitFuncCall(self, node: AST.Index):
        self.outputFile.printf(node.id + "(", indent=self.isNewLine())
        for i in range(len(node.exprList)):
            expr = node.exprList[i]
            self.visit(expr)
            if i + 1 < len(node.exprList):
                self.outputFile.printf(",", indent=self.isNewLine())
        self.outputFile.printf(")", indent=self.isNewLine())

    def visitUop(self, node: AST.Uop):
        if node.op == SeeDotParser.SUB:
            self.outputFile.printf("-", indent=self.isNewLine())
        self.visit(node.expr)

    def visitBop1(self, node: AST.Bop1):
        self.visit(node.expr1)
        if node.op == SeeDotParser.MUL:
            self.outputFile.printf(" * ", indent=self.isNewLine())
        elif node.op == SeeDotParser.SPARSEMUL:
            self.outputFile.printf(" |*| ", indent=self.isNewLine())
        elif node.op == SeeDotParser.MULCIR:
            self.outputFile.printf(" <*> ", indent=self.isNewLine())
        elif node.op == SeeDotParser.CONV:
            self.outputFile.printf(" # ", indent=self.isNewLine())
        elif node.op == SeeDotParser.ADDCIR:
            self.outputFile.printf(" <+> ", indent=self.isNewLine())
        elif node.op == SeeDotParser.SUBCIR:
            self.outputFile.printf(" <-> ", indent=self.isNewLine())
        else:
            assert False, "Illegal State in Unroll Preprocessor Bop2"
        self.visit(node.expr2)

    def visitBop2(self, node: AST.Bop2):
        self.visit(node.expr1)
        if node.op == SeeDotParser.SUB:
            self.outputFile.printf(" - ", indent=self.isNewLine())
        elif node.op == SeeDotParser.ADD:
            self.outputFile.printf(" + ", indent=self.isNewLine())
        else:
            assert False, "Illegal State in Unroll Preprocessor Bop1"
        self.visit(node.expr2)

    def visitFunc(self, node: AST.Func):
        if node.op == SeeDotParser.RELU:
            self.outputFile.printf("relu(", indent=self.isNewLine())
        elif node.op == SeeDotParser.EXP:
            self.outputFile.printf("exp(", indent=self.isNewLine())
        elif node.op == SeeDotParser.ARGMAX:
            self.outputFile.printf("argmax(", indent=self.isNewLine())
        elif node.op == SeeDotParser.SGN:
            self.outputFile.printf("sgn(", indent=self.isNewLine())
        elif node.op == SeeDotParser.TANH:
            self.outputFile.printf("tanh(", indent=self.isNewLine())
        elif node.op == SeeDotParser.SIGMOID:
            self.outputFile.printf("sigmoid(", indent=self.isNewLine())
        else:
            assert False, "Illegal State in Unroll Preprocessor Bop1"
        self.visit(node.expr)
        self.outputFile.printf(") ", indent=self.isNewLine())

    def visitSum(self, node: AST.Sum):
        self.outputFile.printf("$(%s = [%d:%d]) (\n" % (node.name, node.start, node.end), indent=self.isNewLine())
        self.setNewLine()
        self.outputFile.increaseIndent()
        self.visit(node.expr)
        self.outputFile.decreaseIndent()
        self.outputFile.printf("\n)", indent=self.isNewLine())
        self.setNewLine()

    def getNadd(self, baseName:str, curCount, count:int, expr):
        if curCount is None:
            return AST.Let(baseName, self.getNadd(baseName, 0, count, None), expr)
        elif curCount + 1 < count:
            return AST.Bop2(AST.ID(baseName + str(curCount)), SeeDotParser.ADD, self.getNadd(baseName, curCount + 1, count, None))
        else:
            return AST.ID(baseName + str(count - 1))

    def unrollSum(self, resultVar:AST.ID, node:AST.Let):
        unrollNode = node.decl
        toExpr = self.getNadd(node.name, None, unrollNode.unrollFactor, node.expr)
        unrollNode.unrollFactor = min (unrollNode.end - unrollNode.start, unrollNode.unrollFactor)
        for i in reversed(range(unrollNode.unrollFactor)):
            start = unrollNode.start + int(((i) * (unrollNode.end - unrollNode.start))/(unrollNode.unrollFactor))
            end = unrollNode.start + int(((i+1) * (unrollNode.end - unrollNode.start))/(unrollNode.unrollFactor))
            substituter = subs.SubstituteAST(i)
            renamedExpr = substituter.visit(unrollNode.expr)
            subNode = AST.Sum(unrollNode.name, start, end, renamedExpr)
            letNode = AST.Let(node.name + str(i), subNode, toExpr)
            toExpr = letNode
        self.visitLet(toExpr)

    def visitSumUnroll(self, node: AST.SumUnroll):
        self.outputFile.printf("$(%s = [%d:%d]@%d) (\n" % (node.name, node.start, node.end, node.unrollFactor), indent=self.isNewLine())
        self.setNewLine()
        self.outputFile.increaseIndent()
        self.curDepth += 1
        self.visit(node.expr)
        self.curDepth -= 1
        self.outputFile.decreaseIndent()
        self.outputFile.printf("\n)", indent=self.isNewLine())
        self.setNewLine()  

    def visitLoop(self, node: AST.Loop):
        self.outputFile.printf("loop(%s = [%d:%d]," % (node.name, node.start, node.end), indent=self.isNewLine())
        self.visit(node.mutableVar)
        self.outputFile.printf(") (\n", indent=self.isNewLine())
        self.setNewLine()
        self.outputFile.increaseIndent()
        self.visit(node.expr)
        self.outputFile.decreaseIndent()
        self.outputFile.printf("\n)", indent=self.isNewLine())
        self.setNewLine()  

    def unrollLoop(self, resultVar:AST.ID, node:AST.Let):
        unrollNode = node.decl
        toExpr = node.expr
        unrollNode.unrollFactor = min (unrollNode.end - unrollNode.start, unrollNode.unrollFactor)
        for i in reversed(range(unrollNode.unrollFactor)):
            start = unrollNode.start + int(((i) * (unrollNode.end - unrollNode.start))/(unrollNode.unrollFactor))
            end = unrollNode.start + int(((i+1) * (unrollNode.end - unrollNode.start))/(unrollNode.unrollFactor))
            substituter = subs.SubstituteAST(i, unrollNode.mutableVar.name)
            renamedExpr = substituter.visit(unrollNode.expr)
            subNode = AST.Loop(unrollNode.name, start, end, unrollNode.mutableVar, renamedExpr)
            letNode = AST.Let(node.name + (str(i) if i+1 < unrollNode.unrollFactor else ""), subNode, toExpr)
            toExpr = letNode
        self.visitLet(toExpr)

    def visitLoopUnroll(self, node: AST.LoopUnroll):
        self.outputFile.printf("loop(%s = [%d:%d]@%d," % (node.name, node.start, node.end, node.unrollFactor), indent=self.isNewLine())
        self.visit(node.mutableVar)
        self.outputFile.printf(") (\n", indent=self.isNewLine())
        self.setNewLine()
        self.outputFile.increaseIndent()
        self.curDepth += 1
        self.visit(node.expr)
        self.curDepth -= 1
        self.outputFile.decreaseIndent()
        self.outputFile.printf("\n)", indent=self.isNewLine())
        self.setNewLine() 

    def visitCond(self, node: AST.Cond):
        self.visit(node.expr)
        self.outputFile.printf(" ? ", indent=self.isNewLine())
        self.visit(node.trueBlock)
        self.outputFile.printf(" : ", indent=self.isNewLine())
        self.visit(node.falseBlock)

    def visitLet(self, node: AST.Let):
        if isinstance(node.decl, AST.SumUnroll):
            self.unrollSum(node.name, node)
        elif isinstance(node.decl, AST.LoopUnroll):
            self.unrollLoop(node.name, node)
        else:
            self.outputFile.printf("let " + node.name + " = ", indent=self.isNewLine())
            self.visit(node.decl)
            self.outputFile.printf(" in \n", indent=self.isNewLine())
            self.setNewLine()
            self.visit(node.expr)

    def isNewLine(self):
        if self.newline:
            self.newline = False
            return True
        else:
            return False

    def setNewLine(self):
        self.newline = True
