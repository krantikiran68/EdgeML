# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from antlr4 import *
import argparse
import os
import pickle

from seedot.compiler.antlr.seedotLexer import seedotLexer as SeeDotLexer
from seedot.compiler.antlr.seedotParser import seedotParser as SeeDotParser

import seedot.compiler.ast.ast as AST
import seedot.compiler.ast.astBuilder as ASTBuilder
from seedot.compiler.ast.printAST import PrintAST

from seedot.compiler.codegen.arduino import Arduino as ArduinoCodegen
from seedot.compiler.codegen.x86 import X86 as X86Codegen

from seedot.compiler.ir.irBuilder import IRBuilder
import seedot.compiler.ir.irUtil as IRUtil

from seedot.compiler.TF.ProcessTFGraph import main as TFMain

from seedot.compiler.type import InferType
from seedot.util import *
from seedot.writer import Writer


class Compiler:

    def __init__(self, algo, version, target, inputFile, outputDir, profileLogFile, maxScale, outputLogFile):
        if os.path.isfile(inputFile) == False:
            print(inputFile)
            raise Exception("Input file doesn't exist")

        setAlgo(algo)
        setVersion(version)
        setTarget(target)
        self.input = inputFile
        self.outputDir = outputDir
        setProfileLogFile(profileLogFile)
        self.outputLogFile = outputLogFile
        setMaxScale(maxScale)

    def genASTFromFile(self, inputFile):
        # Parse and generate CST for the input
        lexer = SeeDotLexer(FileStream(inputFile))
        tokens = CommonTokenStream(lexer)
        parser = SeeDotParser(tokens)
        tree = parser.expr()

        # Generate AST
        ast = ASTBuilder.ASTBuilder().visit(tree)
        return ast

    def genAST(self, inputFile):
        ext = os.path.splitext(inputFile)[1]

        if ext == ".sd":
            return self.genASTFromFile(inputFile)
        elif ext == ".pkl":
            ast = TFMain()
            # with open(inputFile, 'rb') as file:
            #	ast = pickle.load(file)
            return ast

    def run(self):
        ast = self.genAST(self.input)

        # Pretty printing AST
        # PrintAST().visit(ast)

        # Perform type inference
        InferType().visit(ast)

        IRUtil.init()

        res, state = self.compile(ast)

        if forArduino():
            codegen = ArduinoCodegen(self.outputDir, *state)
        elif forX86():
            codegen = X86Codegen(self.outputDir, *state)
        else:
            assert False

        codegen.printAll(*res)

    def compile(self, ast):
        return self.genCodeWithFuncCalls(ast)

    def genCodeWithFuncCalls(self, ast):

        outputLog = Writer(self.outputLogFile)

        compiler = IRBuilder(outputLog)
        res = compiler.visit(ast)

        outputLog.close()

        state = compiler.varDeclarations, compiler.varScales, compiler.varIntervals, compiler.intConstants, compiler.expTables, compiler.globalVars, compiler.internalVars, compiler.floatConstants

        self.scaleForX = compiler.varScales['X']

        return res, state
