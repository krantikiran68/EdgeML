# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import antlr4 as antlr
import argparse
import os
import pickle

import seedot.compiler.antlr.seedotLexer as seedotLexer
import seedot.compiler.antlr.seedotParser as seedotParser

import seedot.compiler.ast.ast as AST
import seedot.compiler.ast.astBuilder as astBuilder
import seedot.compiler.ast.printAST as printAST

import seedot.compiler.codegen.arduino as arduino
import seedot.compiler.codegen.x86 as x86

import seedot.compiler.ir.irBuilder as irBuilder
import seedot.compiler.ir.irUtil as irUtil

import seedot.compiler.TF.ProcessTFGraph as TFMain

import seedot.compiler.type as type
import seedot.util as util
import seedot.writer as writer


class Compiler:

    def __init__(self, algo, version, target, inputFile, outputDir, profileLogFile, maxScale, outputLogFile):
        if os.path.isfile(inputFile) == False:
            print(inputFile)
            raise Exception("Input file doesn't exist")

        util.setAlgo(algo)
        util.setVersion(version)
        util.setTarget(target)
        self.input = inputFile
        self.outputDir = outputDir
        util.setProfileLogFile(profileLogFile)
        self.outputLogFile = outputLogFile
        util.setMaxScale(maxScale)

    def genASTFromFile(self, inputFile):
        # Parse and generate CST for the input
        lexer = seedotLexer.seedotLexer(antlr.FileStream(inputFile))
        tokens = antlr.CommonTokenStream(lexer)
        parser = seedotParser.seedotParser(tokens)
        tree = parser.expr()

        # Generate AST
        ast = astBuilder.ASTBuilder().visit(tree)
        return ast

    def genAST(self, inputFile):
        ext = os.path.splitext(inputFile)[1]

        if ext == ".sd":
            return self.genASTFromFile(inputFile)
        elif ext == ".pkl":
            ast = TFMain.main()
            # with open(inputFile, 'rb') as file:
            #	ast = pickle.load(file)
            return ast

    def run(self):
        ast = self.genAST(self.input)

        # Pretty printing AST
        # printAST.PrintAST().visit(ast)

        # Perform type inference
        type.InferType().visit(ast)

        irUtil.init()

        res, state = self.compile(ast)

        if util.forArduino():
            codegen = arduino.Arduino(self.outputDir, *state)
        elif util.forX86():
            codegen = x86.X86(self.outputDir, *state)
        else:
            assert False

        codegen.printAll(*res)

    def compile(self, ast):
        return self.genCodeWithFuncCalls(ast)

    def genCodeWithFuncCalls(self, ast):

        outputLog = writer.Writer(self.outputLogFile)

        compiler = irBuilder.IRBuilder(outputLog)
        res = compiler.visit(ast)

        outputLog.close()

        state = compiler.varDeclarations, compiler.varScales, compiler.varIntervals, compiler.intConstants, compiler.expTables, compiler.globalVars, compiler.internalVars, compiler.floatConstants

        self.scaleForX = compiler.varScales['X']

        return res, state
