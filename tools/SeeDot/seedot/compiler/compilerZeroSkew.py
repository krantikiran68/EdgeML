# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import antlr4 as antlr
import argparse
import os
import pickle

import seedot

import seedot.compiler.antlr.seedotLexer as seedotLexer
import seedot.compiler.antlr.seedotParser as seedotParser

import seedot.compiler.ast.ast as AST
import seedot.compiler.ast.astBuilder as astBuilder
import seedot.compiler.ast.printAST as printAST

from seedot.compiler.compiler import Compiler
import seedot.compiler.codegen.arduino as arduino
import seedot.compiler.codegen.x86 as x86
import seedot.compiler.codegen.x86ZeroSkew as x86ZeroSkew
import seedot.compiler.codegen.m3 as m3

import seedot.compiler.ir.irBuilder as irBuilder
import seedot.compiler.ir.irBuilderZeroSkew as irBuilderZeroSkew
import seedot.compiler.ir.irUtil as irUtil

import seedot.compiler.TF.ProcessTFGraph as TFMain
import seedot.compiler.ONNX.process_onnx as process_onnx

import seedot.compiler.type as type
import seedot.util as util
import seedot.writer as writer

import seedot.config as config

import numpy as np

'''
The Compiler class reads in the input code, converts it first into an AST, and subsequently into an IR which
contains a sequence of function calls (which are implemented by hand in a library). The IR is fed into the 
desired target codegen, which outputs the C/C++ code which can be run on the target device.
'''


class CompilerZeroSkew(Compiler):

    def __init__(self, algo, encoding, target, inputFile, outputDir, profileLogFile, maxScale, source, outputLogFile, generateAllFiles=True, id=None, printSwitch=-1, substitutions={}, scaleForX=None, variableToBitwidthMap={}, sparseMatrixSizes={}, demotedVarsList=[], demotedVarsOffsets={}, paramInNativeBitwidth=True):
        super().__init__(algo, encoding, target, inputFile, outputDir, profileLogFile, maxScale, source, outputLogFile, generateAllFiles, id, printSwitch, substitutions, scaleForX, variableToBitwidthMap, sparseMatrixSizes, demotedVarsList, demotedVarsOffsets, paramInNativeBitwidth)
        
        self.biasShifts = {}

    # Method takes in input file location, calls the tokenizer, parser upon the file to generate a parse tree
    # and subsequently calls the ASTBuilder to convert it into an AST.
    def genASTFromFile(self, inputFile):
        # Parse and generate CST for the input.
        lexer = seedotLexer.seedotLexer(antlr.FileStream(inputFile))
        tokens = antlr.CommonTokenStream(lexer)
        parser = seedotParser.seedotParser(tokens)
        tree = parser.expr()

        # Generate AST.
        ast = astBuilder.ASTBuilder().visit(tree)
        return ast

    # Takes in the input file location, and depending on the source type (.sd/onnx/TF) calls an 
    # appropriate method to generate an AST.
    def genAST(self, inputFile):
        ext = os.path.splitext(inputFile)[1]

        if self.source == config.Source.seedot:
            return self.genASTFromFile(inputFile)
        elif self.source == config.Source.onnx:
            ast = process_onnx.get_seedot_ast(inputFile)
            return ast
        else:
            ast = TFMain.main()
            return ast

    # Driver code for compiler module which calls other functions.
    def run(self):
        ast = self.genAST(self.input)

        # Perform type inference.
        type.InferType().visit(ast)

        irUtil.init()

        res, state = self.compile(ast)

        if util.forArduino():
            assert False, "Not supported for Zero Skew representation"
            assert self.problemType == config.ProblemType.classification, "Arduino codegen only for Classification problems"
            codegen = arduino.Arduino(self.outputDir, *state)
        elif util.forM3():
            assert False, "Not supported for Zero Skew representation"
            assert self.problemType == config.ProblemType.regression, "M3 codegen only for Regression problems"
            codegen = m3.M3(self.outputDir, *state)
        elif util.forX86():
            codegen = x86ZeroSkew.X86ZeroSkew(self.outputDir, self.generateAllFiles, self.printSwitch, self.id, self.paramInNativeBitwidth, *state)
        else:
            assert False

        codegen.printAll(*res)

    def compile(self, ast):
        return self.genCodeWithFuncCalls(ast)

    def initializeIntermediateScales(self):      
        if util.getEncoding() == config.Encoding.fixed and config.ddsEnabled:
            self.intermediateScales = self.readDataDrivenScales()

        elif util.getEncoding() == config.Encoding.zskew:
            assert config.ddsEnabled, "ZSkew Supported only with data driven scaling"
            self.intermediateScales = self.readDataDrivenScalesAndZeros()

    # Takes in the AST and calls the IRBuilder to generate an IR which is a sequence of function calls.
    def genCodeWithFuncCalls(self, ast):
        outputLog = writer.Writer(self.outputLogFile)

        self.initializeIntermediateScales()

        compiler = irBuilderZeroSkew.IRBuilderZeroSkew(outputLog, self.intermediateScales, self.substitutions, self.scaleForX, self.variableToBitwidthMap, self.sparseMatrixSizes, self.demotedVarsList, self.demotedVarsOffsets)

        res = compiler.visit(ast)

        util.getLogger().debug(compiler.varScales)
        self.biasShifts = compiler.biasShifts
        self.varScales = dict(compiler.varScales)
        self.varZeros = dict(compiler.varZeros)

        outputLog.close()

        # All state variables are used for codegen.
        state = [compiler.varDeclarations, compiler.varDeclarationsLocal, compiler.varScales, compiler.varZeros, compiler.varIntervals, compiler.intConstants, compiler.expTables, compiler.globalVars, compiler.internalVars, compiler.floatConstants, compiler.substitutions, compiler.demotedVarsOffsets, compiler.varsForBitwidth, compiler.varLiveIntervals, compiler.notScratch, compiler.coLocatedVariables]

        for key in compiler.varDeclarations.keys():
            val = compiler.varDeclarations[key]
            if type.isTensor(val):
                dims = val.shape
                self.varSizes[key] = np.prod(dims)
            else:
                self.varSizes[key] = 1

        # Raw live ranges do not capture the scope of the first/last usage of a variable, so they require post-processing.
        state[13] = self.adjustLiveRanges(state[13], compiler.allDepths)

        for i in compiler.globalVars:
            if util.forM3() and i == 'X':
                continue
            state[13].append(i)

        # In floating-point code used for profiling, the set of variables which are profiled using training data are collected.
        if util.getEncoding() == config.Encoding.floatt:
            self.independentVars = list(compiler.independentVars)
            self.independentVars += compiler.globalVars

        self.substitutions = compiler.substitutions

        # Input and output scales are stored, problem type is identified as regression or classification.
        self.scaleForX = compiler.varScales['X']
        self.scaleForY = compiler.varScales[res[1].idf] if res[1].idf in compiler.varScales else 0
        self.problemType = config.ProblemType.classification if res[1].idf not in compiler.varScales else config.ProblemType.regression

        return res, state

    # The floating point code is run on the training dataset and the ranges of all variables are read to compute the scale.
    def readDataDrivenScales(self):
        tempScales = {}
        error = 0.01
        with open('temp/Predictor/dump.profile', 'r') as f:
            for line in f:
                entries = line.strip().split(",")
                var, m, M = entries
                m, M = float(m), float(M)
                tempScales[var] = util.computeScalingFactor(max(abs(m) + error, abs(M) + error))
        return tempScales
    
    # The code to read scales and zeros for ZSkew representation
    def readDataDrivenScalesAndZeros(self):
        tempScales = {}
        error = 0.01
        with open('temp/Predictor/dump.profile', 'r') as f:
            for line in f:

                entries = line.strip().split(",")
                var, m, M = entries
                m, M = float(m), float(M)
                zero = -1*((m + M)/2)
                m = m + zero
                M = M + zero
                maxVar = config.maxVar8Bit if config.wordLength == 8 else config.maxVar16Bit
                scale = M/maxVar
                zero = int(zero/scale)

                tempScales[var] = scale, zero
        return tempScales

    # Post-processing of live ranges to adjust for different scoping level at the first invocation and at the last invocation.
    def adjustLiveRanges(self, oldRanges, depthData):
        newRanges = {}
        for var in oldRanges:
            begIns = oldRanges[var][0]
            endIns = oldRanges[var][1]
            beginningDepth = depthData[begIns]
            endingDepth = depthData[endIns]
            if endingDepth > beginningDepth:
                while depthData[endIns] > beginningDepth:
                    endIns += 1
                endIns -= 1
            elif endingDepth < beginningDepth:
                while depthData[begIns] > endingDepth:
                    begIns -= 1
                begIns += 1
            newRanges[var] = [begIns, endIns]
        return newRanges
