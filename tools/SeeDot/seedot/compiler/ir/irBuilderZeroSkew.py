# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import operator

from seedot.compiler.antlr.seedotParser import seedotParser as SeeDotParser

import seedot.compiler.ast.ast as AST
from seedot.compiler.ast.astVisitor import ASTVisitor

import seedot.compiler.ir.ir as IR
import seedot.compiler.ir.irUtil as IRUtil

import seedot.config as config
import seedot.compiler.type as Type
from seedot.util import *
from seedot.compiler.ir.irBuilder import IRBuilder

'''
IRBuilder class converts the input AST into IR, which is a sequence of function calls.
Each node in the input grammar is handled with its own function in this class.
'''


class IRBuilderZeroSkew(IRBuilder):
    def __init__(self, outputLog, ddsScaleInfo = {}, substitutions = {}, scaleForX = None, variableToBitwidthMap={}, sparseMatrixSizes={}, demotedVarsList=[], demotedVarsOffsets={}):
        super.__init__(outputLog, ddsScaleInfo, substitutions, scaleForX, variableToBitwidthMap, sparseMatrixSizes, demotedVarsList, demotedVarsOffsets)
        