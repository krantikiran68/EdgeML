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
        super().__init__(outputLog, ddsScaleInfo, substitutions, scaleForX, variableToBitwidthMap, sparseMatrixSizes, demotedVarsList, demotedVarsOffsets)
        self.intermediateVarScales = {}
        self.intermediateVarZeros = {}

        self.varZeros = {}

        for varName in ddsScaleInfo.keys():
            self.intermediateVarScales[varName] = ddsScaleInfo[varName][0]
            self.intermediateVarZeros[varName] = ddsScaleInfo[varName][1]
        
    def visitFuncCall(self, node: AST.FuncCall):
        # The type of each argument is same and is equal to the type of the output.
        # The compiler assumes that the output of the uninterpreted function call is the last argument to the function.
        # Also assumes that the scale of the output is equal to the scale of the first argument.
        progs = []
        exprs = []
        for expr in node.exprList:
            (prog_in, expr_in) = self.visit(expr)
            progs.append(prog_in)
            exprs.append(expr_in)

        prog_out = IR.Prog([])
        for prog_funcCall in progs:
            prog_out = IRUtil.concatPrograms(prog_out, prog_funcCall)

        expr_out = self.getTempVar()

        # Scale of the output is the scale of the first argument.
        scale_out = self.varScales[exprs[0].idf]
        intv_out = self.varIntervals[exprs[0].idf]

        args = dict()
        ch = 'A'
        for expr in exprs:
            args[expr] = ch
            ch = chr(ord(ch) + 1) # Inputs would be labelled A, B, C, ... etc.
        args[expr_out] = expr_out.idf

        ch = 'I'
        for i in node.type.shape:
            args[IR.Int(i)] = ch
            ch = chr(ord(ch) + 1) # Indices would be named I, J, K, ... etc.

        comment = IR.Comment(
            node.name + '(' + ', '.join(expr.idf for expr in exprs) + ')', self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        funcCall = IR.FuncCall(node.name, args)

        prog_funcCall = IR.Prog([comment, funcCall])

        self.counter_inst += 1
        self.updateLiveRange([expr_in for expr_in in node.exprList] + [expr_out])

        prog_out = IRUtil.concatPrograms(prog_out, prog_funcCall)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = node.type
        self.varScales[expr_out.idf] = scale_out
        self.varIntervals[expr_out.idf] = intv_out

        return (prog_out, expr_out)

    def visitBopMul2DTensor(self, node: AST.Bop1):
        (prog_in_A, expr_in_A) = self.visit(node.expr1)

        (prog_in_B, expr_in_B) = self.visit(node.expr2)

        expr_temp = self.getTempVar()
        expr_out = self.getTempVar()

        # Read input scales and bit-widths.
        bitwidth_in_A, scale_in_A, zero_in_A = self.getBitwidthScaleZeros(expr_in_A.idf)
        bitwidth_in_B, scale_in_B, zero_in_B = self.getBitwidthScaleZeros(expr_in_B.idf)
        # Read output scales and bitwidths. In data-driven scaling, the output scale is directly profiled from floating-point runtime.
        # In static scaling used by old SeeDot (PLDI '19), output scale and bit-width is set to None is statically computed later.
        
        bitwidth_out, scale_out, zero_out = self.getBitwidthScaleZeros(expr_out.idf)
        bitwidth_temp, scale_temp, zero_temp = self.getBitwidthScaleZeros(expr_out.idf, native=True)
        bitwidth_temp = 32 if config.wordLength == 8 else 128
    
        type_in_A, type_in_B = node.expr1.type, node.expr2.type
        type_out = node.type

        [I, J] = type_in_A.shape
        [J, K] = type_in_B.shape
        type_temp = Type.Int()

        intv_in_A, intv_in_B = self.varIntervals[expr_in_A.idf], self.varIntervals[expr_in_B.idf]

        ## Removed tree sum for the purpose of zero Skew representation

        intv_out = (0,0)

        # shr_A = self.formatShr(shr_A)
        # shr_B = self.formatShr(shr_B)

        M = (scale_in_A * scale_in_B)/scale_out

        # If either of the input parameters are model parameters, change the function name which would read the model parameter differently on the target device (no difference in x86 mode).
        c = ''
        if expr_in_A.idf in self.globalVars:
            c += 'C'
        else:
            c += 'N'
        if expr_in_B.idf in self.globalVars:
            c += 'C'
        else:
            c += 'N'

        expr_in_A.inputVar = False
        expr_in_B.inputVar = False
        expr_out.inputVar = False
        expr_temp.inputVar = False

        comment = IR.Comment(expr_in_A.idf + ' * ' + expr_in_B.idf, self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        # Bit-width for temporary variables.
        bitwidth_mul = bitwidth_temp# self.getTempBitwidth(bitwidth_in_A, bitwidth_in_B, "mul")

        self.varsForBitwidth[expr_temp.idf] = bitwidth_mul

        # If one variable is already used as a sparse matrix, prevent further use as a dense matrix.
        assert expr_in_A.idf + "idx" not in self.sparseMatrixSizes.keys(), "Cannot use same matrix %s for both sparse and dense multiplication" % expr_in_A.idf

        funcCall = IR.FuncCall("MatMul" + c, {
            expr_in_A: "A",
            expr_in_B: "B",
            expr_out: "C",
            expr_temp: "T",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Int(K): "K",
            scale_in_A: "scale_A",
            zero_in_A: "zero-A",
            scale_in_B: "scale_B",
            zero_in_B: "zero_B",
            scale_out: "scale_C",
            zero_out: "zero_C",
            M: "M"
        }) if not self.vbwEnabled else IR.FuncCall("MatMul" + c + ("<int%d_t, int%d_t, int%d_t, int%d_t>"%(bitwidth_in_A, bitwidth_in_B, bitwidth_mul, bitwidth_out)), {
            expr_in_A: "A",
            expr_in_B: "B",
            expr_out: "C",
            expr_temp: "T",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Int(K): "K",
            scale_in_A: "scale_A",
            zero_in_A: "zero-A",
            scale_in_B: "scale_B",
            zero_in_B: "zero_B",
            scale_out: "scale_C",
            zero_out: "zero_C",
            M: "M",
            IR.Int(demote): "demote"
        })

        self.counter_inst += 1
        self.updateLiveRange([expr_in_A, expr_in_B, expr_out, expr_temp])

        
        prog_mul = [comment, funcCall]

        prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_mul)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = type_out
        self.varScales[expr_out.idf] = scale_out
        self.varZeros[expr_out.idf] = zero_out
        self.varIntervals[expr_out.idf] = intv_out

        self.varDeclarations[expr_temp.idf] = type_temp
        self.varScales[expr_temp.idf] = scale_temp
        self.varZeros[expr_temp.idf] = zero_temp
        self.varIntervals[expr_temp.idf] = (0, 0)

        # Print logs.
        self.log.print(comment.msg)
        self.log.print("\tInput1: scale = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in_A.idf],) + self.varIntervals[expr_in_A.idf]))
        self.log.print("\tInput2: scale = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in_B.idf],) + self.varIntervals[expr_in_B.idf]))
        self.log.print("\tOutput: scale = %d, interval = [%d, %d]" % (
            (self.varScales[expr_out.idf],) + self.varIntervals[expr_out.idf]))

        return (prog_out, expr_out)

    # For any variable, get its bitwidth and scale given the bitwidth assignment.
    def getBitwidthScaleZeros(self, varName, native=False):

        assert (self.ddsEnabled),  "Zero skew without DDS not supported"
        
        if self.ddsEnabled or self.vbwEnabled: # If not enabled, all scales statically computed.
            while varName in self.substitutions:
                varName = self.substitutions[varName]

        if varName in self.varScales.keys(): # Function has been called on this variable or scale has been manually computed.
            if varName in self.demotedVarsList:
                return config.wordLength // 2, self.varScales[varName], self.varZeros[varName]
            else:
                return config.wordLength, self.varScales[varName], self.varZeros[varName]
        elif varName in self.intermediateVarScales.keys(): # This will be populated for DDS mode.
            if varName in self.demotedVarsList and native == False:
                getLogger().debug("irBuilderZeroSkew.py: getBitwidthScaleZeros: Unexpected Way of demoting when using vbw with zero skew, possible source of error")
                return config.wordLength // 2, self.adjustScaleAndZero(self.intermediateVarScales[varName], self.intermediateVarZeros[varName], demote=True)
            else:
                return config.wordLength, self.intermediateVarScales[varName], self.intermediateVarZeros[varName]
        else:
            assert False, "No root found"


    def visitLet(self, node: AST.Let):
        # Visit RHS of the let statement.
        (prog_decl, expr_decl) = self.visit(node.decl)

        type_decl = node.decl.type
        idf = node.name

        # e1 : Int
        if Type.isInt(type_decl):
            # LHS is a new integer variable and needs to be assigned to the list of variables.
            self.varDeclarations[idf] = Type.Int()
            self.internalVars.append(idf)

            # Visit remainder of the program.
            (prog_in, expr_in) = self.visit(node.expr)

            cmd = IR.Assn(IR.Var(idf), expr_decl)
            prog_let = IR.Prog([cmd])

            prog_out = IRUtil.concatPrograms(prog_decl, prog_let, prog_in)

            return (prog_out, expr_in)

        # Left Splice case.
        elif node.leftSplice is not None:
            # We have to assign the value of decl (RHS) into a splice of the LHS variable.
            parentVar = node.name
            while parentVar in self.substitutions:
                parentVar = self.substitutions[parentVar] # Done as all metadata is stored at the end of the substitution chain.
            # Assign the RHS to a splice of LHS.
            (prog_splice, expr_splice) = self.visitLeftSplice(node.leftSplice, expr_decl, self.varDeclarations[parentVar])
            (prog_in, expr_in) = self.visit(node.expr)

            # Profile the LHS as the value would have been updated, hence the scale required for LHS in the floating-point code may be different.
            profile = IR.Prog([])
            
            prog_out = IRUtil.concatPrograms(prog_decl, prog_splice, profile, prog_in)

            return (prog_out, expr_in)
        # e1 : Tensor{(),(..)}
        else:
            # Compute the scale of the LHS variable. RHS/decl may have a different bit-width, hence the scale of LHS has to be adjusted accordingly.
            if idf in self.demotedVarsLists:
                self.varScales[idf], self.varZeros[idf] = self.adjustScaleAndZero(self.varScales[expr_decl.idf], self.varZeros[expr_decl.idf], demote=True)
            self.varIntervals[idf] = self.varIntervals[expr_decl.idf]

            # If LHS is demoted to lower bit-width, the RHS should also be in a lower bit-width, so scale of RHS is also adjusted.
            if idf in self.demotedVarsList:
                self.varScales[expr_decl.idf], self.varZeros[expr_decl.idf] = self.adjustScaleAndZero(self.varScales[expr_decl.idf], self.varZeros[expr_decl.idf])
                self.demotedVarsList.append(expr_decl.idf)
                # self.demotedVarsOffsets[expr_decl.idf] = self.demotedVarsOffsets[idf]
                self.varsForBitwidth[expr_decl.idf] = config.wordLength // 2
            else:
                if expr_decl.idf not in self.varsForBitwidth:
                    self.varsForBitwidth[expr_decl.idf] = config.wordLength

            # For input X, scale is computed as follows.
            if idf == "X" and self.scaleForX is not None:
                self.varScales[idf], self.varZeros[idf] = self.adjustScaleAndZero(self.scaleForX[0], self.scaleForX[1]) if 'X' in self.demotedVarsList else (self.scaleForX[0], self.scaleForX[1])
            
            # If the let statement is a model parameter declaration, then the following is invoked.
            if isinstance(node.decl, AST.Decl):
                self.globalVars.append(idf)
                # TODO: Do I need to update varDeclarations or is it handled already?
                self.varDeclarations[idf] = node.decl.type
                expr_decl.idf = idf
                expr_decl.inputVar = True

            # For mutable variables of a loop, such variables are substituted  later and the details are captured here.
            if idf in self.mutableVars:
                expr_decl.idf = idf

            # In fixed-point mode, for mutable variables the scales need to be adjusted which is done here.
            if idf in self.mutableVars:
                # Add a loop to adjust the scale back to the original one.
                curr_scale = self.varScales[idf]
                curr_zero = self.varZeros[idf]
                idfs = idf
                while idfs in self.substitutions.keys():
                    idfs = self.substitutions[idfs]
                # Read profiled scale of the LHS (profile assumes 16-bit variables) and compute final scale depending on actual bitwidth of LHS.
                if self.ddsEnabled:
                    _, raw_new_scale, raw_new_zero = self.getBitwidthScaleZeros(idfs)
                    new_scale, new_zero = self.adjustScaleAndZero(raw_new_scale, raw_new_zero) if idfs in self.demotedVarsList else (raw_new_scale, raw_new_zero)
                    new_intv = (0, 0)
                else:
                    [minVal, maxVal] = self.mutableVarsProfile[0] # TODO: This function may not work for multiple loops in a code.
                    new_scale, new_zero = self.getScaleAndZero(minVal, maxVal, bw=(config.wordLength // 2 if idfs in self.demotedVarsList else config.wordLength))
                    new_intv = (int(minVal / new_scale + new_zero), int(maxVal / new_scale + new_zero))

                # diff_scale = 2 ** (curr_scale - new_scale) if curr_scale > new_scale else 2 ** (new_scale - curr_scale)

                [I, J] = type_decl.shape
                bitwidth_decl, scale_decl, zero_decl = self.getBitwidthScaleZeros(expr_decl.idf)

                # The mutable loop variable needs to have it's scale adjusted so that it remains the same across iterations for correctness.
                adjust = []
                if curr_scale != new_scale:
                    adjust = [IR.FuncCall("AdjustScale", {
                                        IR.Var(idf): "A",
                                        IR.Int(I): "I",
                                        IR.Int(J): "J",
                                        IR.Float(curr_scale): "old_scale",
                                        IR.Int(curr_zero): "old_zero",
                                        IR.Float(new_scale): "new_scale",
                                        IR.Int(new_zero): "new_zero"
                                })] if not self.vbwEnabled else [IR.FuncCall("AdjustScaleShl<int%d_t>"%(bitwidth_decl), {
                                        IR.Var(idf): "A",
                                        IR.Int(I): "I",
                                        IR.Int(J): "J",
                                        IR.Float(curr_scale): "old_scale",
                                        IR.Int(curr_zero): "old_zero",
                                        IR.Float(new_scale): "new_scale",
                                        IR.Int(new_zero): "new_zero"
                                })]
                
                prog_for_mutable = IR.Prog(adjust)

                # Reset the self.scale value to the profile generated one.
                self.varScales[idf] = new_scale
                self.varZeros[idf] = new_zero
                self.varIntervals[idf] = new_intv
            else:
                prog_for_mutable = IR.Prog([])

            (prog_in, expr_in) = self.visit(node.expr)

            getLogger().warning("Removed the if condition in visitLet")
            # # TODO: When is this triggered and why is this required?
            # if idf in self.mutableVars:
            #     getLogger().warning("TODO: Fix this if condition")
            #     idfs = idf
            #     while idfs in self.substitutions.keys():
            #         idfs = self.substitutions[idfs]
            #     if self.ddsEnabled:
            #         _, raw_new_scale = self.getBitwidthAndScale(idfs)
            #         new_scale = raw_new_scale + (config.wordLength // 2 + self.demotedVarsOffsets[idfs] if idfs in self.demotedVarsList else 0)
            #         new_intv = (0, 0)
            #     else:
            #         [minVal, maxVal] = self.mutableVarsProfile[0]
            #         new_scale = self.getScale(max(abs(minVal), abs(maxVal)))
            #         new_intv = self.getInterval(new_scale, minVal, maxVal)
            #     self.varScales[expr_decl.idf] = new_scale
            #     self.varIntervals[expr_decl.idf] = new_intv

            prog_decl = IRUtil.concatPrograms(
                prog_decl, IR.Prog([prog_for_mutable]))

            # Perform substitutions to consolidate generated names and user-provided names.
            prog_in = prog_in.subst(idf, expr_decl)
            expr_in = expr_in.subst(idf, expr_decl)

            # Consolidate the information about live ranges for lhs and rhs, given the substitutions performed above.
            if idf != expr_decl.idf and idf in self.varLiveIntervals and expr_decl.idf in self.varLiveIntervals:
                self.varLiveIntervals[idf] = [min(self.varLiveIntervals[idf][0], self.varLiveIntervals[expr_decl.idf][0]), max(self.varLiveIntervals[idf][1], self.varLiveIntervals[expr_decl.idf][1])]
                self.varLiveIntervals[expr_decl.idf] = list(self.varLiveIntervals[idf])

            prog_out = IRUtil.concatPrograms(prog_decl, prog_in)

            return (prog_out, expr_in)

    def adjustScaleAndZero(self, scale, zero, demote=True):
        if demote == False: # 8-bit to 16-bit
            new_scale = (scale * config.maxVar8Bit) / config.maxVar16Bit 
            new_zero = int(zero*scale/new_scale)
            return new_scale, new_zero
        else: # 16-bit to 8-bit
            new_scale = (scale * config.maxVar16Bit)/config.maxVar8Bit
            new_zero = int(zero*scale/new_scale)
            return new_scale, new_zero
    
    def getScaleAndZero(self, minVal, maxVal, bw=config.wordLength):
        maxVar = config.maxVar8Bit if bw == 8 else config.maxVar16Bit

        zero = -1*((minVal + maxVal) / 2)

        minVal += zero
        maxVal += zero

        scale = maxVal/maxVar
        return scale, int(zero/scale)

