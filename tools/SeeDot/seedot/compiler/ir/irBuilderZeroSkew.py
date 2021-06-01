# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import operator
import math

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

        self.varScales = {}
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
        zero_out = self.varZeros[exprs[0].idf]
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
        self.varZeros[expr_out.idf] = zero_out
        self.varIntervals[expr_out.idf] = intv_out

        return (prog_out, expr_out)

    def getMatMulShrAndN(self, scale_in_A, scale_in_B, scale_out, zero_in_A, zero_in_B, zero_out, bitiwidth_in_A, bitwidth_in_B, bitwidth_temp, bitwidth_out):
        M = (scale_in_A * scale_in_B)/scale_out
        if  M > 1.0 and math.fabs(M - 1.0) < 0.00001:
            M = 1.0 - 0.0000001
        assert (M < 1.0 and M > 0.0 ), "The multiplier in matmul must be in (0,1)"
        m_scale = self.getScale(M, bitwidth_temp)
        M0 = np.ldexp(M, -m_scale)
        N = -m_scale
        N -= (31 if (bitwidth_temp == 32) else 63)
        return M0, N

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
        clamp_min, clamp_max = self.getClampValues(bitwidth_out)
        intv_out = (0,0)
        demote = 1

        # shr_A = self.formatShr(shr_A)
        # shr_B = self.formatShr(shr_B)
        M0, N = self.getMatMulShrAndN(scale_in_A, scale_in_B, scale_out, zero_in_A, zero_in_B, zero_out, bitwidth_in_A, bitwidth_in_B, bitwidth_temp, bitwidth_out)
        
        # If either of the input parameters are model parameters, change the function name which would read the model parameter differently on the target device (no difference in x86 mode).
        c = ''
        # if expr_in_A.idf in self.globalVars:
        #     c += 'C'
        # else:
        #     c += 'N'
        # if expr_in_B.idf in self.globalVars:
        #     c += 'C'
        # else:
        #     c += 'N'

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
            # expr_temp: "T",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Int(K): "K",
            IR.Int(-1*zero_in_A): "zero_A",
            IR.Int(-zero_in_B): "zero_B",
            IR.Int(zero_out): "zero_C",
            IR.Int(M0): "M0",
            IR.Int(-N): "N",
            IR.Int(clamp_min): "clamp_min",
            IR.Int(clamp_max): "clamp_max"
        }) if not self.vbwEnabled else IR.FuncCall("MatMul" + c + ("<int%d_t, int%d_t, int%d_t, int%d_t>"%(bitwidth_in_A, bitwidth_in_B, bitwidth_mul, bitwidth_out)), {
            expr_in_A: "A",
            expr_in_B: "B",
            expr_out: "C",
            # expr_temp: "T",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Int(K): "K",
            IR.Int(-1*zero_in_A): "zero_A",
            IR.Int(-zero_in_B): "zero_B",
            IR.Int(zero_out): "zero_C",
            IR.Int(M0): "M0",
            IR.Int(-N): "N",
            IR.Int(clamp_min): "clamp_min",
            IR.Int(clamp_max): "clamp_max",
            IR.Int(demote): "demote"
        })

        debugPrint = IR.FuncCall("debugPrint", {
            expr_out: "expr",
            # expr_temp: "T",
            IR.Int(I): "I",
            IR.Int(K): "J",
            IR.Float(scale_out): "scale",
            IR.Int(zero_out): "zero",
            IR.String(expr_out): "varName"
        })

        self.counter_inst += 1
        self.updateLiveRange([expr_in_A, expr_in_B, expr_out, expr_temp])

        
        prog_mul = IR.Prog([comment, funcCall, debugPrint])

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
        self.log.print("\tInput1: scale = %f, zero = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in_A.idf],) + (self.varZeros[expr_in_A.idf],) + self.varIntervals[expr_in_A.idf]))
        self.log.print("\tInput2: scale = %f, zero = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in_B.idf],) + (self.varZeros[expr_in_B.idf],) + self.varIntervals[expr_in_B.idf]))
        self.log.print("\tOutput: scale = %f, zero = %d, interval = [%d, %d]" % (
            (self.varScales[expr_out.idf],) + (self.varZeros[expr_out.idf],) + self.varIntervals[expr_out.idf]))

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
            if idf in self.demotedVarsList:
                self.varScales[idf], self.varZeros[idf] = self.adjustScaleAndZero(self.varScales[expr_decl.idf], self.varZeros[expr_decl.idf], demote=True)
            else:
                self.varScales[idf], self.varZeros[idf] = self.varScales[expr_decl.idf], self.varZeros[expr_decl.idf]
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
            #         _, raw_new_scale = self.getBitwidthScaleZeros(idfs)
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
        
        if maxVal == minVal:
            
            scale = math.fabs(1.0/maxVal) if (maxVal > 1.0) else math.fabs(maxVal)
            zero = 0
            return scale, zero
        zero = -1*((minVal + maxVal) / 2)

        minVal += zero
        maxVal += zero
        
        if math.fabs(maxVal) < 0.00000001:
            scale = 1.0
        else:
            scale = maxVal/maxVar

        return scale, int(zero/scale)

        
    def getInterval(self, scale: int, val_min: float, val_max: float):
        assert False, "This should not be called for zero skew implementation"


    # Floating-point numbers in the input code.
    def visitFloat(self, node: AST.Float):
        val = node.value
        scale, zero = self.getScaleAndZero(val, val)
        intv = (int(val/scale) + zero, int(val/scale) + zero)
        val_int = IR.DataType.getInt(int(val/scale) + zero)

        prog = IR.Prog([])
        expr = self.getTempVar()

        # Updating metadata.
        self.varDeclarations[expr.idf] = node.type
        self.varScales[expr.idf] = scale
        self.varZeros[expr.idf] = zero
        self.varIntervals[expr.idf] = intv
        self.intConstants[expr.idf] = val_int
        self.floatConstants[expr.idf] = val

        return (prog, expr)

    # Declaration for model parameters in the input code.
    def visitDecl(self, node: AST.Decl):
        minVal, maxVal = node.range

        assert minVal <= maxVal, "Range of a variable with values (%.6f, %.6f) is not valid" % (
            minVal, maxVal)

        # The range for model parameters is specified in the input code, which enables us to directly compute their scale.
        scale, zero = self.getScaleAndZero(minVal, maxVal)
        intv = (int(minVal/scale) + zero, int(maxVal/scale) + zero)

        prog = IR.Prog([])
        expr = self.getTempVar()
        expr.inputVar = True

        # Updating metadata.
        self.varScales[expr.idf] = scale
        self.varZeros[expr.idf] = zero
        self.varIntervals[expr.idf] = intv

        return (prog, expr)
    
    def getInterval(self, minVal, maxVal, scale, zero):
        return (int(minVal/scale) + zero, int(maxVal/scale) + zero)
    
    def getNumInFixedPoint(self, num_float, scale):
        assert False, "Illegal Function getNumInFixedPoint for Zero skew representation"
    
    def getNumInZeroSkew(self, num_float, scale, zero):
        return IR.Int(int(num_float/scale) + zero)

    # Init is used for initializing mutable loop variables whose values are updated repeatedly.
    def visitInit(self, node: AST.Init):
        if node.value == 0:
            # getScale() fails for 0. Hence, replacing it with a very low value.
            minVal, maxVal = -0.000001, 0.000001
        else:
            minVal, maxVal = node.value, node.value

        expr = self.getTempVar()

        # Computing the scale of the variable, either using runtime profile data (new SeeDot OOPSLA '20) or using initial value (old SeeDot PLDI '19).
        _, scale, zero = self.getBitwidthScaleZeros(expr.idf)
        
        intv = self.getInterval(scale, zero, minVal, maxVal)

        comment = IR.Comment('init([%s], %.6f)' % (
            ', '.join(map(str, node.shape)), node.value), self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        # If the initial value is zero, memset is more efficient to set all values to zero.
        if node.value == 0:
            memset = IR.Memset(expr, node.type.size())
            prog_init = IR.Prog([comment, memset])
        # Using loops to initialize non-zero values instead of memset.
        else:
            iters_in = self.getTempIterators(len(node.shape))

            loopShape = []  # Contains the shape of the tensor being initialized.
            loopIters = []  # Iterators which will be used to iterate to each tensor element.

            for order in range(len(node.shape)):
                loopShape.append(node.shape[order])
                loopIters.append(iters_in[order])
                loop = IRUtil.loop(loopShape, loopIters, [
                IR.Assn(IRUtil.addIndex(expr, iters_in), 
                self.getNumInZeroSkew(node.value, scale, zero))
            ])

            prog_init = IR.Prog([comment] + loop)

        self.counter_inst += 1
        self.updateLiveRange(expr)

        prog_out = prog_init
        expr_out = expr

        # Updating metadata.
        self.varDeclarations[expr_out.idf] = node.type
        self.varScales[expr_out.idf] = scale
        self.varZeros[expr_out.idf] = zero
        self.varIntervals[expr_out.idf] = intv

        # Logging debug messages.
        self.log.print(comment.msg)
        self.log.print("\tOutput: scale = %d, interval = [%d, %d]" % (
            (self.varScales[expr_out.idf],) + self.varIntervals[expr_out.idf]))

        return (prog_out, expr)

    # out = in ^ T
    def visitTransp(self, node: AST.Transp):
        (prog_in, expr_in) = self.visit(node.expr)

        expr_out = self.getTempVar()

        type_out = node.type
        [I, J] = type_out.shape

        # The input and output scale are same as the values of the input and output tensor are the same.
        bw_out, scale_out, zero_out = self.getBitwidthScaleZeros(expr_in.idf)
        intv_out = self.varIntervals[expr_in.idf]

        expr_in.inputVar = False
        expr_out.inputVar = False

        comment = IR.Comment(expr_in.idf + "^T", self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        # If the input variable is demoted to lower bit-width, demote the output as well as no extra information can be stored in the extra bits.
        if forFixed():
            self.varsForBitwidth[expr_out.idf] = bw_out
            if bw_out != config.wordLength:
                self.demotedVarsList.append(expr_out.idf)

        funcCall = IR.FuncCall("Transpose", {
            expr_in: "A",
            expr_out: "B",
            IR.Int(I): "I",
            IR.Int(J): "J"
        }) if not self.vbwEnabled else IR.FuncCall("Transpose<int%d_t>" % (bw_out), {
            expr_in: "A",
            expr_out: "B",
            IR.Int(I): "I",
            IR.Int(J): "J"
        })

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        prog_transp = IR.Prog([comment, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in, prog_transp)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = type_out
        self.varScales[expr_out.idf] = scale_out
        self.varZeros[expr_out.idf] = zero_out
        self.varIntervals[expr_out.idf] = intv_out

        return (prog_out, expr_out)

    def visitSplice(self, node: AST.Splice):
        (prog_in, expr_in) = self.visit(node.expr)

        vars_in = []
        progs_in = []

        # Each indexing variable can be a complex expression so iterate through them all.
        for var in node.vars:
            part_prog_in, part_expr_in = self.visit(var)
            progs_in.append(part_prog_in)
            vars_in.append(part_expr_in)

        type_in = node.expr.type
        type_out = node.type

        # Keeping input and output scales same because the output tensor is a subtensor of the input, and generally the range of values remain the same in both.
        bw_out, scale_out, zero_out = self.getBitwidthScaleZeros(expr_in.idf)

        expr_out = self.getTempVar()

        # If the input variable is demoted to lower bit-width, demote the output as well as no extra information can be stored in the extra bits.
        self.varsForBitwidth[expr_out.idf] = bw_out
        if self.varsForBitwidth[expr_out.idf] != config.wordLength:
            self.demotedVarsList.append(expr_out.idf)

        # Computing loop iterators for LHS and RHS.
        iters_in = self.getTempIterators(type_in.dim)
        iters_out = self.getTempVars(type_out.dim)

        loopShape = [] # Shape of the output tensor which will dictate the range of the iterators.
        loopIters = [] # Iterator which will iterate across different dimensions of the tensor.
        loopAssns = [] # Assignment carried out within one loop body.
        for order in range(type_in.dim):
            loopShape.append(node.sizes[order])
            loopIters.append(iters_in[order])
            loopAssns.append(IR.Assn(iters_out[order], IRUtil.add(iters_in[order], vars_in[order])))

        expr_out_idx = IRUtil.addIndex(expr_out, iters_in)
        expr_in_idx = IRUtil.addIndex(expr_in, iters_out)
        loop = IRUtil.loop(loopShape, loopIters, loopAssns + [
                IR.Assn(expr_out_idx, expr_in_idx)
            ])

        # Comment in the output code to show the input command for the corresponding output code.
        out_indices = ']['.join([i.idf for i in iters_in])
        in_indices = ']['.join([i.idf for i in iters_out])
        comment = IR.Comment("%s[%s] = %s[%s]"%(expr_out_idx.idf, out_indices, expr_in_idx.idf, in_indices), self.counter_inst+1)

        self.allDepths[self.counter_inst+1] = self.curDepth
        prog_splice = IR.Prog([comment] + loop)

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        # In case the target variable is contiguous, we can optimize (use memcpy instead of a loop).
        canOptimize = True
        loopShapeMustBeOne = False
        for i in range(len(loopShape) - 1, -1, -1):
            if loopShapeMustBeOne:
                if loopShape[i] != 1:
                    canOptimize = False
            else:
                if loopShape[i] == type_in.shape[i]:
                    continue
                elif loopShape[i] < type_in.shape[i]:
                    loopShapeMustBeOne = True
                    continue
                else:
                    assert False, "Illegal State, subtensor dimensions must be less than original tensor dimensions"
        canOptimize = canOptimize and (expr_in.idf not in self.globalVars)

        if canOptimize:
            prog_splice = IR.Prog([comment, IR.Memcpy(expr_out, expr_in, np.prod(loopShape), [IR.Int(0) for i in range(len(vars_in))], vars_in)])
        else:
            assert True

        # Concatenating the code for main expression and the indexing expressions.
        prog_out = IR.Prog([])
        prog_out = IRUtil.concatPrograms(prog_out, prog_in)
        for prog in progs_in:
            prog_out = IRUtil.concatPrograms(prog_out, prog)
        prog_out = IRUtil.concatPrograms(prog_out, prog_splice)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = type_out
        self.varScales[expr_out.idf] = scale_out
        self.varZeros[expr_out.idf] = zero_out
        self.varIntervals[expr_out.idf] = (0,0)

        # Update declarations.
        for var in iters_out:
            self.varDeclarations[var.idf] = Type.Int()
            self.internalVars.append(var.idf)

        return (prog_out, expr_out)

    def visitReshape(self, node: AST.Reshape):
        (prog_in, expr_in) = self.visit(node.expr)

        '''
        reshape(A, (T1, T2), (N, H, W))

        cmd1:  t1 = t2 = 0;
        loop: for n in 0:N:
                 for h in 0:H:
                   for w in 0:W:
        cmd3:        B[t1][t2] = A[n][h][w]
        cmd5:        t2++;
                     if (t2 == T2)
                       t2 = 0;
        cmd5_:         t1++;
        '''

        type_in = node.expr.type
        type_out = node.type

        # Compute scaling factors.
        bw_out, scale_out, zero_out = self.getBitwidthScaleZeros(expr_in.idf)
        intv_out = self.varIntervals[expr_in.idf]

        # Declare variables.
        expr_out = self.getTempVar()

        # If the input variable is demoted to lower bit-width, demote the output as well as no extra information can be stored in the extra bits.
        self.varsForBitwidth[expr_out.idf] = bw_out
        if self.varsForBitwidth[expr_out.idf] != config.wordLength:
            self.demotedVarsList.append(expr_out.idf)

        iters_in = self.getTempIterators(type_in.dim)
        iters_out = self.getTempVars(type_out.dim)

        # Initialize to 0.
        cmd1 = [IR.Assn(var, IRUtil.zero) for var in iters_out]

        # Incrementing the first index.
        first_iter = iters_out[0]
        cmd5_ = IRUtil.incCmd(first_iter)

        # Incrementing other indices using a loop.
        cmd5 = [cmd5_]
        for i in range(1, type_out.dim):
            curr_iter = iters_out[i]
            curr_size = IR.Int(type_out.shape[i])
            cmd5 = [IRUtil.incCmd(curr_iter), IR.If(IRUtil.eq(curr_iter, curr_size), [
                IRUtil.initVarToZero(curr_iter)] + cmd5)]

        # Outer loop.
        # The iterators are selected based on the selection order specified by the user.
        loopShape = []
        loopIters = []
        
        if node.order == None:
            node.order = [i+1 for i in range(type_in.dim)]

        for order in node.order:
            order = order - 1
            loopShape.append(type_in.shape[order])
            loopIters.append(iters_in[order])

        loop = IRUtil.loop(loopShape, loopIters, [IR.Assn(IRUtil.addIndex(
            expr_out, iters_out), IRUtil.addIndex(expr_in, iters_in))] + cmd5)

        # Finalize.
        comment = IR.Comment("reshape(" + expr_in.idf + ", (" + ', '.join(str(e)
            for e in type_out.shape) + "), (" + ', '.join(str(e) for e in node.order) + ")", self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        # In case the reshaped array's memory layout is identical to original array, we can optimize (use memcpy instead of loop).
        canOptimize = True
        for i in range(len(node.order)):
            if node.order[i] != i+1:
                canOptimize = False
        # The input variable 'X' is handled differently in M3 codegen.
        if not (forM3() and expr_in.idf == 'X'):
            canOptimize = canOptimize and expr_in.idf not in self.globalVars

        if canOptimize:
            prog_memcpy = IR.Memcpy(expr_out, expr_in, type_out.size(), [IR.Int(0) for i in range(type_out.dim)], [IR.Int(0) for i in range(type_in.dim)])
            prog_reshape = IR.Prog([comment] + [prog_memcpy])
        else:
            prog_reshape = IR.Prog([comment] + cmd1 + loop)

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        prog_out = IRUtil.concatPrograms(prog_in, prog_reshape)

        # Update context.
        self.varDeclarations[expr_out.idf] = type_out
        self.varScales[expr_out.idf] = scale_out
        self.varZeros[expr_out.idf] = zero_out
        self.varIntervals[expr_out.idf] = intv_out

        # Update declarations.
        for var in iters_out:
            self.varDeclarations[var.idf] = Type.Int()
            self.internalVars.append(var.idf)

        return (prog_out, expr_out)

    # B = reverse(A, axis=...)
    def visitReverse(self, node: AST.Reverse):
        (prog_in, expr_in) = self.visit(node.expr)

        prog_out = IR.Prog([])
        prog_out = IRUtil.concatPrograms(prog_out, prog_in)

        expr_out = self.getTempVar()

        # Scale of the output is the scale of the first argument as the values of output and first argument remain the same.
        intv_out = self.varIntervals[expr_in.idf]
        bitwidth_in, scale_in, zero_in = self.getBitwidthScaleZeros(expr_in.idf)
        bw_out, scale_out, zero_out = self.getBitwidthScaleZeros(expr_in.idf)

        args = dict()
        args[expr_in] = 'A'
        args[IR.Int(node.axis)] = 'axis'

        ch = 'I'
        for i in node.type.shape:
            args[IR.Int(i)] = ch
            ch = chr(ord(ch) + 1) # Indices will be labelled I, J, K, ... etc.

        args[expr_out] = 'B'

        comment = IR.Comment(
            "reverse" + '(' + expr_in.idf + ',' + str(node.axis) + ')', self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        funcCall = IR.FuncCall('Reverse' + str(len(node.type.shape)), args) if not self.vbwEnabled else IR.FuncCall('Reverse' + str(len(node.type.shape)) + '<int' + str(bitwidth_in) + '_t>', args)

        prog_funcCall = IR.Prog([comment, funcCall])

        # If the input variable is demoted to lower bit-width, demote the output as well as no extra information can be stored in the extra bits.
        self.varsForBitwidth[expr_out.idf] = bw_out
        if self.varsForBitwidth[expr_out.idf] != config.wordLength:
            self.demotedVarsList.append(expr_out.idf)

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        prog_out = IRUtil.concatPrograms(prog_out, prog_funcCall)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = node.type
        self.varScales[expr_out.idf] = scale_out
        self.varZeros[expr_out.idf] = zero_out
        self.varIntervals[expr_out.idf] = intv_out

        return (prog_out, expr_out)
    
    def visitUop(self, node: AST.Uop):
        (prog_in, expr_in) = self.visit(node.expr)

        if node.op == SeeDotParser.ADD:
            return (prog_in, expr_in)

        assert node.op == SeeDotParser.SUB

        type_out = node.type
        
        self.allDepths[self.counter_inst+1] = self.curDepth

        # e : Int
        if Type.isInt(type_out):
            prog_out = prog_in
            expr_out = IRUtil.negate(expr_in)

            self.notScratch.append(expr_out.idf)

            # Just to be safe, check that the scaling factor of the integer variable is never tracked
            assert expr_in.idf not in self.varScales and expr_in.idf not in self.varIntervals
        # e: Tensor(), or Tensor(..)
        else:
            expr_out = self.getTempVar()
            iters = self.getTempIterators(type_out.dim)

            if type_out.isShapeOne():
                self.notScratch.append(expr_out.idf)

            bitwidth_out, scale_out, zero_out = self.getBitwidthScaleZeros(expr_in.idf)

            # If the input variable is demoted to lower bit-width, demote the output as well as no extra information can be stored in the extra bits.
            self.varsForBitwidth[expr_out.idf] = bitwidth_out
            if bitwidth_out != config.wordLength:
                self.demotedVarsList.append(expr_out.idf)

            (m, M) = self.varIntervals[expr_in.idf]
            intv_out = (-M, -m)

            lhs = IRUtil.addIndex(expr_out, iters)
            rhs = IRUtil.negate(IRUtil.addIndex(expr_in, iters))
            loop = IRUtil.loop(type_out.shape, iters, [IR.Assn(lhs, rhs)])
            prog_uop = IR.Prog(loop)

            prog_out = IRUtil.concatPrograms(prog_in, prog_uop)

            # Update metadata.
            self.varDeclarations[expr_out.idf] = type_out
            self.varScales[expr_out.idf] = scale_out
            self.varZeros[expr_out.idf] = zero_out
            self.varIntervals[expr_out.idf] = intv_out

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        return (prog_out, expr_out)

    # out = in_A * in_B
    def visitBopMulInt(self, node: AST.Bop1):
        (prog_in_A, expr_in_A) = self.visit(node.expr1)

        (prog_in_B, expr_in_B) = self.visit(node.expr2)

        prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B)
        expr_out = IRUtil.mul(expr_in_A, expr_in_B)

        # Just to be safe, check that the scaling factor of the integer variables is never tracked.
        if isinstance(expr_in_A, IR.Var):
            assert expr_in_A.idf not in self.varScales and expr_in_A.idf not in self.varZeros and expr_in_A.idf not in self.varIntervals 
        if isinstance(expr_in_B, IR.Var):
            assert expr_in_B.idf not in self.varScales and expr_in_B.idf not in self.varZeros and expr_in_B.idf not in self.varIntervals

        return (prog_out, expr_out)
    
    # out = in_A * in_B
    def visitBopMul1DTensor(self, node: AST.Bop1):
        (prog_in_A, expr_in_A) = self.visit(node.expr1)

        (prog_in_B, expr_in_B) = self.visit(node.expr2)

        type_in_A, type_in_B = node.expr1.type, node.expr2.type
        type_out = node.type

        expr_out = self.getTempVar()

        # Read input scales and bit-widths.
        bitwidth_in_A, scale_in_A, zero_in_A = self.getBitwidthScaleZeros(expr_in_A.idf)
        bitwidth_in_B, scale_in_B, zero_in_B = self.getBitwidthScaleZeros(expr_in_B.idf)
        
        # Read output scales and bit-widths. In data-driven scaling, the output scale is directly profiled from floating-point runtime.
        # In static scaling used by old SeeDot (PLDI '19), output scale and bit-width is set to None is statically computed later.
        bitwidth_out, scale_out, zero_out = self.getBitwidthScaleZeros(expr_out.idf)
        bitwidth_temp, scale_temp, zero_temp = self.getBitwidthScaleZeros(expr_out.idf, native=True)
    
        intv_in_A, intv_in_B = self.varIntervals[expr_in_A.idf], self.varIntervals[expr_in_B.idf]

        # Compute scaling hyperparameters given input and output scales. If static scaling of old SeeDot is used, also compute the output scale and bit-width.
        # shr_A, shr_B, H1, H2, demote, scale_out = self.getShrTreeSumAndDemoteParamsForMul(bitwidth_in_A, scale_in_A, bitwidth_in_B, scale_in_B, bitwidth_temp, scale_temp, bitwidth_out, scale_out, 1)

        intv_out = (0, 0) # self.getIntvervalForMul(intv_in_A, shr_A, intv_in_B, shr_B)

        # Ensuring that in the generated code, the scalar is the first argument.
        if type_in_A.dim == 0:
            a, b = expr_in_A, expr_in_B
            bitwidth_in_A, bitwidth_in_B = bitwidth_in_A, bitwidth_in_B
            scale_in_A, scale_in_B = scale_in_A, scale_in_B
            zero_in_A, zero_in_B = zero_in_A, zero_in_B
            [I, J] = type_in_B.shape
        else:
            a, b = expr_in_B, expr_in_A
            bitwidth_in_A, bitwidth_in_B = bitwidth_in_B, bitwidth_in_A
            scale_in_A, scale_in_B = scale_in_B, scale_in_A
            zero_in_B, zero_in_A = zero_in_A, zero_in_B
            [I, J] = type_in_A.shape

        bitwidth_mul = self.getTempBitwidth(bitwidth_in_A, bitwidth_in_B, "mul")
    
        M0, N = self.getMatMulShrAndN(scale_in_A, scale_in_B, scale_out, zero_in_A, zero_in_B, zero_out, bitwidth_in_A, bitwidth_in_B, bitwidth_mul, bitwidth_out)
        a.inputVar = False
        b.inputVar = False
        expr_out.inputVar = False

        clamp_min, clamp_max = self.getClampValues(bitwidth_out)

        comment = IR.Comment(expr_in_A.idf + ' * ' + expr_in_B.idf, self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth
        # Compute bit-width of intermediate variable.
        funcCall = IR.FuncCall("MatMulBroadcastA", {
            a: "A",
            b: "B",
            expr_out: "C",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Int(-1*zero_in_A): "zero_A",
            IR.Int(-zero_in_B): "zero_B",
            IR.Int(zero_out): "zero_C",
            IR.Int(M0): "M0",
            IR.Int(-N): "N",
            IR.Int(clamp_min): "clamp_min",
            IR.Int(clamp_max): "clamp_max"
        }) if not self.vbwEnabled else IR.FuncCall("MatMulBroadcastA<int%d_t, int%d_t, int%d_t, int%d_t>"%(bitwidth_in_A, bitwidth_in_B, bitwidth_mul, bitwidth_out), {
            a: "A",
            b: "B",
            expr_out: "C",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Int(-1*zero_in_A): "zero_A",
            IR.Int(-zero_in_B): "zero_B",
            IR.Int(zero_out): "zero_C",
            IR.Int(M0): "M0",
            IR.Int(-N): "N",
            IR.Int(clamp_min): "clamp_min",
            IR.Int(clamp_max): "clamp_max"
            # IR.Int(demote): "demote"
        })

        self.counter_inst += 1
        self.updateLiveRange([a, b, expr_out])

        debugPrint = IR.FuncCall("debugPrint", {
                expr_out: "expr",
                # expr_temp: "T",
                IR.Int(I): "I",
                IR.Int(J): "J",
                IR.Float(scale_out): "scale",
                IR.Int(zero_out): "zero",
                IR.String(expr_out): "varName"
            })


        prog_mul = IR.Prog([comment, funcCall, debugPrint])

        prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_mul)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = type_out
        self.varScales[expr_out.idf] = scale_out
        self.varZeros[expr_out.idf] = zero_out
        self.varIntervals[expr_out.idf] = intv_out

        # Printing logs.
        self.log.print(comment.msg)
        self.log.print("\tInput1: scale = %f, zero = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in_A.idf],) + (self.varZeros[expr_in_A.idf],) + self.varIntervals[expr_in_A.idf]))
        self.log.print("\tInput2: scale = %f, zero = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in_B.idf],) + (self.varZeros[expr_in_B.idf],) + self.varIntervals[expr_in_B.idf]))
        self.log.print("\tOutput: scale = %f, zero = %d, interval = [%d, %d]" % (
            (self.varScales[expr_out.idf],) + (self.varZeros[expr_out.idf],) + self.varIntervals[expr_out.idf]))

        return (prog_out, expr_out)
    
    def getTempBitwidth(self, bitwidthA, bitwidthB, op, bitwidthC=None):
        if op == "mul":
            # assert bitwidthC is None, "Illegal call to getTempBitwidth()"
            # biggerBitWidth = max(bitwidthA, bitwidthB)
            return (32 if config.wordLength == 8 else 128)
        elif op == "add":
            # assert bitwidthC is not None, "Illegal call to getTempBitwidth()"
            # biggerBitWidth = max(bitwidthA, bitwidthB, bitwidthC)
            return (32 if config.wordLength == 8 else 128)
        else:
            assert False, "Illegal operation specified for temp bitwidth"
    
    # out = in_A <*> in_B
    def visitBopMulCir(self, node: AST.Bop1):
        (prog_in_A, expr_in_A) = self.visit(node.expr1)

        (prog_in_B, expr_in_B) = self.visit(node.expr2)

        type_out = node.type

        expr_out = self.getTempVar()

        assert type_out.dim == 2

        [I, J] = type_out.shape

        # Read input scales.
        bitwidth_in_A, scale_in_A, zero_in_A = self.getBitwidthScaleZeros(expr_in_A.idf)
        bitwidth_in_B, scale_in_B, zero_in_B = self.getBitwidthScaleZeros(expr_in_B.idf)
        # Read output scales and bit-widths. In data-driven scaling, the output scale is directly profiled from floating-point runtime.
        # In static scaling used by old SeeDot (PLDI '19), output scale and bit-width is set to None is statically computed later.
        bitwidth_out, scale_out, zero_out = self.getBitwidthScaleZeros(expr_out.idf)
        bitwidth_temp, scale_temp, zero_temp = self.getBitwidthScaleZeros(expr_out.idf, native=True)

        intv_in_A, intv_in_B = self.varIntervals[expr_in_A.idf], self.varIntervals[expr_in_B.idf]

        # The theoretical output scale in scale_raw might be different than profiled scale scale_out.
        # We perform a scale adjustment in this case for correctness.
        # TODO: Introduce a post-processing pass to merge consecutive scale adjustments hence generated.
        # adjust = []
        # if self.ddsEnabled:
        #     if scale_raw != scale_out:
        #         diff = 2 ** abs(scale_raw - scale_out)
        #         if scale_raw > scale_out:
        #             adjust = [IR.FuncCall("AdjustScaleShl" + (("<int%d_t>"%bitwidth_out) if self.vbwEnabled else ""), {
        #                         expr_out: "A",
        #                         IR.Int(I): "I",
        #                         IR.Int(J): "J",
        #                         IR.Int(diff): "scale"
        #                     })]
        #         else:
        #             adjust = [IR.FuncCall("AdjustScaleShr" + (("<int%d_t>"%bitwidth_out) if self.vbwEnabled else ""), {
        #                         expr_out: "A",
        #                         IR.Int(I): "I",
        #                         IR.Int(J): "J",
        #                         IR.Int(diff): "scale"
        #                     })]
        # else:
        #     scale_out = scale_raw

        intv_out = (0,0) # self.getIntvervalForMul(intv_in_A, shr_A, intv_in_B, shr_B)

        expr_in_A.inputVar = False
        expr_in_B.inputVar = False
        expr_out.inputVar = False

        comment = IR.Comment(expr_in_A.idf + ' <*> ' + expr_in_B.idf, self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        bitwidth_mul = self.getTempBitwidth(bitwidth_in_A, bitwidth_in_B, "mul")

        M0, N = self.getMatMulShrAndN(scale_in_A, scale_in_B, scale_out, zero_in_A, zero_in_B, zero_out, bitwidth_in_A, bitwidth_in_B, bitwidth_mul, bitwidth_out)
        clamp_min, clamp_max = self.getClampValues(bitwidth_out)
        funcCall = IR.FuncCall("Hadamard", {
            expr_in_A: "A",
            expr_in_B: "B",
            expr_out: "C",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Int(-1*zero_in_A): "zero_A",
            IR.Int(-zero_in_B): "zero_B",
            IR.Int(zero_out): "zero_C",
            IR.Int(M0): "M0",
            IR.Int(-N): "N",
            IR.Int(clamp_min): "clamp_min",
            IR.Int(clamp_max): "clamp_max"
        }) if not self.vbwEnabled else IR.FuncCall("Hadamard<int%d_t, int%d_t, int%d_t, int%d_t>"%(bitwidth_in_A, bitwidth_in_B, bitwidth_mul, bitwidth_out), {
            expr_in_A: "A",
            expr_in_B: "B",
            expr_out: "C",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Int(-1*zero_in_A): "zero_A",
            IR.Int(-zero_in_B): "zero_B",
            IR.Int(zero_out): "zero_C",
            IR.Int(M0): "M0",
            IR.Int(-N): "N",
            IR.Int(clamp_min): "clamp_min",
            IR.Int(clamp_max): "clamp_max"
        })

        debugPrint = IR.FuncCall("debugPrint", {
                expr_out: "expr",
                # expr_temp: "T",
                IR.Int(I): "I",
                IR.Int(J): "J",
                IR.Float(scale_out): "scale",
                IR.Int(zero_out): "zero",
                IR.String(expr_out): "varName"
            })

        self.counter_inst += 1
        self.updateLiveRange([expr_in_A, expr_in_B, expr_out])

        prog_mul = IR.Prog([comment, funcCall, debugPrint])

        prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_mul)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = type_out
        self.varScales[expr_out.idf] = scale_out
        self.varZeros[expr_out.idf] = zero_out
        self.varIntervals[expr_out.idf] = intv_out

        # Print logs.
        self.log.print(comment.msg)
        self.log.print("\tInput1: scale = %f, zero = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in_A.idf],) + (self.varZeros[expr_in_A.idf],) + self.varIntervals[expr_in_A.idf]))
        self.log.print("\tInput2: scale = %f, zero = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in_B.idf],) + (self.varZeros[expr_in_B.idf],) + self.varIntervals[expr_in_B.idf]))
        self.log.print("\tOutput: scale = %f, zero = %d, interval = [%d, %d]" % (
            (self.varScales[expr_out.idf],) + (self.varZeros[expr_out.idf],) + self.varIntervals[expr_out.idf]))


        return (prog_out, expr_out)
    
    # out = in_A 'op' in_B
    def visitBop2(self, node: AST.Bop2):
        (prog_in_A, expr_in_A) = self.visit(node.expr1)

        (prog_in_B, expr_in_B) = self.visit(node.expr2)

        type_out = node.type

        if node.op == SeeDotParser.ADD:
            (op_ir, op_fn) = (IR.Op.Op['+'], operator.add)
            funcName = "MatAdd"
        elif node.op == SeeDotParser.SUB:
            (op_ir, op_fn) = (IR.Op.Op['-'], operator.sub)
            funcName = "MatSub"

        # e : Int
        if Type.isInt(type_out):
            prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B)
            expr_out = IR.IntBop(expr_in_A, op_ir, expr_in_B)

            # Just to be safe that the scaling factor of the integer variable is never tracked.
            if isinstance(expr_in_A, IR.Var):
                assert expr_in_A.idf not in self.varScales and expr_in_A.idf not in self.varZeros and expr_in_A.idf not in self.varIntervals
            if isinstance(expr_in_B, IR.Var):
                assert expr_in_B.idf not in self.varScales and expr_in_B.idf not in self.varZeros and expr_in_B.idf not in self.varIntervals
        # e : Tensor(), or Tensor(..)
        else:
            assert type_out.dim == 2 or (type_out.dim == 4 and config.vbwEnabled), "Addition/subtraction of tensors is currently only supported for 2D tensors. Addition for 4D tensors is supported when VBW is enabled"

            type_A = node.expr1.type
            type_B = node.expr2.type

            assert (not type_out.dim == 4) or (type_A.dim == type_B.dim and expr_in_A.idf not in self.globalVars and expr_in_B.idf not in self.globalVars and node.op == SeeDotParser.ADD), "For 4D operation, no broadcasting supported, inputs should not be model parameters, and operation cannot be subtraction"

            # Depending on whether one of the inputs is a model parameter, change the function name so that the model parameter is read differently in the arduino codegen. No difference in case of x86 code.
            c = ''
            # if op_fn == operator.add:
            #     if expr_in_A.idf in self.globalVars:
            #         c += 'C'
            #     else:
            #         c += 'N'
            #     if expr_in_B.idf in self.globalVars:
            #         c += 'C'
            #     else:
            #         c += 'N'

            # If one of the inputs is a scalar, the operator will broadcast that input.
            if type_A.dim == 0:
                funcName += 'BroadCastA'
                c = ''
            elif type_B.dim == 0:
                funcName += 'BroadCastB'
                c = ''

            expr_out = self.getTempVar()

            # Read input scale.
            bitwidth_in_A, scale_in_A, zero_in_A = self.getBitwidthScaleZeros(expr_in_A.idf)
            bitwidth_in_B, scale_in_B, zero_in_B = self.getBitwidthScaleZeros(expr_in_B.idf)
            # Read output scale.

            bitwidth_out, scale_out, zero_out = self.getBitwidthScaleZeros(expr_out.idf)
            
            # Compute scaling hyperparameters given input and output scales. If static scaling of old SeeDot is used, also compute the output scale and bit-width.
            bitwidth_temp = self.getTempBitwidth(bitwidth_in_A, bitwidth_in_B, "add", bitwidth_out)
            
            (left_shift, shrA, nA, shrB, nB, shrC, nC) = self.getScaleAndZeroForAddAndSub(scale_in_A, zero_in_A, scale_in_B, zero_in_B, scale_out, zero_out, bitwidth_in_A, bitwidth_in_B, bitwidth_temp, bitwidth_out, op_fn)
            
            intv_in_A, intv_in_B = self.varIntervals[expr_in_A.idf], self.varIntervals[expr_in_B.idf]
            clamp_min, clamp_max = self.getClampValues(bitwidth_out)
            
            # demoteLog = shr_out - 8 if shr_out >= 8 else 0
            # shr_out = min(shr_out, 8)
            # irdemote = self.formatShr(demoteLog)

            if type_out.dim == 2:
                [I, J] = type_out.shape
            elif type_out.dim == 4:
                [N, H, W, C] = type_out.shape
            else:
                assert False, "Unsupported dimension for addition"

            # shr_A = self.formatShr(shr_A)
            # shr_B = self.formatShr(shr_B)
            # shr_out = self.formatShr(shr_out)

            expr_in_A.inputVar = False
            expr_in_B.inputVar = False
            expr_out.inputVar = False

            comment = IR.Comment(expr_in_A.idf + ' ' +
                                 op_ir.name + ' ' + expr_in_B.idf, self.counter_inst+1)
            self.allDepths[self.counter_inst+1] = self.curDepth

            # Generate output function call depending on dimensionality of the input / output.
            if type_out.dim == 2:
                funcCall = IR.FuncCall(funcName + c, {
                    expr_in_A: "A",
                    expr_in_B: "B",
                    expr_out: "C",
                    IR.Int(I): "I",
                    IR.Int(J): "J",
                    IR.Int(left_shift): "left_shift",
                    IR.Int(-zero_in_A): "zero_in_A",
                    IR.Int(shrA): "shrA",
                    IR.Int(-nA): "nA",
                    IR.Int(-zero_in_B): "zero_in_B",
                    IR.Int(shrB): "shrB",
                    IR.Int(-nB): "nB",
                    IR.Int(zero_out): "zero_out",
                    IR.Int(shrC): "shrC",
                    IR.Int(-nC): "nC",
                    IR.Int(clamp_min): "clamp_min",
                    IR.Int(clamp_max): "clamp_max"
                }) if not self.vbwEnabled else IR.FuncCall(funcName + c + ("<int%d_t, int%d_t, int%d_t, int%d_t>" % (bitwidth_in_A, bitwidth_in_B, bitwidth_temp, bitwidth_out)), {
                    expr_in_A: "A",
                    expr_in_B: "B",
                    expr_out: "C",
                    IR.Int(I): "I",
                    IR.Int(J): "J",
                    IR.Int(left_shift): "left_shift",
                    IR.Int(-zero_in_A): "zero_in_A",
                    IR.Int(shrA): "shrA",
                    IR.Int(-nA): "nA",
                    IR.Int(-zero_in_B): "zero_in_B",
                    IR.Int(shrB): "shrB",
                    IR.Int(-nB): "nB",
                    IR.Int(zero_out): "zero_out",
                    IR.Int(shrC): "shrC",
                    IR.Int(-nC): "nC",
                    IR.Int(clamp_min): "clamp_min",
                    IR.Int(clamp_max): "clamp_max"
                    # irdemote: "demote"
                })
            elif type_out.dim == 4:
                funcCall = IR.FuncCall(funcName + "4", {
                    expr_in_A: "A",
                    expr_in_B: "B",
                    expr_out: "X",
                    IR.Int(N): "N",
                    IR.Int(H): "H",
                    IR.Int(W): "W",
                    IR.Int(C): "C",
                    IR.Int(left_shift): "left_shift",
                    IR.Int(-zero_in_A): "zero_in_A",
                    IR.Int(shrA): "shrA",
                    IR.Int(-nA): "nA",
                    IR.Int(-zero_in_B): "zero_in_B",
                    IR.Int(shrB): "shrB",
                    IR.Int(-nB): "nB",
                    IR.Int(zero_out): "zero_out",
                    IR.Int(shrC): "shrC",
                    IR.Int(-nC): "nC",
                    IR.Int(clamp_min): "clamp_min",
                    IR.Int(clamp_max): "clamp_max"
                }) if not self.vbwEnabled else IR.FuncCall(funcName + "4" + ("<int%d_t, int%d_t, int%d_t, int%d_t>" % (bitwidth_in_A, bitwidth_in_B, self.getTempBitwidth(bitwidth_in_A, bitwidth_in_B, "add", bitwidth_out), bitwidth_out)), {
                    expr_in_A: "A",
                    expr_in_B: "B",
                    expr_out: "X",
                    IR.Int(N): "N",
                    IR.Int(H): "H",
                    IR.Int(W): "W",
                    IR.Int(C): "C",
                    IR.Int(left_shift): "left_shift",
                    IR.Int(-zero_in_A): "zero_in_A",
                    IR.Int(shrA): "shrA",
                    IR.Int(-nA): "nA",
                    IR.Int(-zero_in_B): "zero_in_B",
                    IR.Int(shrB): "shrB",
                    IR.Int(-nB): "nB",
                    IR.Int(zero_out): "zero_out",
                    IR.Int(shrC): "shrC",
                    IR.Int(-nC): "nC",
                    IR.Int(clamp_min): "clamp_min",
                    IR.Int(clamp_max): "clamp_max"
                    # irdemote: "demote"
                })

            debugPrint = IR.FuncCall("debugPrint", {
                expr_out: "expr",
                # expr_temp: "T",
                IR.Int(I): "I",
                IR.Int(J): "J",
                IR.Float(scale_out): "scale",
                IR.Int(zero_out): "zero",
                IR.String(expr_out): "varName"
            })


            self.counter_inst += 1
            self.updateLiveRange([expr_in_A, expr_in_B, expr_out])

            if type_out.dim == 4:
                if expr_in_A.idf not in self.globalVars and bitwidth_in_A == bitwidth_out:
                    self.setMemorySharableVariables(expr_in_A, expr_out)
                elif expr_in_B.idf not in self.globalVars and bitwidth_in_B == bitwidth_out:
                    self.setMemorySharableVariables(expr_in_B, expr_out)



            # The theoretical output scale in scale_raw might be different than profiled scale scale_out.
            # We perform a scale adjustment in this case for correctness.
            # TODO: Introduce a post-processing pass to merge consecutive scale adjustments hence generated.
            # if type_out.dim == 2:
            #     adjust = []
            #     if forFixed():
            #         if scale_out_unadjusted != scale_out:
            #             if scale_out_unadjusted > scale_out:
            #                 diff_scale = 2 ** (scale_out_unadjusted - scale_out)
            #                 adjust = [IR.FuncCall("AdjustScaleShl" + (("<int%d_t>"%bitwidth_out) if self.vbwEnabled else ""), {
            #                                     expr_out: "A",
            #                                     IR.Int(I): "I",
            #                                     IR.Int(J): "J",
            #                                     IR.Int(diff_scale): "scale"
            #                                     })]
            #             elif scale_out_unadjusted < scale_out:
            #                 diff_scale = 2 ** (scale_out - scale_out_unadjusted)
            #                 adjust = [IR.FuncCall("AdjustScaleShr" + (("<int%d_t>"%bitwidth_out) if self.vbwEnabled else ""), {
            #                                     expr_out: "A",
            #                                     IR.Int(I): "I",
            #                                     IR.Int(J): "J",
            #                                     IR.Int(diff_scale): "scale"
            #                                     })]
            # elif type_out.dim == 4:
            #     adjust = []
            #     if forFixed():
            #         if scale_out_unadjusted != scale_out:
            #             if scale_out_unadjusted > scale_out:
            #                 diff_scale = 2 ** (scale_out_unadjusted - scale_out)
            #                 adjust = [IR.FuncCall("AdjustScaleShl" + (("<int%d_t>"%bitwidth_out) if self.vbwEnabled else ""), {
            #                                     expr_out: "A",
            #                                     IR.Int(N): "N",
            #                                     IR.Int(H): "H",
            #                                     IR.Int(W): "W",
            #                                     IR.Int(C): "C",
            #                                     IR.Int(diff_scale): "scale"
            #                                     })]
            #             elif scale_out_unadjusted < scale_out:
            #                 diff_scale = 2 ** (scale_out - scale_out_unadjusted)
            #                 adjust = [IR.FuncCall("AdjustScaleShr" + (("<int%d_t>"%bitwidth_out) if self.vbwEnabled else ""), {
            #                                     expr_out: "A",
            #                                     IR.Int(N): "N",
            #                                     IR.Int(H): "H",
            #                                     IR.Int(W): "W",
            #                                     IR.Int(C): "C",
            #                                     IR.Int(diff_scale): "scale"
            #                                     })]
            # else:
            #     assert False, "Illegal number of dimensions"

            prog_bop = IR.Prog( [comment, funcCall, debugPrint])

            prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_bop)

            # Updating metadata.
            self.varDeclarations[expr_out.idf] = type_out
            self.varScales[expr_out.idf] = scale_out
            self.varZeros[expr_out.idf] = zero_out
            self.varIntervals[expr_out.idf] = (0,0)

            # Print log.
            self.log.print(comment.msg)
            self.log.print("\tInput1: scale = %f, zero = %d, interval = [%d, %d]" % (
                (self.varScales[expr_in_A.idf],) + (self.varZeros[expr_in_A.idf],) + self.varIntervals[expr_in_A.idf]))
            self.log.print("\tInput2: scale = %f, zero = %d, interval = [%d, %d]" % (
                (self.varScales[expr_in_B.idf],) + (self.varZeros[expr_in_B.idf],) + self.varIntervals[expr_in_B.idf]))
            self.log.print("\tOutput: scale = %f, zero = %d, interval = [%d, %d]" % (
                (self.varScales[expr_out.idf],) + (self.varZeros[expr_out.idf],) + self.varIntervals[expr_out.idf]))


        return (prog_out, expr_out)
    
    def getScaleAndZeroForAddAndSub(self, scale_in_A, zero_in_A, scale_in_B, zero_in_B, scale_out, zero_out, bitwidth_in_A, bitwidth_in_B, bitwidth_temp, bitwidth_out, op_fn):
        if op_fn == operator.add or op_fn == operator.sub:
            if not config.vbwEnabled:   
                # q3  = (s1/s3)*(q1-z1) + (s2/s3)*(q2-z2)

                left_shift = 0 # int((bitwidth_temp - bitwidth_in_A) - 1)
                # Make the input quantized to 32-bits. 

                s1_s3 = scale_in_A
                s2_s3 = scale_in_B
                s3_s3 = 1.0/scale_out
                m1, n1 = self.getQuantizedMultiplierLTO(s1_s3, bitwidth_temp, bitwidth_in_A)
                m2, n2 = self.getQuantizedMultiplierLTO(s2_s3, bitwidth_temp, bitwidth_in_B)
                m3, n3 = self.getQuantizedMultiplierLTO(s3_s3, bitwidth_temp, bitwidth_out)
                n1 -= (31 if (bitwidth_temp == 32) else 63) + left_shift
                n2 -= (31 if (bitwidth_temp == 32) else 63) + left_shift
                n3 -= (31 if (bitwidth_temp == 32) else 63) + left_shift

                return (left_shift, m1, n1, m2, n2, m3, n3)
            else:
                assert False, "No implementation yet for vbw"
        else:
            assert False, "Op_fn can be add or sub only"
    
    def getScale(self, val: float, bw):
        l = np.log2(val)
        if int(l) == l:
            c = l + 1
        else:
            c = np.ceil(l)
        return -int((bw - 1) - c)

    def getQuantizedMultiplierLTO(self, m, bitwidth_temp, bitwidth_in_A):
        # assert m <=1.0
        scale = self.getScale(m, bitwidth_temp)

        m0 = np.ldexp(m, -scale)
        return m0, -scale
    
    def visitMathExp(self, node: AST.Func):
        # Used in the old SeeDot (PLDI '19) version.
        # Tunable parameter.
        MIN = 0.1

        (prog_in, expr_in) = self.visit(node.expr)

        type_in = node.expr.type

        # Reading input scale and bit-width.
        bitwidth_in, scale_in, zero_in = self.getBitwidthScaleZeros(expr_in.idf)

        '''
        1.  y = ((int) (exp(((float)e) / shr1) * shr2))
        '''

        maxExp = np.exp(MIN)

        expr_out = self.getTempVar()

        # Reading / Computing output bit-width.
        if self.ddsEnabled:
            bitwidth_out, _, _ = self.getBitwidthScaleZeros(expr_out.idf)
        else:
            bitwidth_out = config.wordLength // 2 if expr_out.idf in self.demotedVarsList else config.wordLength
        # Computing output scale.
        scale_out, zero_out = self.getScaleAndZero(maxExp, maxExp, bw=(config.wordLength // 2 if expr_out.idf in self.demotedVarsList else config.wordLength)) 

        intv_out = self.getInterval(maxExp, maxExp, scale_out, zero_out)

        [I, J] = type_in.shape

        
        cmd0 = IR.Comment('exp(' + expr_in.idf + ')', self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        funcCall = IR.FuncCall("Exp", {
            expr_in: "A",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Float(scale_in): "scale_in",
            IR.Int(zero_in): "zero_in",
            IR.Float(scale_out): "scale_out",
            IR.Int(zero_out): "zero_out",
            expr_out: "B"
        }) if not self.vbwEnabled else IR.FuncCall("Exp<int%d_t, int%d_t>"%(bitwidth_in, bitwidth_out), {
            expr_in: "A",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Float(scale_in): "scale_in",
            IR.Int(zero_in): "zero_in",
            IR.Float(scale_out): "scale_out",
            IR.Int(zero_out): "zero_out",
            expr_out: "B",
            IR.Int(1): "demote"
        })

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        # This method is used in the profiling floating point stage to check whether the input values are beyond a threshold.
        # Input values beyond a threshold are always mapped to zero in fixed-point code, hence these datapoints hold little use in the fixed-point mode.
        # rangeCheck = IR.FuncCall("checkRange2", {
        #     expr_in: "A",
        #     IR.Int(I): "I",
        #     IR.Int(J): "J"
        # })  if self.functionReducedProfiling and forFloat() else IR.Comment("Recommend switching on Function Reduced Profiling for sound output")

        # profile = IR.FuncCall("Profile2", {
        #     expr_out: "Var",
        #     IR.Int(I): "I",
        #     IR.Int(J): "J",
        #     IR.String(expr_out): "VarName"
        # })
        # if forFloat():
        #     self.independentVars.append(expr_out.idf)

        prog_exp = IR.Prog([cmd0, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in, prog_exp)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = type_in
        self.varScales[expr_out.idf] = scale_out
        self.varZeros[expr_out.idf] = zero_out
        self.varIntervals[expr_out.idf] = intv_out

        return (prog_out, expr_out)

    def getAlphaCount(self, max, shl):
        assert False, "No impelemntation of AlphaCount for ZeroSkew"
    
    def visitArgMax(self, node: AST.Func):
        (prog_in, expr_in) = self.visit(node.expr)

        type_out = node.expr.type

        assert type_out.dim == 2, "'argmax' operator currently only supports 2D tensors."

        # Read input scale.
        bitwidth_in, scale_in, zero_in = self.getBitwidthScaleZeros(expr_in.idf)

        [I, J] = type_out.shape

        expr_out = self.getTempVar()

        expr_in.inputVar = False

        comment = IR.Comment('argmax(' + expr_in.idf + ')', self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        funcCall = IR.FuncCall("ArgMax", {
            expr_in: "A",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Float(scale_in): "scale_in",
            IR.Int(-zero_in): "zero_in",
            expr_out: "index"
        }) if not self.vbwEnabled else IR.FuncCall("ArgMax<int%d_t>"%(bitwidth_in), {
            expr_in: "A",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Float(scale_in): "scale_in",
            IR.Int(-zero_in): "zero_in",
            expr_out: "index"
        })

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        prog_argmax = IR.Prog([comment, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in, prog_argmax)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = Type.Int()
        self.internalVars.append(expr_out.idf)

        return (prog_out, expr_out)
    
    def visitTanh(self, node: AST.Func):
        # Old implementation of TanH, where a linear approximation is used.
        # The floating-point version of this method uses math.h implementation of exp(x).
        (prog_in, expr_in) = self.visit(node.expr)

        type_in = node.expr.type
        [I, J] = type_in.shape

        expr_out = self.getTempVar()

        # Read input scale.
        bitwidth_in, scale_in, zero_in = self.getBitwidthScaleZeros(expr_in.idf)
        intv_in = self.varIntervals[expr_in.idf]

        # If input is demoted to lower bit-width, demote the output to lower bit-width as well.
        if expr_in.idf in self.demotedVarsList:
            self.demotedVarsList.append(expr_out.idf)
            self.varsForBitwidth[expr_out.idf] = config.wordLength // 2

        tanh_limit = self.getNumInZeroSkew(config.tanhLimit, scale_in, zero_in)

        tanh_intv = self.getInterval(
            config.tanhLimit, config.tanhLimit, scale_in, zero_in)
        intv_out = self.updateTanhIntv(intv_in, tanh_intv)

        scale_new, zero_new = self.getScaleAndZero(config.tanhLimit, config.tanhLimit)
        getLogger().debug("Scale changes in TanH operation: old = %f, new = %f, diff = %f" % (
            scale_in, scale_new, abs(scale_in - scale_new)))

        expr_in.inputVar = False

        comment = IR.Comment("tanh(" + expr_in.idf + ")", self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        scale_out, zero_out = self.getScaleAndZero(-config.tanhLimit, config.tanhLimit, bw = config.wordLength//2 if expr_out.idf in self.demotedVarsList else config.wordLength)
        # scale_out = scale_in # self.getScale(1.5)
        # tanh_limit_out = 2 ** -scale_out
        M1, N1, M2, N2, clamp_radius = self.getTanHShrAndN(scale_in, zero_in, scale_out, zero_out, intv_in, bitwidth_in)

        funcCall = IR.FuncCall("TanH", {
            expr_in: "A",
            expr_out: "B",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Int(-zero_in): "zero_in",
            IR.Int(M1): "M1",
            IR.Int(-N1): "N1",
            IR.Int(zero_out): "zero_out",
            IR.Int(M2): "M2",
            IR.Int(-N2): "N2",
            IR.Int(clamp_radius): "clamp_radius"
        }) if not self.vbwEnabled else IR.FuncCall("TanH<int%d_t>"%(bitwidth_in), {
            expr_in: "A",
            expr_out: "B",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Int(-zero_in): "zero_in",
            IR.Int(M1): "M1",
            IR.Int(-N1): "N1",
            IR.Int(zero_out): "zero_out",
            IR.Int(M2): "M2",
            IR.Int(-N2): "N2",
            IR.Int(clamp_radius): "clamp_radius"
        })

        debugPrint = IR.FuncCall("debugPrint", {
                expr_out: "expr",
                # expr_temp: "T",
                IR.Int(I): "I",
                IR.Int(J): "J",
                IR.Float(scale_out): "scale",
                IR.Int(zero_out): "zero",
                IR.String(expr_out): "varName"
            })

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        prog_tanh = IR.Prog([comment, funcCall, debugPrint])

        prog_out = IRUtil.concatPrograms(prog_in, prog_tanh)

        # Updating metadata.
        self.varDeclarations[expr_out.idf] = type_in
        self.varIntervals[expr_out.idf] = intv_out
        self.varScales[expr_out.idf] = scale_out
        self.varZeros[expr_out.idf] = zero_out

        return (prog_out, expr_out)
    
    def getTanHShrAndN(self, scale_in, zero_in, scale_out, zero_out, intv_in, bitwidth_in, bitwidth_temp = None):
        
        assert config.wordLength == 8, "TanH not implemented for anything other than 8-bits"
        if bitwidth_temp == None:
            bitwidth_temp = 32 if (config.wordLength == 8) else 64
        
        M1, N1 = self.getQuantizedMultiplierLTO(scale_in, bitwidth_temp, bitwidth_in)
        N1 -= ((31 if (bitwidth_temp == 32) else 63) + (27 if (bitwidth_temp == 32) else 55))
        M2, N2 = self.getQuantizedMultiplierLTO(1.0/scale_out, bitwidth_temp, bitwidth_in)
        N2 -= 24 if (bitwidth_temp == 32) else 48

        getLogger().debug("TanH fixed point scale in Zero Skew: " + str(scale_out))

        clamp_min, clamp_max = self.getClampValues(bitwidth_in)
        return M1, N1, M2, N2, min(abs(clamp_max), abs(clamp_min))
        
    def getSigmoidShrAndN(self, scale_in, zero_in, scale_out, zero_out, intv_in, bitwidth_in, bitwidth_temp = None):
        # if intv_in == (0,0):
        #     intv_in = (-config.maxVar8Bit, config.maxVar8Bit)
        #     assert (not config.vbwEnabled) and (config.wordLength == 8) 
        # sigmoid_min, sigmoid_max = intv_in
        # getLogger().debug("TanH internval in Zero Skew: " + str(sigmoid_min) + ", " + str(sigmoid_max))
        # assert config.wordLength == 8, "Sigmoid not implemented for anything other than 8-bits"
        # float_max = sigmoid_max * scale_in
        # scale_comp = self.getScale(float_max, bw=bitwidth_in)
        getLogger().debug("Sigmoid fixed point scale in Zero Skew: " + str(scale_out))
        if bitwidth_temp == None:
            bitwidth_temp = 32 if (config.wordLength == 8) else 64
        
        M1, N1 = self.getQuantizedMultiplierLTO(scale_in, bitwidth_temp, bitwidth_in)
        N1 -= ((31 if (bitwidth_temp == 32) else 63) + (27 if (bitwidth_temp == 32) else 55))
        M2, N2 = self.getQuantizedMultiplierLTO(1.0/scale_out, bitwidth_temp, bitwidth_in)
        N2 -= 23 if (bitwidth_temp == 32) else 46
        
        clamp_min, clamp_max = self.getClampValues(bitwidth_in)
        return M1, N1, M2, N2, min(abs(clamp_max), abs(clamp_min))


    def visitSigmoid(self, node: AST.Func):
        # y = max(min( x/4 + 2/4 , 1), 0), 1).
        # Old implementation, fixed point code uses linear approximation.
        # Floating point implementation uses math.h.
        denominator = 2
        addition = 0.5
        sigmoid_limit = 1

        (prog_in, expr_in) = self.visit(node.expr)

        type_in = node.expr.type
        [I, J] = type_in.shape

        expr_out = self.getTempVar()

        # Read input scales and bit-width.
        bitwidth_in, scale_in, zero_in = self.getBitwidthScaleZeros(expr_in.idf)
        intv_in = self.varIntervals[expr_in.idf]

        # If input is demoted to lower bit-width, demote the output variable to lower bit-width as well.
        if expr_in.idf in self.demotedVarsList:
            self.demotedVarsList.append(expr_out.idf)
            self.varsForBitwidth[expr_out.idf] = config.wordLength // 2
        else:
            self.varsForBitwidth[expr_out.idf] = config.wordLength

        # Scale sigmoid limit and other constants.
        addition_int = self.getNumInZeroSkew(addition, scale_in, zero_in)
        sigmoid_limit_int = self.getNumInZeroSkew(sigmoid_limit, scale_in, zero_in)

        # Compute new interval.
        [m, M] = intv_in
        m, M = float(scale_in*(m - zero_in)), float(scale_in * (M - zero_in))
        m_new = max(min((m / denominator) + addition,
                        sigmoid_limit), 0)
        M_new = max(min((M / denominator) + addition,
                        sigmoid_limit), 0)
        assert m_new <= M_new, "The range of sigmoid has changed. Re-check the assertion."

        scale_out, zero_out = self.getScaleAndZero(0, 1, bw=self.varsForBitwidth[expr_out.idf])
        # scale_out = self.getScale(1.5) + ((config.wordLength // 2 + self.demotedVarsOffsets[expr_in.idf]) if expr_in.idf in self.demotedVarsList else 0)

        # # Computing hyperparameters for linear approximation of Sigmoid.
        # max_val = max(abs(m_new), abs(M_new))
        # max_val_f = np.ldexp(max_val, scale_in)

        # if forFloat():
        #     addition_ir = IR.Float(addition)
        #     sigmoid_limit_ir = IR.Float(sigmoid_limit)
        # else:
        #     addition_ir = addition_int
        #     sigmoid_limit_ir = sigmoid_limit_int

        # scale_in_num = 2 ** -scale_in
        # scale_out_num = 2 ** -scale_out

        # bitwidth_out, scale_out, zero_out = self.getBitwidthScaleZeros(expr_out.idf) 
        expr_in.inputVar = False

        M1, N1, M2, N2, clamp_radius = self.getSigmoidShrAndN(scale_in, zero_in, scale_out, zero_out, intv_in, bitwidth_in)

        comment = IR.Comment("Sigmoid(" + expr_in.idf + ")", self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        funcCall = IR.FuncCall("Sigmoid", {
            expr_in: "A",
            expr_out: "B",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Int(-zero_in): "zero_in",
            IR.Int(M1): "M1",
            IR.Int(-N1): "N1",
            IR.Int(zero_out): "zero_out",
            IR.Int(M2): "M2",
            IR.Int(-N2): "N2",
            IR.Int(clamp_radius): "clamp_radius"
        }) if not self.vbwEnabled else IR.FuncCall("Sigmoid<int%d_t>"%(bitwidth_in), {
            expr_in: "A",
            expr_out: "B",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Int(-zero_in): "zero_in",
            IR.Int(M1): "M1",
            IR.Int(-N1): "N1",
            IR.Int(zero_out): "zero_out",
            IR.Int(M2): "M2",
            IR.Int(-N2): "N2",
            IR.Int(clamp_radius): "clamp_radius"
        })

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        debugPrint = IR.FuncCall("debugPrint", {
                expr_out: "expr",
                # expr_temp: "T",
                IR.Int(I): "I",
                IR.Int(J): "J",
                IR.Float(scale_out): "scale",
                IR.Int(zero_out): "zero",
                IR.String(expr_out): "varName"
            })

        prog_sigmoid = IR.Prog([comment, funcCall, debugPrint])

        prog_out = IRUtil.concatPrograms(prog_in, prog_sigmoid)

        # Updating metadata.
        self.varDeclarations[expr_out.idf] = type_in
        self.varScales[expr_out.idf] = scale_out
        self.varZeros[expr_out.idf] = zero_out
        self.varIntervals[expr_out.idf] = (0,0)

        # Print log.
        self.log.print(comment.msg)
        self.log.print("\tInput1: scale = %f, zero = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in.idf],) + (self.varZeros[expr_in.idf],) + self.varIntervals[expr_in.idf]))
        self.log.print("\tOutput: scale = %f, zero = %d, interval = [%d, %d]" % (
            (self.varScales[expr_out.idf],) + (self.varZeros[expr_out.idf],) + self.varIntervals[expr_out.idf]))


        return (prog_out, expr_out)
    
    # let a[i1:+n1][i2:+n2]... = ...
    def visitLeftSplice(self, node: AST.LeftSplice, expr_in, nodeVarType):
        # Used to assign a splice of a tensor to some value.
        vars_in = []
        progs_in = []
        for var in node.vars:
            part_prog_in, part_expr_in = self.visit(var)
            progs_in.append(part_prog_in)
            vars_in.append(part_expr_in)

        expr_out = IR.Var(node.expr) 

        # Read input and output scales.
        bw_in, scale_in, zero_in = self.getBitwidthScaleZeros(expr_in.idf)
        bw_out, scale_out, zero_out = self.getBitwidthScaleZeros(expr_out.idf)

        loop_dim = len(node.sizes)

        # Computing indices on LHS and RHS in for loop.
        iters_in = self.getTempIterators(loop_dim) 
        iters_out = self.getTempVars(loop_dim)

        loopShape = [] # Shape of tensor to be assigned, limits of the iterators.
        loopIters = [] # Iterators to iterate across different dimensions.
        loopAssns = [] # Assignment statements within the loop body.
        for order in range(loop_dim):
            loopShape.append(node.sizes[order])
            loopIters.append(iters_in[order])
            loopAssns.append(IR.Assn(iters_out[order], IRUtil.add(iters_in[order], vars_in[order])))

        expr_in_idx = IRUtil.addIndex(expr_in, iters_in)
        expr_out_idx = IRUtil.addIndex(expr_out, iters_out)

        # Adjusting scale in the input and output code will be done in a single command at the end

        if math.fabs(scale_in - scale_out) > 0.000001 or (zero_in != zero_out):
            cmd2 = IR.Assn(expr_out_idx, self.scale_adjust(expr_in_idx, scale_in, scale_out, zero_in, zero_out))
        else:
            cmd2 = IR.Assn(expr_out_idx, expr_in_idx)

        # Compared to right splice iters_out is the index for LHS.
        loop = IRUtil.loop(loopShape, loopIters, loopAssns + [
                cmd2
            ])

        # Comments for showing the input used to generate the given output.
        out_indices = ']['.join([i.idf for i in iters_out])
        in_indices = ']['.join([i.idf for i in iters_in])
        comment = IR.Comment("%s[%s] = %s[%s]"%(expr_out_idx.idf, out_indices, expr_in_idx.idf, in_indices), self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth
        prog_splice = IR.Prog([comment] + loop)

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        # In case the target variable is contiguous, we can optimize (use memcpy instead of a loop).
        canOptimize = True
        loopShapeMustBeOne = False
        for i in range(len(loopShape) - 1, -1, -1):
            if loopShapeMustBeOne:
                if loopShape[i] != 1:
                    canOptimize = False
            else:
                if loopShape[i] == nodeVarType.shape[i]:
                    continue
                elif loopShape[i] < nodeVarType.shape[i]:
                    loopShapeMustBeOne = True
                    continue
                else:
                    assert False, "Illegal State, subtensor dimensions must be less than original tensor dimensions"
        canOptimize = canOptimize and (expr_in.idf not in self.globalVars) and bw_in == bw_out and (math.fabs(scale_in - scale_out) < 0.000001 and (zero_in == zero_out))

        if canOptimize:
            prog_splice = IR.Prog([comment, IR.Memcpy(expr_out, expr_in, np.prod(loopShape), vars_in, [IR.Int(0) for i in range(len(vars_in))])])
        else:
            assert True

        prog_out = IR.Prog([])
        for prog in progs_in:
            prog_out = IRUtil.concatPrograms(prog_out, prog)
        prog_out = IRUtil.concatPrograms(prog_out, prog_splice)

        # Update declarations.
        for var in iters_out:
            self.varDeclarations[var.idf] = Type.Int()
            self.internalVars.append(var.idf)

        return (prog_out, expr_out)

    def scale_adjust(self, e: IR.Expr, scale_in: float, scale_out: float, zero_in: int, zero_out: int) -> IR.Expr:
        mul = scale_out / scale_in
        add = zero_out - int(mul * zero_in)
        return IR.IntBop(IR.IntBop(e, IR.Op.Op['*'], IR.Float(mul)), IR.Op.Op['+'], IR.Int(add))

    
     # out = $x[start:end] in
    def visitSum(self, node: AST.Sum):
        assert False, "Not implemented for Zero Skew Representation"
    
        # out = loop(x[start:end]) (expr) in
    def visitLoop(self, node: AST.Loop):
        '''
        for (i = 0; i < n; i++)
          prog_in
        '''

        idf = node.mutableVar.name
        self.mutableVars.append(idf)

        # Update the scale and interval of the mutable variable only during fixed-point code generation.
        scale, zero, intv = self.readProfileForMutableVars(idf)
        bitwidth, _, _ = self.getBitwidthScaleZeros(idf) # (init 0 default scale currently stored in varScales which has to be overwritten).
        if bitwidth != config.wordLength:
            idfs = idf
            while idfs in self.substitutions.keys():
                idfs = self.substitutions[idfs]
            scale, zero = self.adjustScaleAndZero(scale, zero) 
        self.varScales[idf] = scale
        self.varZeros[idf] = zero
        self.varIntervals[idf] = intv

        prevVarDecls = dict(self.varDeclarations)

        start, end = node.start, node.end

        comment = IR.Comment("loop(%s = [%d, %d], %s)" % (
            node.name, start, end, idf), self.counter_inst+1) # The comment is before visiting the loop statements so the self.counter_inst's earlier value is used.
        self.allDepths[self.counter_inst+1] = self.curDepth

        self.counter_inst += 1
        self.curDepth += 1

        (prog_in, expr_in) = self.visit(node.expr)

        self.curDepth -=1

        # This variable contains variables that need to be declared within the for loop.
        # No longer needed as the current memory management (config.x86MemoryOptimize) takes care of variables declared locally within loop.
        forDecls = {}

        assert start == 0, "'loop' operator currently supports only iterations starting from 0."

        var = IR.Var(node.name)

        loop = IR.For(var, 0, IRUtil.lt(
            var, IR.Int(end - start)), prog_in.cmd_l, 0, forDecls)

        self.updateLiveRange([expr_in])

        profile = []

        prog_out = IR.Prog([comment, loop] + profile)

        return (prog_out, expr_in)

    # Used by old SeeDot for reading profile for exponentiation.
    # NOTE: Works only when there is one variable for exponentiation.
    # New SeeDot uses same data driven scaling platform for all variables.
    def readProfileForMutableVars(self, idf):
        # Data-driven parameters.
        inputFile = getProfileLogFile()

        with open(inputFile, 'r') as f:
            for line in f:
                entries = line.strip().split(", ")
                row = list(map(float, entries))
                self.mutableVarsProfile.append(row)

        [minVal, maxVal] = self.mutableVarsProfile[0]

        scale, zero = self.getScaleAndZero(minVal, maxVal)
        intv = self.getInterval(minVal, maxVal, scale, zero)

        return scale, zero, intv
    
    def getClampValues(self, bitwidth_out):
        return (-1*config.maxVar8Bit, config.maxVar8Bit) if bitwidth_out == 8 else (-1*config.maxVar16Bit, config.maxVar16Bit)