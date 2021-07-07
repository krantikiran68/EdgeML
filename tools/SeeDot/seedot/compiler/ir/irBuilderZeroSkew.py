# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import operator
import math

from numpy.core.fromnumeric import shape

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
        self.vbwEnabled = config.vbwEnabled 
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
        if math.fabs(M - 1.0) < 0.000000001:
            M = 1.0 - 0.0000001
        # assert (M < 1.0 and M > 0.0 ), "The multiplier in matmul must be in (0,1)"
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
        if expr_out.idf == 'a':
            print("Hre!")
        bitwidth_out, scale_out, zero_out = self.getBitwidthScaleZeros(expr_out.idf)
        bitwidth_temp = self.getTempBitwidth(bitwidth_in_A, bitwidth_in_B, "mul", bitwidth_out)
    
        type_in_A, type_in_B = node.expr1.type, node.expr2.type
        type_out = node.type

        [I, J] = type_in_A.shape
        [J, K] = type_in_B.shape

        ## Removed tree sum for the purpose of zero Skew representation
        clamp_min, clamp_max = self.getClampValues(bitwidth_out)
        intv_out = (0,0)

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

        
        comment = IR.Comment(expr_in_A.idf + ' * ' + expr_in_B.idf, self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        # Bit-width for temporary variables.
        bitwidth_mul = bitwidth_temp# self.getTempBitwidth(bitwidth_in_A, bitwidth_in_B, "mul")

        # If one variable is already used as a sparse matrix, prevent further use as a dense matrix.
        assert expr_in_A.idf + "idx" not in self.sparseMatrixSizes.keys(), "Cannot use same matrix %s for both sparse and dense multiplication" % expr_in_A.idf

        funcCall = IR.FuncCall("MatMul" + c, {
            expr_in_A: "A",
            expr_in_B: "B",
            expr_out: "C",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Int(K): "K",
            IR.Float(scale_in_A): "scale_in_A",
            IR.Float(scale_in_B): "scale_in_B",
            IR.Float(scale_out): "scale_out",
            IR.Int(-1*zero_in_A): "zero_A",
            IR.Int(-zero_in_B): "zero_B",
            IR.Int(zero_out): "zero_C",
            IR.Int(M0): "M0",
            IR.Int(-N): "N",
            IR.Int(clamp_min): "clamp_min",
            IR.Int(clamp_max): "clamp_max"
        }) if not self.vbwEnabled else IR.FuncCall("MatMul" + c + ("<uint%d_t, uint%d_t, int%d_t, uint%d_t>"%(bitwidth_in_A, bitwidth_in_B, bitwidth_mul, bitwidth_out)), {
            expr_in_A: "A",
            expr_in_B: "B",
            expr_out: "C",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Int(K): "K",
            IR.Float(scale_in_A): "scale_in_A",
            IR.Float(scale_in_B): "scale_in_B",
            IR.Float(scale_out): "scale_out",
            IR.Int(-1*zero_in_A): "zero_A",
            IR.Int(-zero_in_B): "zero_B",
            IR.Int(zero_out): "zero_C",
            IR.Int(M0): "M0",
            IR.Int(-N): "N",
            IR.Int(clamp_min): "clamp_min",
            IR.Int(clamp_max): "clamp_max",
            # IR.Int(demote): "demote"
        })

        debugPrint = []
        if config.zeroSkewDebug:
            debugPrint.append(IR.FuncCall("debugPrint", {
                expr_out: "expr",
                IR.Int(I): "I",
                IR.Int(K): "J",
                IR.Float(scale_out): "scale",
                IR.Int(zero_out): "zero",
                IR.String(expr_out): "varName"
            }))

        self.counter_inst += 1
        self.updateLiveRange([expr_in_A, expr_in_B, expr_out])

        
        prog_mul = IR.Prog([comment, funcCall] + (debugPrint if config.zeroSkewDebug else []))

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
                new_scale, new_zero = self.adjustScaleAndZero(self.intermediateVarScales[varName], self.intermediateVarZeros[varName], demote=True)
                return config.wordLength // 2, new_scale, new_zero
            else:
                return config.wordLength, self.intermediateVarScales[varName], self.intermediateVarZeros[varName]
        else:
            assert False, "No root found"


    def visitLet(self, node: AST.Let):
        # Visit RHS of the let statement.
        if(isinstance(node.decl, AST.Decl)):
            self.declVarName = node.name
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

                # Reset the self.scale value to the profile generated one.
                self.varScales[idf] = new_scale
                self.varZeros[idf] = new_zero
                self.varIntervals[idf] = new_intv
            else:
                pass
                
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
            new_zero = int(zero * scale / new_scale)
            return new_scale, new_zero
        else: # 16-bit to 8-bit
            new_scale = (scale * config.maxVar16Bit) / config.maxVar8Bit
            new_zero = int(zero * scale / new_scale)
            return new_scale, new_zero
    
    def getScaleAndZero(self, minVal, maxVal, bw=config.wordLength, decl=False, varName = None):
        maxVar = config.maxVar8Bit if bw == 8 else config.maxVar16Bit

        # if decl and varName != 'X':
        #     zero = 0
        #     scale = (maxVal - minVal)/(2 * maxVar)
        #     return scale, int(zero)
        
        if maxVal == minVal:
            scale = math.fabs(1.0/maxVal) if (maxVal > 1.0) else math.fabs(maxVal)
            zero = 0
            return scale, zero
        zero = -minVal

        minVal += zero
        maxVal += zero
        
        if math.fabs(maxVal) < 0.00000001:
            scale = 1.0
        else:
            scale = maxVal / maxVar

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
        scale, zero = self.getScaleAndZero(minVal, maxVal, decl=True, varName = self.declVarName)
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
        if node.value == (-zero * scale):
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
            getLogger().debug("Changed loop assignment in visitInit")
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
        self.varsForBitwidth[expr_out.idf] = bw_out
        if bw_out != config.wordLength:
            self.demotedVarsList.append(expr_out.idf)

        funcCall = IR.FuncCall("Transpose", {
            expr_in: "A",
            expr_out: "B",
            IR.Int(I): "I",
            IR.Int(J): "J"
        }) if not self.vbwEnabled else IR.FuncCall("Transpose<uint%d_t>" % (bw_out), {
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

        funcCall = IR.FuncCall('Reverse' + str(len(node.type.shape)), args) if not self.vbwEnabled else IR.FuncCall('Reverse' + str(len(node.type.shape)) + '<uint' + str(bitwidth_in) + '_t>', args)

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

        bitwidth_mul = self.getTempBitwidth(bitwidth_in_A, bitwidth_in_B, "mul", bitwidth_out)
    
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
            IR.Float(scale_in_A): "scale_in_A",
            IR.Float(scale_in_B): "scale_in_B",
            IR.Float(scale_out): "scale_out",
            IR.Int(-1*zero_in_A): "zero_A",
            IR.Int(-zero_in_B): "zero_B",
            IR.Int(zero_out): "zero_C",
            IR.Int(M0): "M0",
            IR.Int(-N): "N",
            IR.Int(clamp_min): "clamp_min",
            IR.Int(clamp_max): "clamp_max"
        }) if not self.vbwEnabled else IR.FuncCall("MatMulBroadcastA<uint%d_t, uint%d_t, int%d_t, uint%d_t>"%(bitwidth_in_A, bitwidth_in_B, bitwidth_mul, bitwidth_out), {
            a: "A",
            b: "B",
            expr_out: "C",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Float(scale_in_A): "scale_in_A",
            IR.Float(scale_in_B): "scale_in_B",
            IR.Float(scale_out): "scale_out",
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
                IR.Int(I): "I",
                IR.Int(J): "J",
                IR.Float(scale_out): "scale",
                IR.Int(zero_out): "zero",
                IR.String(expr_out): "varName"
            })


        prog_mul = IR.Prog([comment, funcCall] + ([debugPrint] if config.zeroSkewDebug else []))

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
    
    def getTempBitwidth(self, bitwidthA = None, bitwidthB = None, op = None, bitwidthC=None):
        if op == "sparse_mul":
            assert (bitwidthA is not None) and (bitwidthB is not None) and (bitwidthC is not None)
            # assert bitwidthC is None, "Illegal call to getTempBitwidth()"
            # biggerBitWidth = max(bitwidthA, bitwidthB)
            return (32 if min(bitwidthA, bitwidthB) == 8 else 64)
        elif op == "mul":
            assert (bitwidthA is not None) and (bitwidthB is not None) and (bitwidthC is not None)
            # assert bitwidthC is None, "Illegal call to getTempBitwidth()"
            # biggerBitWidth = max(bitwidthA, bitwidthB)
            return (32 if max(bitwidthA, bitwidthB, bitwidthC) == 8 else 64)
        elif op == "add":
            assert (bitwidthA is not None) and (bitwidthB is not None) and (bitwidthC is not None)
            # assert bitwidthC is not None, "Illegal call to getTempBitwidth()"
            # biggerBitWidth = max(bitwidthA, bitwidthB, bitwidthC)
            return (32 if max(bitwidthA, bitwidthB, bitwidthC) == 8 else 64)
        elif op == "sigmoid":
            assert bitwidthA is not None
            return 32 if (bitwidthA == 8) else 64
        elif op == "tanh":
            assert bitwidthA is not None
            return 32 if (bitwidthA == 8) else 64
        elif op == "exp":
            assert bitwidthA is not None and bitwidthC is not None
            return 32 if (max(bitwidthA, bitwidthC) == 8) else 64
        elif op == None:
            getLogger().debug("Non add-sub specified for temp bitwidth")
            return (32 if config.wordLength == 8 else 64)
    
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
        #             adjust = [IR.FuncCall("AdjustScaleShl" + (("<uint%d_t>"%bitwidth_out) if self.vbwEnabled else ""), {
        #                         expr_out: "A",
        #                         IR.Int(I): "I",
        #                         IR.Int(J): "J",
        #                         IR.Int(diff): "scale"
        #                     })]
        #         else:
        #             adjust = [IR.FuncCall("AdjustScaleShr" + (("<uint%d_t>"%bitwidth_out) if self.vbwEnabled else ""), {
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

        bitwidth_mul = self.getTempBitwidth(bitwidth_in_A, bitwidth_in_B, "mul", bitwidth_out)

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
        }) if not self.vbwEnabled else IR.FuncCall("Hadamard<uint%d_t, uint%d_t, int%d_t, uint%d_t>"%(bitwidth_in_A, bitwidth_in_B, bitwidth_mul, bitwidth_out), {
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

        prog_mul = IR.Prog([comment, funcCall] + ([debugPrint] if config.zeroSkewDebug else []))

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

    # out = mbconv(A, filters, weights, biases, <params>)
    # This is a specialised implementation of mobilenet conv layers which prevent excessive memory bloat during intermediate computations.
    def visitMbconv(self, node: AST.MBConv):
        if not (config.ddsEnabled and config.vbwEnabled):
            assert False, "MBConv is currently only supported if VBW and DDS modes are switched on"

        assert forX86(), "MBConv not implemented for Arduino devices"

        # Process all inputs for MBConv.
        (prog_in_A, expr_in_A) = self.visit(node.expr1)
        (prog_in_F1, expr_in_F1) = self.visit(node.exprF1)
        (prog_in_W1, expr_in_W1) = self.visit(node.exprW1)
        (prog_in_B1, expr_in_B1) = self.visit(node.exprB1)
        (prog_in_F2, expr_in_F2) = self.visit(node.exprF2)
        (prog_in_W2, expr_in_W2) = self.visit(node.exprW2)
        (prog_in_B2, expr_in_B2) = self.visit(node.exprB2)
        (prog_in_F3, expr_in_F3) = self.visit(node.exprF3)
        (prog_in_W3, expr_in_W3) = self.visit(node.exprW3)
        (prog_in_B3, expr_in_B3) = self.visit(node.exprB3)

        [expr_treeSum, expr_out] = self.getTempVars(2)
        [expr_bufX, expr_bufT] = self.getTempVars(2)

        [N, H, W, Cin] = node.expr1.type.shape
        [_, _, _, _, Ct] = node.exprF1.type.shape
        [_, Hf, Wf, _, _] = node.exprF2.type.shape
        [_, _, _, _, Cout] = node.exprF3.type.shape

        # type_treeSum = Type.Tensor([np.max((Hf * Wf, Ct, Cin))])
        type_out = node.type
        type_bufX = Type.Tensor([Hf, W, Ct])
        type_bufT = Type.Tensor([Ct])

        # Process bit-width and scales for all inputs.
        bitwidth_in_A, scale_in_A, zero_in_A = self.getBitwidthScaleZeros(expr_in_A.idf)
        bitwidth_in_F1, scale_in_F1, zero_in_F1 = self.getBitwidthScaleZeros(expr_in_F1.idf)
        bitwidth_in_W1, scale_in_W1, zero_in_W1 = self.getBitwidthScaleZeros(expr_in_W1.idf)
        bitwidth_in_X, scale_in_X, zero_in_X = self.getBitwidthScaleZeros(expr_out.idf + "x1")
        bitwidth_in_X = np.max((bitwidth_in_A, bitwidth_in_F1, bitwidth_in_W1))
        bitwidth_in_B1, scale_in_B1, zero_in_B1 = self.getBitwidthScaleZeros(expr_in_B1.idf)
        bitwidth_in_F2, scale_in_F2, zero_in_F2 = self.getBitwidthScaleZeros(expr_in_F2.idf)
        bitwidth_in_W2, scale_in_W2, zero_in_W2 = self.getBitwidthScaleZeros(expr_in_W2.idf)
        bitwidth_in_T, scale_in_T, zero_in_T = self.getBitwidthScaleZeros(expr_out.idf + "x3")
        bitwidth_in_T = np.max((bitwidth_in_X, bitwidth_in_F2, bitwidth_in_W2))
        bitwidth_in_B2, scale_in_B2, zero_in_B2 = self.getBitwidthScaleZeros(expr_in_B2.idf)
        bitwidth_in_F3, scale_in_F3, zero_in_F3 = self.getBitwidthScaleZeros(expr_in_F3.idf)
        bitwidth_in_W3, scale_in_W3, zero_in_W3 = self.getBitwidthScaleZeros(expr_in_W3.idf)
        bitwidth_in_B3, scale_in_B3, zero_in_B3 = self.getBitwidthScaleZeros(expr_in_B3.idf)
        bitwidth_out, scale_out, zero_out = self.getBitwidthScaleZeros(expr_out.idf)

        # Compute intermediate scales and scaling factors for all operations which are included in MBConv.
        # Stage 1 Step 1: Multiplication
        bitwidth_temp1 = self.getTempBitwidth(bitwidth_in_A, bitwidth_in_F1, "mul", bitwidth_in_X)
        M11, N11 = self.getMatMulShrAndN(scale_in_A, scale_in_F1, scale_in_X, zero_in_A, zero_in_F1, zero_in_X, bitwidth_in_A, bitwidth_in_F1, bitwidth_temp1, bitwidth_in_X)

        # Stage 1 Step 2: Batch Normalisation and ReLU6
        bitwidth_temp_ub1 = self.getTempBitwidth(bitwidth_in_X, bitwidth_in_W1, "mul", bitwidth_in_X)
        (left_shift1, M12, N12, M13, N13, M14, N14) = self.getScaleAndZeroForAddAndSub(scale_in_X, zero_in_X, scale_in_B1, zero_in_B1, scale_in_X, zero_in_X, bitwidth_in_X, bitwidth_in_B1, bitwidth_temp_ub1, bitwidth_in_X, operator.add)
        # bitwidth_temp = self.getTempBitwidth(bitwidth_in_X, bitwidth_in_W1, "mul", bitwidth_in_X)
        M15, N15 = self.getMatMulShrAndN(scale_in_X, scale_in_W1, scale_in_X, zero_in_X, zero_in_W1, zero_in_X, bitwidth_in_X, bitwidth_in_W1, bitwidth_temp_ub1, bitwidth_in_X)
        clamp_min_X, clamp_max_X = self.getClampValues(bitwidth_in_X)

        # Stage 2 Step 1: Multiplication
        bitwidth_temp2 = self.getTempBitwidth(bitwidth_in_X, bitwidth_in_F2, "mul", bitwidth_in_T)
        M21, N21 = self.getMatMulShrAndN(scale_in_X, scale_in_F2, scale_in_T, zero_in_X, zero_in_F2, zero_in_T, bitwidth_in_X, bitwidth_in_F2, bitwidth_temp2, bitwidth_in_T)

        # Stage 2 Step 2: Batch Normalisation and ReLU6
        bitwidth_temp_ub2 = self.getTempBitwidth(bitwidth_in_T, bitwidth_in_W2, "mul", bitwidth_in_T)
        (left_shift2, M22, N22, M23, N23, M24, N24) = self.getScaleAndZeroForAddAndSub(scale_in_T, zero_in_T, scale_in_B2, zero_in_B2, scale_in_T, zero_in_T, bitwidth_in_T, bitwidth_in_B2, bitwidth_temp_ub2, bitwidth_in_T, operator.add)
        # bitwidth_temp = self.getTempBitwidth(bitwidth_in_T, bitwidth_in_W2, "mul", bitwidth_in_T)
        M25, N25 = self.getMatMulShrAndN(scale_in_T, scale_in_W2, scale_in_T, zero_in_T, zero_in_W2, zero_in_T, bitwidth_in_T, bitwidth_in_W2, bitwidth_temp_ub2, bitwidth_in_T)
        clamp_min_T, clamp_max_T = self.getClampValues(bitwidth_in_T)

        # Stage 3 Step 1: Multiplication
        bitwidth_temp3 = self.getTempBitwidth(bitwidth_in_T, bitwidth_in_F3, "mul", bitwidth_out)
        M31, N31 = self.getMatMulShrAndN(scale_in_T, scale_in_F3, scale_out, zero_in_T, zero_in_F3, zero_out, bitwidth_in_T, bitwidth_in_F3, bitwidth_temp3, bitwidth_out)

        # Stage 3 Step 2: Batch Normalisation
        bitwidth_temp_ub3 = self.getTempBitwidth(bitwidth_out, bitwidth_in_W3, "mul", bitwidth_out)
        (left_shift3, M32, N32, M33, N33, M34, N34) = self.getScaleAndZeroForAddAndSub(scale_out, zero_out, scale_in_B3, zero_in_B3, scale_out, zero_out, bitwidth_out, bitwidth_in_B3, bitwidth_temp_ub3, bitwidth_out, operator.add)
        # bitwidth_temp = self.getTempBitwidth(bitwidth_out, bitwidth_in_W3, "mul", bitwidth_out)
        M35, N35 = self.getMatMulShrAndN(scale_out, scale_in_W3, scale_out, zero_out, zero_in_W3, zero_out, bitwidth_out, bitwidth_in_W3, bitwidth_temp_ub3, bitwidth_out)
        clamp_min_C, clamp_max_C = self.getClampValues(bitwidth_out)

        expr_in_A.inputVar = False
        expr_in_F1.inputVar = False
        expr_in_W1.inputVar = False
        expr_in_B1.inputVar = False
        expr_in_F2.inputVar = False
        expr_in_W2.inputVar = False
        expr_in_B2.inputVar = False
        expr_in_F3.inputVar = False
        expr_in_W3.inputVar = False
        expr_in_B3.inputVar = False
        expr_out.inputVar = False
        expr_bufX.inputVar = False
        expr_bufT.inputVar = False

        bitwidth_temp = np.max((bitwidth_temp1, bitwidth_temp2, bitwidth_temp3))

        # Setting metadata.
        self.varsForBitwidth[expr_bufX.idf] = bitwidth_in_X
        self.varsForBitwidth[expr_bufT.idf] = bitwidth_in_T

        comment = IR.Comment('MBconv(%s)' %(expr_in_A.idf), self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        argMap = {
            expr_in_A: "A",
            expr_in_F1: "F1",
            expr_in_W1: "BN1W",
            expr_in_B1: "BN1B",
            expr_in_F2: "F2",
            expr_in_W2: "BN2W",
            expr_in_B2: "BN2B",
            expr_in_F3: "F3",
            expr_in_W3: "BN3W",
            expr_in_B3: "BN3B",
            expr_out: "C",
            expr_bufX: "X",
            expr_bufT: "T",
            IR.Int(N): "N",
            IR.Int(H): "H",
            IR.Int(W): "W",
            IR.Int(Cin): "Cin",
            IR.Int(Ct): "Ct",
            IR.Int(Hf): "HF",
            IR.Int(Wf): "WF",
            IR.Int(Cout): "Cout",
            IR.Int(type_out.shape[1]): "Hout",
            IR.Int(type_out.shape[2]): "Wout",
            IR.Int(node.padding[0]): "HPADL",
            IR.Int(node.padding[1]): "HPADR",
            IR.Int(node.padding[2]): "WPADL",
            IR.Int(node.padding[3]): "WPADR",
            IR.Int(node.stride[0]): "HSTR",
            IR.Int(node.stride[1]): "WSTR",
            IR.Int(-zero_in_A): "zeroA",
            IR.Int(-zero_in_F1): "zeroF1",
            IR.Int(-zero_in_W1): "zeroBN1W",
            IR.Int(-zero_in_B1): "zeroBN1B",
            IR.Int(-zero_in_F2): "zeroF2",
            IR.Int(-zero_in_W2): "zeroBN2W",
            IR.Int(-zero_in_B2): "zeroBN2B",
            IR.Int(-zero_in_F3): "zeroF3",
            IR.Int(-zero_in_W3): "zeroBN3W",
            IR.Int(-zero_in_B3): "zeroBN3B",
            IR.Int(zero_out): "zeroC",
            IR.Int(zero_in_X): "zeroX",
            IR.Int(zero_in_T): "zeroT",
            IR.Int(left_shift1): "left_shift1",
            IR.Int(left_shift2): "left_shift2",
            IR.Int(left_shift3): "left_shift3",
            IR.Int(M11): "M11",
            IR.Int(-N11): "N11",
            IR.Int(M12): "M12",
            IR.Int(-N12): "N12",
            IR.Int(M13): "M13",
            IR.Int(-N13): "N13",
            IR.Int(M14): "M14",
            IR.Int(-N14): "N14",
            IR.Int(M15): "M15",
            IR.Int(-N15): "N15",
            IR.Int(M21): "M21",
            IR.Int(-N21): "N21",
            IR.Int(M22): "M22",
            IR.Int(-N22): "N22",
            IR.Int(M23): "M23",
            IR.Int(-N23): "N23",
            IR.Int(M24): "M24",
            IR.Int(-N24): "N24",
            IR.Int(M25): "M25",
            IR.Int(-N25): "N25",
            IR.Int(M31): "M31",
            IR.Int(-N31): "N31",
            IR.Int(M32): "M32",
            IR.Int(-N32): "N32",
            IR.Int(M33): "M33",
            IR.Int(-N33): "N33",
            IR.Int(M34): "M34",
            IR.Int(-N34): "N34",
            IR.Int(M35): "M35",
            IR.Int(-N35): "N35",
            IR.Int(clamp_min_X): "clamp_min_X",
            IR.Int(clamp_max_X): "clamp_max_X",
            IR.Int(clamp_min_T): "clamp_min_T",
            IR.Int(clamp_max_T): "clamp_max_T",
            IR.Int(clamp_min_C): "clamp_min_C",
            IR.Int(clamp_max_C): "clamp_max_C"
        }

        # Generating the argument map which is used in the codegen.
        templateArgs = ("<uint%s_t" + (", uint%s_t" * 12) + ", int%s_t, int%s_t, int%s_t, int%s_t>") % (bitwidth_in_A, bitwidth_in_F1, bitwidth_in_W1, bitwidth_in_B1, bitwidth_in_F2, bitwidth_in_W2, bitwidth_in_B2, bitwidth_in_F3, bitwidth_in_W3, bitwidth_in_B3, bitwidth_out, bitwidth_in_X, bitwidth_in_T, bitwidth_temp_ub1, bitwidth_temp_ub2, bitwidth_temp_ub3, bitwidth_temp)
        funcCall = IR.FuncCall("MBConv" + templateArgs, argMap)

        self.counter_inst += 1
        self.updateLiveRange([expr_in_A, expr_in_F1, expr_in_F2, expr_in_F3, expr_in_W1, expr_in_W2, expr_in_W3, expr_in_B1, expr_in_B2, expr_in_B3, expr_out, expr_bufX, expr_bufT])

        prog_mbconv = IR.Prog([comment, funcCall])
        prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_F1, prog_in_W1, prog_in_B1, prog_in_F2, prog_in_W2, prog_in_B2, prog_in_F3, prog_in_W3, prog_in_B3, prog_mbconv)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = type_out
        self.varDeclarations[expr_bufX.idf] = type_bufX
        self.varDeclarations[expr_bufT.idf] = type_bufT

        self.varScales[expr_out.idf] = scale_out
        self.varScales[expr_bufX.idf] = scale_in_X
        self.varScales[expr_bufT.idf] = scale_in_T

        self.varZeros[expr_out.idf] = zero_out
        self.varZeros[expr_bufX.idf] = zero_in_X
        self.varZeros[expr_bufT.idf] = zero_in_T

        # Intervals not needed necesarily for the compiler to run, updating this variable for being compatible with old SeeDot (PLDI '19).
        self.varIntervals[expr_out.idf] = (0, 0)
        self.varIntervals[expr_bufX.idf] = (0, 0)
        self.varIntervals[expr_bufT.idf] = (0, 0)

        # Printing log.
        self.log.print(comment.msg)
        self.log.print("\tInput1: scale = %d, zero = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in_A.idf],) + (self.varZeros[expr_in_A.idf],) + self.varIntervals[expr_in_A.idf]))
        self.log.print("\tOutput: scale = %d, zero = %d, interval = [%d, %d]" % (
            (self.varScales[expr_out.idf],) + (self.varZeros[expr_out.idf],) + self.varIntervals[expr_out.idf]))

        return (prog_out, expr_out)

    # out = conv(A, B, <params>)
    def visitConvolution(self, node: AST.Convolution):
        (prog_in_A, expr_in_A) = self.visit(node.expr1)
        (prog_in_B, expr_in_B) = self.visit(node.expr2)

        [expr_treeSum, expr_out] = self.getTempVars(2)

        [N, H, W, Cin] = node.expr1.type.shape
        [G, Hf, Wf, CinF, CoutF] = node.expr2.type.shape

        # type_treeSum = Type.Tensor([Hf * Wf * CinF])
        type_out = node.type

        # Read input scales.
        bitwidth_in_A, scale_in_A, zero_in_A = self.getBitwidthScaleZeros(expr_in_A.idf)
        bitwidth_in_B, scale_in_B, zero_in_B = self.getBitwidthScaleZeros(expr_in_B.idf)
        # Read output scales.
        bitwidth_out, scale_out, zero_out = self.getBitwidthScaleZeros(expr_out.idf)

        intv_out = (0, 0)
        clamp_min, clamp_max = self.getClampValues(bitwidth_out)

        bitwidth_temp = self.getTempBitwidth(bitwidth_in_A, bitwidth_in_B, "mul", bitwidth_out)

        # Compute scaling hyperparameters given input and output scales. If static scaling of old SeeDot is used, also compute the output scale and bit-width.
        M0, N0 = self.getMatMulShrAndN(scale_in_A, scale_in_B, scale_out, zero_in_A, zero_in_B, zero_out, bitwidth_in_A, bitwidth_in_B, bitwidth_temp, bitwidth_out)

        expr_in_A.inputVar = False
        expr_in_B.inputVar = False
        expr_out.inputVar = False
        expr_treeSum.inputVar = False

        comment = IR.Comment('conv(%s, %s)' %(expr_in_A.idf, expr_in_B.idf), self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        argMap = {
            expr_in_A: "A",
            expr_in_B: "B",
            expr_out: "C",
            IR.Int(N): "N",
            IR.Int(H): "H",
            IR.Int(W): "W",
            IR.Int(Cin): "CIN",
            IR.Int(Hf): "HF",
            IR.Int(Wf): "WF",
            IR.Int(CinF): "CINF",
            IR.Int(CoutF): "COUTF",
            IR.Int(type_out.shape[1]): "HOUT",
            IR.Int(type_out.shape[2]): "WOUT",
            IR.Int(node.padding[0]): "HPADL",
            IR.Int(node.padding[1]): "HPADR",
            IR.Int(node.padding[2]): "WPADL",
            IR.Int(node.padding[3]): "WPADR",
            IR.Int(node.stride[0]): "HSTR",
            IR.Int(node.stride[1]): "WSTR",
            IR.Int(node.dilation[0]): "HDL",
            IR.Int(node.dilation[1]): "WDL",
            IR.Int(G): "G",
            IR.Float(scale_in_A): "scale_in_A",
            IR.Float(scale_in_B): "scale_in_B",
            IR.Float(scale_out): "scale_out",
            IR.Int(-zero_in_A): "zero_A",
            IR.Int(-zero_in_B): "zero_B",
            IR.Int(zero_out): "zero_C",
            IR.Int(M0): "M0",
            IR.Int(-N0): "N",
            IR.Int(clamp_min): "clamp_min",
            IR.Int(clamp_max): "clamp_max",
        }

        if not self.vbwEnabled:
            funcCall = IR.FuncCall("Convolution", argMap)
        else:
            funcCall = IR.FuncCall("Convolution" + ("<uint%d_t, uint%d_t, int%d_t, uint%d_t>"%(bitwidth_in_A, bitwidth_in_B, bitwidth_temp, bitwidth_out)), argMap)

        self.counter_inst += 1
        self.updateLiveRange([expr_in_A, expr_in_B, expr_out])

        if Hf == Wf == CinF == CoutF == 1 and bitwidth_in_A == bitwidth_out:
            self.setMemorySharableVariables(expr_in_A, expr_out)

        debugPrint = []
        if config.zeroSkewDebug:
            debugPrint.append(IR.FuncCall("debugPrint", {
                expr_out: "expr",
                IR.Int(N): "N",
                IR.Int(type_out.shape[1]): "H",
                IR.Int(type_out.shape[2]): "W",
                IR.Int(CoutF * G): "C",
                IR.Float(scale_out): "scale",
                IR.Int(zero_out): "zero",
                IR.String(expr_out): "VarName"
            }))

        prog_conv = IR.Prog([comment, funcCall] + (debugPrint if config.zeroSkewDebug else []))
        prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_conv)

        # Update context for output variable.
        self.varDeclarations[expr_out.idf] = type_out
        self.varScales[expr_out.idf] = scale_out
        self.varZeros[expr_out.idf] = zero_out
        self.varIntervals[expr_out.idf] = intv_out

        # Print log.
        self.log.print(comment.msg)
        self.log.print("\tInput1: scale = %f, zero = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in_A.idf],) + (self.varZeros[expr_in_A.idf],) + self.varIntervals[expr_in_A.idf]))
        self.log.print("\tInput2: scale = %f, zero = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in_B.idf],) + (self.varZeros[expr_in_B.idf],) + self.varIntervals[expr_in_B.idf]))
        self.log.print("\tOutput: scale = %f, zero = %d, interval = [%d, %d]" % (
            (self.varScales[expr_out.idf],) + (self.varZeros[expr_out.idf],) + self.varIntervals[expr_out.idf]))

        return (prog_out, expr_out)

    # out = in_A <+-> in_B
    def visitBopAddOrSubCir(self, node: AST.Bop1):
        (prog_in_A, expr_in_A) = self.visit(node.expr1)
        (prog_in_B, expr_in_B) = self.visit(node.expr2)

        type_out = node.type

        if node.op == SeeDotParser.ADDCIR:
            (op_ir, op_fn) = (IR.Op.Op['+'], operator.add)
            add = True
        elif node.op == SeeDotParser.SUBCIR:
            (op_ir, op_fn) = (IR.Op.Op['-'], operator.sub)
            add = False

        assert op_fn == operator.add, "Compiler currently does not support convolution-like subtraction."

        expr_out = self.getTempVar()

        # Read input scales and bit-widths.
        bitwidth_in_A, scale_in_A, zero_in_A = self.getBitwidthScaleZeros(expr_in_A.idf)
        bitwidth_in_B, scale_in_B, zero_in_B = self.getBitwidthScaleZeros(expr_in_B.idf)
        # Read output scales and bit-widths.
        bitwidth_out, scale_out, zero_out = self.getBitwidthScaleZeros(expr_out.idf)

        clamp_min, clamp_max = self.getClampValues(bitwidth_out)

        # Compute scaling hyperparameters given input and output scales. If static scaling of old SeeDot is used, also compute the output scale and bit-width.
        bitwidth_temp = self.getTempBitwidth(bitwidth_in_A, bitwidth_in_B, "add", bitwidth_out)
        (left_shift, shrA, nA, shrB, nB, shrC, nC) = self.getScaleAndZeroForAddAndSub(scale_in_A, zero_in_A, scale_in_B, zero_in_B, scale_out, zero_out, bitwidth_in_A, bitwidth_in_B, bitwidth_temp, bitwidth_out, op_fn)

        expr_in_A.inputVar = False
        expr_in_B.inputVar = False
        expr_out.inputVar = False

        comment = IR.Comment(expr_in_A.idf + " <" +
                             op_ir.name + "> " + expr_in_B.idf, self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        # Generate function call for the output code depending on whether the input is 2-D or 4-D.
        if type_out.dim == 4:
            [N, H, W, C] = type_out.shape
            funcCall = IR.FuncCall("AddOrSubCir4D", {
                expr_in_A: "A",
                expr_in_B: "B",
                expr_out: "X",
                IR.Int(N): "N",
                IR.Int(H): "H",
                IR.Int(W): "W",
                IR.Int(C): "C",
                IR.Float(scale_in_A): "scale_in_A",
                IR.Float(scale_in_B): "scale_in_B",
                IR.Float(scale_out): "scale_out",
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
                IR.Int(clamp_max): "clamp_max",
                IR.Bool(add): "add"
            }) if not self.vbwEnabled else IR.FuncCall("AddOrSubCir4D" + ("<uint%d_t, uint%d_t, int%d_t, uint%d_t>" % (bitwidth_in_A, bitwidth_in_B, bitwidth_temp, bitwidth_out)), {
                expr_in_A: "A",
                expr_in_B: "B",
                expr_out: "X",
                IR.Int(N): "N",
                IR.Int(H): "H",
                IR.Int(W): "W",
                IR.Int(C): "C",
                IR.Float(scale_in_A): "scale_in_A",
                IR.Float(scale_in_B): "scale_in_B",
                IR.Float(scale_out): "scale_out",
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
                IR.Int(clamp_max): "clamp_max",
                IR.Bool(add): "add"
            })
            debugPrint = IR.FuncCall("debugPrint", {
                expr_out: "expr",
                IR.Int(N): "N",
                IR.Int(H): "H",
                IR.Int(W): "W",
                IR.Int(C): "C",
                IR.Float(scale_out): "scale",
                IR.Int(zero_out): "zero",
                IR.String(expr_out): "varName"
            })
        elif type_out.dim == 2:
            [H, W] = type_out.shape
            funcCall = IR.FuncCall("AddOrSubCir2D", {
                expr_in_A: "A",
                expr_in_B: "B",
                expr_out: "X",
                IR.Int(H): "H",
                IR.Int(W): "W",
                IR.Float(scale_in_A): "scale_in_A",
                IR.Float(scale_in_B): "scale_in_B",
                IR.Float(scale_out): "scale_out",
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
                IR.Int(clamp_max): "clamp_max",
                IR.Bool(add): "add"
            }) if not self.vbwEnabled else IR.FuncCall("AddOrSubCir2D" + ("<uint%d_t, uint%d_t, int%d_t, uint%d_t>" % (bitwidth_in_A, bitwidth_in_B, bitwidth_temp, bitwidth_out)), {
                expr_in_A: "A",
                expr_in_B: "B",
                expr_out: "X",
                IR.Int(H): "H",
                IR.Int(W): "W",
                IR.Float(scale_in_A): "scale_in_A",
                IR.Float(scale_in_B): "scale_in_B",
                IR.Float(scale_out): "scale_out",
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
                IR.Int(clamp_max): "clamp_max",
                IR.Bool(add): "add"
            })
            debugPrint = IR.FuncCall("debugPrint", {
                expr_out: "expr",
                IR.Int(H): "H",
                IR.Int(W): "W",
                IR.Float(scale_out): "scale",
                IR.Int(zero_out): "zero",
                IR.String(expr_out): "varName"
            })
        else:
            assert False, "AddCir only supports 2D and 4D tensors."

        self.counter_inst += 1
        self.updateLiveRange([expr_in_A, expr_in_B, expr_out])

        if bitwidth_in_A == bitwidth_out:
            self.setMemorySharableVariables(expr_in_A, expr_out)

        prog_cir = IR.Prog([comment, funcCall] + ([debugPrint] if config.zeroSkewDebug else []))

        prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_cir)

        # Update metadata.
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

    def visitNormaliseL2(self, node:AST.Func):
        (prog_in, expr_in) = self.visit(node.expr)
        intv_out = (0, 0)

        expr_out = self.getTempVar()
        bw_in, scale_in, zero_in = self.getBitwidthScaleZeros(expr_in.idf)
        bw_out, scale_out, zero_out = self.getBitwidthScaleZeros(expr_out.idf)

        bw_temp = 32 if bw_in == 8 else 64
        # bw_out = bw_in

        # We propagate the demotion of bit-width.
        if bw_out != config.wordLength:
            self.demotedVarsList.append(expr_out.idf)
        self.varsForBitwidth[expr_out.idf] = bw_out

        zero_out = 128 if bw_out == 8 else 32768
        clamp_min, clamp_max = self.getClampValues(bw_out)

        expr_in.inputVar = False

        comment = IR.Comment("normaliseL2(" + expr_in.idf + ")", self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        # Since NormaliseL2 does not get profiled now. We do not demote the output.
        if node.type.dim == 4:
            [N, H, W, C] = node.type.shape
            funcCall = IR.FuncCall("NormaliseL2", {
                expr_in: "A",
                expr_out: "B",
                IR.Int(N): "N",
                IR.Int(H): "H",
                IR.Int(W): "W",
                IR.Int(C): "C",
                IR.Float(scale_in): "scale_in",
                IR.Float(scale_out): "scale_out",
                IR.Int(-1*zero_in): "zero_in",
                IR.Int(zero_out): "zero_out",
                IR.Int(clamp_min): "clamp_min",
                IR.Int(clamp_max): "clamp_max"
            }) if not self.vbwEnabled else IR.FuncCall("NormaliseL2<uint%d_t, int%d_t>"%(bw_in, bw_temp), {
                expr_in: "A",
                expr_out: "B",
                IR.Int(N): "N",
                IR.Int(H): "H",
                IR.Int(W): "W",
                IR.Int(C): "C",
                IR.Float(scale_in): "scale_in",
                IR.Float(scale_out): "scale_out",
                IR.Int(-1*zero_in): "zero_in",
                IR.Int(zero_out): "zero_out",
                IR.Int(clamp_min): "clamp_min",
                IR.Int(clamp_max): "clamp_max"
            })
        else:
            assert False, "inverseL2Norm only supports 4D tensors."

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        self.setMemorySharableVariables(expr_in, expr_out)

        prog_func = IR.Prog([comment, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in, prog_func)

        self.varDeclarations[expr_out.idf] = node.type
        self.varScales[expr_out.idf] = scale_out
        self.varZeros[expr_out.idf] = zero_out
        self.varIntervals[expr_out.idf] = (0, 0)

        return (prog_out, expr_out)

    # out = relu(in)
    def visitRelu(self, node: AST.Func):
        (prog_in, expr_in) = self.visit(node.expr)
        intv_out = (0, 0)

        bitwidth_in_A, scale_in_A, zero_in_A = self.getBitwidthScaleZeros(expr_in.idf)
        m, M = self.getIntervalFromScaleZero(bitwidth_in_A, scale_in_A, zero_in_A)
        scale_out, zero_out = self.getScaleAndZero(0, M, bw=bitwidth_in_A)

        bitwidth_temp = 32
        clamp_min, clamp_max = self.getClampValues(bitwidth_in_A)
        M0, N0 = self.getMatMulShrAndN(scale_in_A, 1.0, scale_out, zero_in_A, 0, zero_out, bitwidth_in_A, bitwidth_in_A, bitwidth_temp, bitwidth_in_A)

        expr_in.inputVar = False

        comment = IR.Comment("relu(" + expr_in.idf + ")", self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        if node.type.dim == 4:
            [N, H, W, C] = node.type.shape
            funcCall = IR.FuncCall("Relu4D", {
                expr_in: "A",
                IR.Int(N): "N",
                IR.Int(H): "H",
                IR.Int(W): "W",
                IR.Int(C): "C",
                IR.Float(scale_in_A): "scale_in",
                IR.Float(scale_out): "scale_out",
                IR.Int(zero_in_A): "zero_A",
                IR.Int(zero_out): "zero_Out",
                IR.Int(M0): "M0",
                IR.Int(-N0): "N",
                IR.Int(clamp_min): "clamp_min",
                IR.Int(clamp_max): "clamp_max"
            })
        elif node.type.dim == 2:
            [H, W] = node.type.shape
            funcCall = IR.FuncCall("Relu2D", {
                expr_in: "A",
                IR.Int(H): "H",
                IR.Int(W): "W",
                IR.Float(scale_in_A): "scale_in_A",
                IR.Float(scale_out): "scale_out",
                IR.Int(zero_in_A): "zero_A",
                IR.Int(zero_out): "zero_Out",
                IR.Int(M0): "M0",
                IR.Int(-N0): "N",
                IR.Int(clamp_min): "clamp_min",
                IR.Int(clamp_max): "clamp_max"
            })
        else:
            assert False, "Relu operator currently only supports 2D and 4D tensors."

        self.counter_inst += 1
        self.updateLiveRange([expr_in])

        prog_relu = IR.Prog([comment, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in, prog_relu)

        self.varIntervals[expr_in.idf] = intv_out
        self.varScales[expr_in.idf] = scale_out
        self.varZeros[expr_in.idf] = zero_out

        return (prog_out, expr_in)

    # out = relu(in)
    def visitRelu6(self, node: AST.Func):
        (prog_in, expr_in) = self.visit(node.expr)
        intv_out = (0, 0)

        type_in = node.expr.type

        expr_out = self.getTempVar()

        # Read input scale and bit-width. Output scale and bit-width are the same as input.
        bitwidth_in_A, scale_in_A, zero_in_A = self.getBitwidthScaleZeros(expr_in.idf)
        scale_out, zero_out = self.getScaleAndZero(0, 6, bw=bitwidth_in_A)

        bitwidth_temp = 32
        clamp_min, clamp_max = self.getClampValues(bitwidth_in_A)
        M0, N0 = self.getMatMulShrAndN(scale_in_A, 1.0, scale_out, zero_in_A, 0, zero_out, bitwidth_in_A, bitwidth_in_A, bitwidth_temp, bitwidth_in_A)

        # If input variable is demoted to 8 bits, demote the output variable to 8 bits too.
        if expr_in.idf in self.demotedVarsList:
            self.demotedVarsList.append(expr_out.idf)
            self.varsForBitwidth[expr_out.idf] = config.wordLength // 2

        expr_in.inputVar = False
        expr_out.inputVar = False

        comment = IR.Comment("relu6(" + expr_in.idf + ")", self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        assert node.type.dim == 4, "Relu6 only implemented for 4 dimensional tensors"
        [N, H, W, C] = node.type.shape
        funcCall = IR.FuncCall("Relu6", {
            expr_in: "A",
            expr_out: "B",
            IR.Int(N): "N",
            IR.Int(H): "H",
            IR.Int(W): "W",
            IR.Int(C): "C",
            IR.Float(scale_in_A): "scale_in",
            IR.Float(scale_out): "scale_out",
            IR.Int(zero_in_A): "zero_A",
            IR.Int(zero_out): "zero_Out",
            IR.Int(M0): "M0",
            IR.Int(-N0): "N",
            IR.Int(clamp_min): "clamp_min",
            IR.Int(clamp_max): "clamp_max",
        }) if not self.vbwEnabled else IR.FuncCall("Relu6<uint%d_t, int%d_t, uint%d_t>" % (bitwidth_in_A, bitwidth_temp, bitwidth_in_A), {
            expr_in: "A",
            expr_out: "B",
            IR.Int(N): "N",
            IR.Int(H): "H",
            IR.Int(W): "W",
            IR.Int(C): "C",
            IR.Float(scale_in_A): "scale_in",
            IR.Float(scale_out): "scale_out",
            IR.Int(zero_in_A): "zero_A",
            IR.Int(zero_out): "zero_Out",
            IR.Int(M0): "M0",
            IR.Int(-N0): "N",
            IR.Int(clamp_min): "clamp_min",
            IR.Int(clamp_max): "clamp_max",
        })

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        prog_relu = IR.Prog([comment, funcCall])

        prog_out = IRUtil.concatPrograms(prog_in, prog_relu)

        # Update metadata.
        self.varIntervals[expr_out.idf] = intv_out
        self.varDeclarations[expr_out.idf] = type_in
        self.varZeros[expr_out.idf] = zero_out
        self.varScales[expr_out.idf] = scale_out

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
                    IR.Float(scale_in_A): "scale_in_A",
                    IR.Float(scale_in_B): "scale_in_B",
                    IR.Float(scale_out): "scale_out",
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
                }) if not self.vbwEnabled else IR.FuncCall(funcName + c + ("<uint%d_t, uint%d_t, int%d_t, uint%d_t>" % (bitwidth_in_A, bitwidth_in_B, bitwidth_temp, bitwidth_out)), {
                    expr_in_A: "A",
                    expr_in_B: "B",
                    expr_out: "C",
                    IR.Int(I): "I",
                    IR.Int(J): "J",
                    IR.Float(scale_in_A): "scale_in_A",
                    IR.Float(scale_in_B): "scale_in_B",
                    IR.Float(scale_out): "scale_out",
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
                }) if not self.vbwEnabled else IR.FuncCall(funcName + "4" + ("<uint%d_t, uint%d_t, int%d_t, uint%d_t>" % (bitwidth_in_A, bitwidth_in_B, self.getTempBitwidth(bitwidth_in_A, bitwidth_in_B, "add", bitwidth_out), bitwidth_out)), {
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

            debugPrint = 0
            if type_out.dim == 2:
                debugPrint = IR.FuncCall("debugPrint", {
                    expr_out: "expr",
                    # expr_temp: "T",
                    IR.Int(I): "I",
                    IR.Int(J): "J",
                    IR.Float(scale_out): "scale",
                    IR.Int(zero_out): "zero",
                    IR.String(expr_out): "varName"
                })
            else:
                debugPrint = IR.FuncCall("debugPrint", {
                    expr_out: "expr",
                    # expr_temp: "T",
                    IR.Int(N): "N",
                    IR.Int(H): "H",
                    IR.Int(W): "W",
                    IR.Int(C): "C",
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
            #                 adjust = [IR.FuncCall("AdjustScaleShl" + (("<uint%d_t>"%bitwidth_out) if self.vbwEnabled else ""), {
            #                                     expr_out: "A",
            #                                     IR.Int(I): "I",
            #                                     IR.Int(J): "J",
            #                                     IR.Int(diff_scale): "scale"
            #                                     })]
            #             elif scale_out_unadjusted < scale_out:
            #                 diff_scale = 2 ** (scale_out - scale_out_unadjusted)
            #                 adjust = [IR.FuncCall("AdjustScaleShr" + (("<uint%d_t>"%bitwidth_out) if self.vbwEnabled else ""), {
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
            #                 adjust = [IR.FuncCall("AdjustScaleShl" + (("<uint%d_t>"%bitwidth_out) if self.vbwEnabled else ""), {
            #                                     expr_out: "A",
            #                                     IR.Int(N): "N",
            #                                     IR.Int(H): "H",
            #                                     IR.Int(W): "W",
            #                                     IR.Int(C): "C",
            #                                     IR.Int(diff_scale): "scale"
            #                                     })]
            #             elif scale_out_unadjusted < scale_out:
            #                 diff_scale = 2 ** (scale_out - scale_out_unadjusted)
            #                 adjust = [IR.FuncCall("AdjustScaleShr" + (("<uint%d_t>"%bitwidth_out) if self.vbwEnabled else ""), {
            #                                     expr_out: "A",
            #                                     IR.Int(N): "N",
            #                                     IR.Int(H): "H",
            #                                     IR.Int(W): "W",
            #                                     IR.Int(C): "C",
            #                                     IR.Int(diff_scale): "scale"
            #                                     })]
            # else:
            #     assert False, "Illegal number of dimensions"

            prog_bop = IR.Prog( [comment, funcCall] + ([debugPrint] if config.zeroSkewDebug else []))

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
            # q3  = (s1/s3)*(q1-z1) + (s2/s3)*(q2-z2)

            left_shift =  int(bitwidth_temp/2) # int((bitwidth_temp - bitwidth_in_A) - 1)
            # Make the input quantized to 32-bits. 

            s1_s3 = scale_in_A
            s2_s3 = scale_in_B
            s3_s3 = 1.0/scale_out
            m1, n1 = self.getQuantizedMultiplierLTO(s1_s3, bitwidth_temp, bitwidth_in_A)
            m2, n2 = self.getQuantizedMultiplierLTO(s2_s3, bitwidth_temp, bitwidth_in_B)
            m3, n3 = self.getQuantizedMultiplierLTO(s3_s3, bitwidth_temp, bitwidth_out)
            n1 -= ((31 if (bitwidth_temp == 32) else 63))
            n2 -= ((31 if (bitwidth_temp == 32) else 63))
            n3 -= ((31 if (bitwidth_temp == 32) else 63)- left_shift)

            if bitwidth_temp == 32:
                assert (n1 <= 31) and (n2 <= 31) and (n3 <= 31), "Right shift value too high"
            
            if bitwidth_temp == 64:
                assert (n1 <= 63) and (n2 <= 63) and (n3 <= 63), "Right shift value too high"

            return (left_shift, m1, n1, m2, n2, m3, n3)
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
        }) if not self.vbwEnabled else IR.FuncCall("ArgMax<uint%d_t>"%(bitwidth_in), {
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

    def visitSgn(self, node: AST.Func):
        (prog_in, expr_in) = self.visit(node.expr)

        expr_out = self.getTempVar()
        type_in = node.expr.type

        expr_in_idx = IRUtil.addIndex(expr_in, [IRUtil.zero] * type_in.dim)
        bitwidth_in, scale_in, zero_in = self.getBitwidthScaleZeros(expr_in.idf)

        expr_in_idx = IRUtil.addStrPrefixAndSuffix("ConvertZSkewToFloat<uint%d_t>("%(bitwidth_in), expr_in_idx, ",%d,%f)"%(zero_in, scale_in))

        comment = IR.Comment('sgn(' + expr_in.idf + ')', self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        cmd1 = IR.Assn(expr_out, IRUtil.cond_zero(
            expr_in_idx, IRUtil.one, IRUtil.zero))

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        prog_sgn = IR.Prog([comment, cmd1])

        prog_out = IRUtil.concatPrograms(prog_in, prog_sgn)

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
        bitwidth_temp = bitwidth_temp = self.getTempBitwidth(bitwidthA=bitwidth_in, op="tanh")

        intv_in = self.varIntervals[expr_in.idf]

        # If input is demoted to lower bit-width, demote the output to lower bit-width as well.
        tmp_var = expr_in.idf
        while tmp_var in self.substitutions.keys():
            tmp_var = self.substitutions[tmp_var]
        if tmp_var in self.demotedVarsList:
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
            IR.Float(scale_in): "scale_in",
            IR.Int(-zero_in): "zero_in",
            IR.Int(M1): "M1",
            IR.Int(-N1): "N1",
            IR.Float(scale_out): "scale_out",
            IR.Int(zero_out): "zero_out",
            IR.Int(M2): "M2",
            IR.Int(-N2): "N2",
            IR.Int(clamp_radius): "clamp_radius"
        }) if not self.vbwEnabled else IR.FuncCall("TanH<uint%d_t, int%d_t>"%(bitwidth_in, bitwidth_temp), {
            expr_in: "A",
            expr_out: "B",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Float(scale_in): "scale_in",
            IR.Int(-zero_in): "zero_in",
            IR.Int(M1): "M1",
            IR.Int(-N1): "N1",
            IR.Float(scale_out): "scale_out",
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

        prog_tanh = IR.Prog([comment, funcCall] + ([debugPrint] if config.zeroSkewDebug else []))

        prog_out = IRUtil.concatPrograms(prog_in, prog_tanh)

        # Updating metadata.
        self.varDeclarations[expr_out.idf] = type_in
        self.varIntervals[expr_out.idf] = intv_out
        self.varScales[expr_out.idf] = scale_out
        self.varZeros[expr_out.idf] = zero_out

        return (prog_out, expr_out)
    
    def getTanHShrAndN(self, scale_in, zero_in, scale_out, zero_out, intv_in, bitwidth_in, bitwidth_temp = None):
        bitwidth_temp = 32 if bitwidth_in == 8 else 64
        # assert config.wordLength == 8, "TanH not implemented for anything other than 8-bits"
        if bitwidth_temp == 32:
            M1, N1 = self.getQuantizedMultiplierLTO(scale_in, bitwidth_temp, bitwidth_in)
            N1 -= (31  + 18)
            M2, N2 = self.getQuantizedMultiplierLTO(1.0/scale_out, bitwidth_temp, bitwidth_in)
            N2 -= (31 - 7)
        elif bitwidth_temp == 64:
            M1, N1 = self.getQuantizedMultiplierLTO(scale_in, bitwidth_temp, bitwidth_in)
            N1 -= (63 + 18)
            M2, N2 = self.getQuantizedMultiplierLTO(1.0/scale_out, bitwidth_temp, bitwidth_in)
            N2 -= (63 - 7)
        else:
            assert False, "Only 8 and 16 bit operations supported"

        getLogger().debug("TanH fixed point scale in Zero Skew: " + str(scale_out))

        clamp_min, clamp_max = self.getClampValues(bitwidth_in)       
        return M1, N1, M2, N2, clamp_max/2
        
    def getSigmoidShrAndN(self, scale_in, zero_in, scale_out, zero_out, intv_in, bitwidth_in, bitwidth_temp = None):
        # if intv_in == (0,0):
        #     intv_in = (-config.maxVar8Bit, config.maxVar8Bit)
        #     assert (not config.vbwEnabled) and (config.wordLength == 8) 
        # sigmoid_min, sigmoid_max = intv_in
        # getLogger().debug("TanH internval in Zero Skew: " + str(sigmoid_min) + ", " + str(sigmoid_max))
        # assert config.wordLength == 8, "Sigmoid not implemented for anything other than 8-bits"
        # float_max = sigmoid_max * scale_in
        # scale_comp = self.getScale(float_max, bw=bitwidth_in)
        bitwidth_temp = 32 if bitwidth_in == 8 else 64
        getLogger().debug("Sigmoid fixed point scale in Zero Skew: " + str(scale_out))
        if bitwidth_temp == 32:
            M1, N1 = self.getQuantizedMultiplierLTO(scale_in, bitwidth_temp, bitwidth_in)
            N1 -= (31  + 18)
            M2, N2 = self.getQuantizedMultiplierLTO(1.0/scale_out, bitwidth_temp, bitwidth_in)
            N2 -= (31 - 8)
            
        elif bitwidth_temp == 64:
            M1, N1 = self.getQuantizedMultiplierLTO(scale_in, bitwidth_temp, bitwidth_in)
            N1 -= (63 + 18)
            M2, N2 = self.getQuantizedMultiplierLTO(1.0/scale_out, bitwidth_temp, bitwidth_in)
            N2 -= (63 - 8)
        else:
            assert False, "Only 8 and 16 bit operations supported"

        clamp_min, clamp_max = self.getClampValues(bitwidth_in)
        return M1, N1, M2, N2, clamp_max/2

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
        tmp_var = expr_in.idf
        while tmp_var in self.substitutions.keys():
            tmp_var = self.substitutions[tmp_var]
        if tmp_var in self.demotedVarsList:
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

        bitwidth_temp = self.getTempBitwidth(bitwidthA=bitwidth_in, op="sigmoid")

        funcCall = IR.FuncCall("Sigmoid", {
            expr_in: "A",
            expr_out: "B",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Float(scale_in): "scale_in",
            IR.Int(-zero_in): "zero_in",
            IR.Int(M1): "M1",
            IR.Int(-N1): "N1",
            IR.Float(scale_out): "scale_out",
            IR.Int(zero_out): "zero_out",
            IR.Int(M2): "M2",
            IR.Int(-N2): "N2",
            IR.Int(clamp_radius): "clamp_radius"
        }) if not self.vbwEnabled else IR.FuncCall("Sigmoid<uint%d_t,int%d_t>"%(bitwidth_in, bitwidth_temp), {
            expr_in: "A",
            expr_out: "B",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Float(scale_in): "scale_in",
            IR.Int(-zero_in): "zero_in",
            IR.Int(M1): "M1",
            IR.Int(-N1): "N1",
            IR.Float(scale_out): "scale_out",
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

        prog_sigmoid = IR.Prog([comment, funcCall] + ([debugPrint] if config.zeroSkewDebug else []))

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
        '''
        expr_out
        i = 0
        for (j = 0; j < n; j++)
          expr_in = prog_in
          expr_out = expr_out + expr_in
          i++

        1.  for i in [0, C]:
        2.    expr_out[i] = expr_out[i] + shr(expr_in[i])
        '''

        var_idf = node.name
        self.varDeclarations[var_idf] = Type.Int()
        self.internalVars.append(var_idf)

        start, end = node.start, node.end
        comment = IR.Comment("sum(i = [%d, %d])" % (start, end), self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth
        self.counter_inst += 1
        self.curDepth += 1

        (prog_in, expr_in) = self.visit(node.expr)

        self.curDepth -= 1

        expr_out = self.getTempVar()
        type_out = node.type

        var = IR.Var(var_idf)
        var_iter = self.getTempIterator()
        iters = self.getTempIterators(type_out.dim)

        # Read input scale and bitwidth.
        bitwidth_in, scale_in, zero_in = self.getBitwidthScaleZeros(expr_in.idf)
        # Read output scale and bitwidth if known from profiling.
        bitwidth_out, scale_out, zero_out = self.getBitwidthScaleZeros(expr_out.idf)
        
        intv_in = self.varIntervals[expr_in.idf]

        bitwidth_temp = self.getTempBitwidth(bitwidth_in, bitwidth_out, "add", bitwidth_out)
        # Read / Compute the output scale and associated hyperparameters.
        (left_shift, shrA, nA, shrB, nB, shrC, nC) = self.getScaleAndZeroForAddAndSub(scale_in, zero_in, scale_out, zero_out, scale_out, zero_out, bitwidth_in, bitwidth_out, bitwidth_temp, bitwidth_out, operator.add)
        intv_out = (0,0) 


        expr_in_idx = IRUtil.addIndex(expr_in, iters)
        expr_out_idx = IRUtil.addIndex(expr_out, iters)
        add_func = "AddInPlace" if not config.vbwEnabled else "AddInPlace<uint%d_t, int%d_t, uint%d_t>"%(bitwidth_in, bitwidth_temp, bitwidth_out)
        # Adjusting scale of input and output in the fixed-point code.
        cmd1 = self.memsetZeroSkew(expr_out, type_out.shape, scale_out, zero_out)
        clamp_min, clamp_max = self.getClampValues(bitwidth_out)
        
        cmd2 = IR.FuncCall(add_func, {
            expr_in:"expr_in",
            expr_out:"expr_out",
            IR.Int(type_out.shape[0]):"I",
            IR.Int(type_out.shape[1]): "J",
            IR.Float(scale_in): "scale_in",
            IR.Float(scale_out): "scale_out",
            IR.Int(zero_in): "zero_in",
            IR.Int(zero_out): "zero_out",
            IR.Int(left_shift): "left_shift",
            IR.Int(shrA): "shrA",
            IR.Int(-nA): "nA",
            IR.Int(shrB): "shrB",
            IR.Int(-nB): "nB",
            IR.Int(shrC): "shrC",
            IR.Int(-nC): "nC",
            IR.Int(clamp_min): "clamp_min",
            IR.Int(clamp_max): "clamp_max"
        })
        treeSum = IRUtil.loop(type_out.shape, iters, [cmd2])

        assert type_out.dim == 2, "Only 2 dim Summation supported for now due to laziness of programmer"

        # Final program to sum output of each iteration.
        prog_sum = cmd1 + [
                    IR.Assn(var, IR.Int(start)),
                    IR.For(var_iter, 0, IRUtil.lt(var_iter, IR.Int(end - start)),
                           prog_in.cmd_l + [cmd2]  + [IR.Assn(var, IRUtil.inc(var))])
                    ]

        self.updateLiveRange([expr_in, expr_out])

        prog_out = IR.Prog([comment] + prog_sum)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = type_out
        self.varScales[expr_out.idf] = scale_out
        self.varZeros[expr_out.idf] = zero_out
        self.varIntervals[expr_out.idf] = intv_out

        return (prog_out, expr_out)
    
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
        return (0, config.maxVar8Bit) if bitwidth_out == 8 else (0, config.maxVar16Bit)
    
    def visitBopSparseMul(self, node: AST.Bop1):
        (prog_in_A, expr_in_A) = self.visit(node.expr1)

        (prog_in_B, expr_in_B) = self.visit(node.expr2)

        [P, Q] = node.expr1.type.shape
        [Q, R] = node.expr2.type.shape

        assert R == 1, "Sparse matrix multiplication currently only support multiplication with a vector"

        tmp_out = self.getTempVar()
        expr_out = self.getTempVar()
        type_out = node.type

        # Reading input scales.
        bitwidth_in_A, scale_in_A, zero_in_A = self.getBitwidthScaleZeros(expr_in_A.idf)
        bitwidth_in_Aid = (config.wordLength // 2) if (expr_in_A.idf + 'idx') in self.demotedVarsList else config.wordLength
        bitwidth_in_B, scale_in_B, zero_in_B = self.getBitwidthScaleZeros(expr_in_B.idf)
        # Read output scales and bit-widths. In data-driven scaling, the output scale is directly profiled from floating-point runtime.
        # In static scaling used by old SeeDot (PLDI '19), output scale and bitwidth is set to None is statically computed later.
        bitwidth_out, scale_out, zero_out = self.getBitwidthScaleZeros(expr_out.idf)
        bitwidth_temp = self.getTempBitwidth(bitwidth_in_A, bitwidth_in_B, "sparse_mul", bitwidth_out)

        intv_in_A, intv_in_B = self.varIntervals[expr_in_A.idf], self.varIntervals[expr_in_B.idf]

        # Compute scaling hyperparameters given input and output scales. If static scaling of old SeeDot is used, also compute the output scale and bit-width.
        M0, N = self.getMatMulShrAndN(scale_in_A, scale_in_B, scale_out, zero_in_A, zero_in_B, zero_out, bitwidth_in_A, bitwidth_in_B, bitwidth_temp, bitwidth_out)
        (left_shift, shrA, nA, shrB, nB, shrC, nC) = self.getScaleAndZeroForAddAndSub(scale_out, zero_out, scale_out, zero_out, scale_out, zero_out, bitwidth_out, bitwidth_out, bitwidth_temp, bitwidth_out, operator.add)
        intv_out = (0, 0)

        in_A_idx = IR.Var(expr_in_A.idf +
                          'idx', expr_in_A.idx, inputVar=True)
        in_A_val = IR.Var(expr_in_A.idf +
                          'val', expr_in_A.idx, inputVar=True)


        in_A_idx.inputVar = False
        in_A_val.inputVar = False
        expr_in_B.inputVar = False
        expr_out.inputVar = False
        tmp_out.inputVar = False

        comment = IR.Comment(expr_in_A.idf + ' |*| ' + expr_in_B.idf, self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        # The output variable needs to be set to zero as the matrix multiplication implementation assumes this.
        cmd1 = self.memsetZeroSkew(expr_out, type_out.shape, scale_out, zero_out)
        cmd2 = self.memsetZeroSkew(tmp_out, type_out.shape, scale_out, 0)

        bitwidth_mul = bitwidth_temp
        self.varsForBitwidth[tmp_out.idf] = bitwidth_temp

        # For input variable 'X', the data is streamed on the target device, which necessitates a different function implementation.
        if expr_in_B.idf == 'X':
            funcName = "SparseMatMulX"
        else:
            funcName = "SparseMatMul"
        
        clamp_min, clamp_max = self.getClampValues(bitwidth_out)

        funcCall = IR.FuncCall(funcName, {
            in_A_idx: "Aidx",
            in_A_val: "Aval",
            expr_in_B: "B",
            expr_out: "C",
            tmp_out: "tmp",
            IR.Int(P): "P",
            IR.Int(Q): "K",
            IR.Float(scale_in_A): "scaleA",
            IR.Float(scale_in_B): "scaleB",
            IR.Float(scale_out): "scale_out",
            IR.Int(left_shift): "left_shift",
            IR.Int(-zero_in_A): "zeroA",
            IR.Int(-zero_in_B): "zeroB",
            IR.Int(zero_out): "zeroC",
            IR.Int(M0): "M0",
            IR.Int(-N): "N",
            IR.Int(clamp_min): "clamp_min",
            IR.Int(clamp_max): "clamp_max"
        }) if not self.vbwEnabled else IR.FuncCall("%s<uint%d_t, uint%d_t, uint%d_t, int%d_t, uint%d_t>"%(funcName, bitwidth_in_A, bitwidth_in_Aid, bitwidth_in_B, bitwidth_mul, bitwidth_out), {
            in_A_idx: "Aidx",
            in_A_val: "Aval",
            expr_in_B: "B",
            expr_out: "C",
            tmp_out: "tmp",
            IR.Int(P): "P",
            IR.Int(Q): "K",
            IR.Float(scale_in_A): "scaleA",
            IR.Float(scale_in_B): "scaleB",
            IR.Float(scale_out): "scale_out",
            IR.Int(left_shift): "left_shift",
            IR.Int(-zero_in_A): "zeroA",
            IR.Int(-zero_in_B): "zeroB",
            IR.Int(zero_out): "zeroC",
            IR.Int(M0): "M0",
            IR.Int(-N): "N",
            IR.Int(clamp_min): "clamp_min",
            IR.Int(clamp_max): "clamp_max"
        })

        debugPrint = []
        if config.zeroSkewDebug:
            debugPrint.append(IR.FuncCall("debugPrint", {
                expr_out: "expr",
                IR.Int(P): "I",
                IR.Int(R): "J",
                IR.Float(scale_out): "scale",
                IR.Int(zero_out): "zero",
                IR.String(expr_out): "varName"
            }))

        self.counter_inst += 1
        self.updateLiveRange([in_A_idx, in_A_val, expr_in_B, expr_out, tmp_out])

        # Profiling the floating-point output for scale computation for the fixed-point code (only used if ddsEnabled is True).
        
        if forFloat():
            self.independentVars.append(expr_out.idf)

        prog_mul = IR.Prog([comment] + cmd1 + [funcCall] + debugPrint)

        prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_mul)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = type_out
        self.varScales[expr_out.idf] = scale_out
        self.varZeros[expr_out.idf] = zero_out
        self.varIntervals[expr_out.idf] = intv_out
        self.varDeclarations[tmp_out.idf] = type_out
        self.varScales[tmp_out.idf] = scale_out
        self.varZeros[tmp_out.idf] = 0
        self.varIntervals[tmp_out.idf] = intv_out

        # Update metadata for sparse matrix.
        self.varDeclarations.update({in_A_idx.idf: Type.Tensor([self.sparseMatrixSizes[expr_in_A.idf + 'idx']]),
                                     in_A_val.idf: Type.Tensor([self.sparseMatrixSizes[expr_in_A.idf + 'val']]),
                                     })

        # Include sparse matrices in global variables.
        if in_A_idx.idf not in self.globalVars:
            self.globalVars.append(in_A_idx.idf)
        if in_A_val.idf not in self.globalVars:
            self.globalVars.append(in_A_val.idf)

        # Print log.
        self.log.print(comment.msg)
        self.log.print("\tInput1: scale = %f, zero = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in_A.idf],) + (self.varZeros[expr_in_A.idf],) + self.varIntervals[expr_in_A.idf]))
        self.log.print("\tInput2: scale = %f, zero = %d, interval = [%d, %d]" % (
            (self.varScales[expr_in_B.idf],) + (self.varZeros[expr_in_B.idf],) + self.varIntervals[expr_in_B.idf]))
        self.log.print("\tOutput: scale = %f, zero = %d, interval = [%d, %d]" % (
            (self.varScales[expr_out.idf],) + (self.varZeros[expr_out.idf],) + self.varIntervals[expr_out.idf]))

        return (prog_out, expr_out)
    
    # out = +- in
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
            
            bitwidth_in, scale_in, zero_in = self.getBitwidthScaleZeros(expr_in.idf)

            m, M = self.getIntervalFromScaleZero(bitwidth_in, scale_in, zero_in)
            bitwidth_out = bitwidth_in
            m, M = -M, -m
            scale_out, zero_out = self.getScaleAndZero(m, M, bw=bitwidth_out)

            # If the input variable is demoted to lower bit-width, demote the output as well as no extra information can be stored in the extra bits.
            self.varsForBitwidth[expr_out.idf] = bitwidth_out
            if bitwidth_out != config.wordLength:
                self.demotedVarsList.append(expr_out.idf)

            intv_out = (m, M)
            bitwidth_temp = self.getTempBitwidth(bitwidth_in, bitwidth_in, op="mul", bitwidthC=bitwidth_out)
            
            clamp_min, clamp_max = self.getClampValues(bitwidth_out)
            funcNameA = "UnaryNegate<uint%d_t, int%d_t>("%(bitwidth_in, bitwidth_temp)
            expr = IRUtil.addStrPrefixAndSuffix(funcNameA, IRUtil.addIndex(expr_in, iters), 
                                    ", %d, %d, %d, %d)"%(-zero_in, zero_out, \
                                        clamp_min, clamp_max), bitwidth_in)
            lhs = IRUtil.addIndex(expr_out, iters)
            rhs = expr
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
    
    # out = in_cond > 0? in_A: in_B
    def visitCond(self, node: AST.Cond):
        (prog_in_cond, expr_in_cond) = self.visit(node.expr)

        (prog_in_A, expr_in_A) = self.visit(node.trueBlock)

        (prog_in_B, expr_in_B) = self.visit(node.falseBlock)

        type_in_cond = node.expr.type
        type_in_A = node.trueBlock.type

        if Type.isInt(type_in_cond):
            expr_in_cond_idx = expr_in_cond
        else:
            expr_in_cond_idx = IRUtil.addIndex(
                expr_in_cond, [IRUtil.zero] * type_in_cond.dim)
            bitwidth_in_cond_idx, scale_in_cond_idx, zero_in_cond_idx = self.getBitwidthScaleZeros(expr_in_cond_idx.idf)
            expr_in_cond_idx = IRUtil.addStrPrefixAndSuffix("ConvertZSkewToFloat<uint%d_t>("%(bitwidth_in_cond_idx), expr_in_cond_idx, ", %d, %f)"%(zero_in_cond_idx, scale_in_cond_idx), bitwidth_in_cond_idx)

        # e2, e3 : Int
        if Type.isInt(type_in_A):
            # TODO: Update the scale and intv of expr_out based on in_A and in_B.
            prog_out = IRUtil.concatPrograms(
                prog_in_cond, prog_in_A, prog_in_B)
            expr_out = IRUtil.cond_zero(expr_in_cond_idx, expr_in_A, expr_in_B)

            if isinstance(expr_in_A, IR.Var):
                assert expr_in_A.idf not in self.varScales and expr_in_A.idf not in self.varZeros and expr_in_A.idf not in self.varIntervals
            if isinstance(expr_in_B, IR.Var):
                assert expr_in_B.idf not in self.varScales and expr_in_B.idf not in self.varZeros and expr_in_B.idf not in self.varIntervals
        # e2, e3 : Tensor(), or Tensor(..)
        else:
            expr_out = self.getTempVar()
            iters = self.getTempIterators(type_in_A.dim)

            # Read input scales and bit-widths.
            bitwidth_in_A, scale_in_A, zero_in_A = self.getBitwidthScaleZeros(expr_in_A.idf)
            bitwidth_in_B, scale_in_B, zero_in_B = self.getBitwidthScaleZeros(expr_in_B.idf)
            intv_in_A, intv_in_B = self.varIntervals[expr_in_A.idf], self.varIntervals[expr_in_B.idf]

            bitwidth_out = max(bitwidth_in_A, bitwidth_in_B)

            m_A, M_A = self.getIntervalFromScaleZero(bitwidth_in_A, scale_in_A, zero_in_A)
            m_B, M_B = self.getIntervalFromScaleZero(bitwidth_in_B, scale_in_B, zero_in_B)

            m_out, M_out = min(m_A, m_B), max(M_A, M_B)
            scale_out, zero_out = self.getScaleAndZero(m_out, M_out, bw=bitwidth_out)

            scale_out = max(scale_in_A, scale_in_B)
            
            # prog_assn
            expr_in_A_idx = IRUtil.addIndex(expr_in_A, iters)
            expr_in_B_idx = IRUtil.addIndex(expr_in_B, iters)
            expr_out_idx = IRUtil.addIndex(expr_out, iters)

            bitwidth_temp = self.getTempBitwidth(bitwidth_in_A, bitwidth_in_A, op="mul")
            m1, n1 = self.getMatMulShrAndN(scale_in_A, 1.0, scale_out, zero_in_A, 0, 
                    zero_out, bitwidth_in_A, bitwidth_in_A, bitwidth_temp, bitwidth_out)
            funcNameA = "AdjustScaleZero(" if not config.vbwEnabled else \
                        "AdjustScaleZero<uint%d_t, int%d_t, uint%d_t>("%(bitwidth_in_A, bitwidth_temp, bitwidth_out)
            
            bitwidth_temp = self.getTempBitwidth(bitwidth_in_B, bitwidth_in_B, op="mul")
            m2, n2 = self.getMatMulShrAndN(scale_in_B, 1.0, scale_out, zero_in_B, 0, 
                    zero_out, bitwidth_in_B, bitwidth_in_B, bitwidth_temp, bitwidth_out)
            funcNameB = "AdjustScaleZero(" if not config.vbwEnabled else \
                        "AdjustScaleZero<uint%d_t, int%d_t, uint%d_t>("%(bitwidth_in_B, bitwidth_temp, bitwidth_out)

            clamp_min, clamp_max = self.getClampValues(bitwidth_out)
            true_expr = IRUtil.addStrPrefixAndSuffix(funcNameA, expr_in_A_idx, 
                                    ", %d, %d, %d, %d, %d, %d)"%(zero_in_A, zero_out, \
                                        m1, -n1, clamp_min, clamp_max), bitwidth_in_A)
            false_expr = IRUtil.addStrPrefixAndSuffix(funcNameB, expr_in_B_idx, 
                                    ", %d, %d, %d, %d, %d, %d)"%(zero_in_B, zero_out, \
                                        m2, -n2, clamp_min, clamp_max), bitwidth_in_B)
            
            rhs = IRUtil.cond_zero(expr_in_cond_idx,
                                   true_expr,
                                   false_expr)
            cmdl_assn = IRUtil.loop(type_in_A.shape, iters, [
                                    IR.Assn(expr_out_idx, rhs)])
            prog_cond = IR.Prog(cmdl_assn)

            prog_out = IRUtil.concatPrograms(
                prog_in_cond, prog_in_A, prog_in_B, prog_cond)

            # Update metadata.
            self.varDeclarations[expr_out.idf] = type_in_A
            self.varScales[expr_out.idf] = scale_out
            self.varZeros[expr_out.idf] = zero_out
            self.varIntervals[expr_out.idf] = [m_out, M_out]

        self.allDepths[self.counter_inst+1] = self.curDepth
        self.counter_inst += 1
        self.updateLiveRange([expr_in_A, expr_in_B, expr_in_cond, expr_out])

        return (prog_out, expr_out)
    
    def getIntervalFromScaleZero(self, bitwidth, scale, zero):
        if bitwidth == 8:
            max = config.maxVar8Bit
        elif bitwidth == 16:
            max = config.maxVar16Bit
        else:
            assert False, "Unsupported bitwidth of variable: %d"%(bitwidth)
        
        m = -zero * scale
        M = scale*(max - zero)
        return [m, M]
    
    def visitExp(self, node: AST.Func):
        # Used in the old SeeDot (PLDI '19) version.
        # Tunable parameter.

        (prog_in, expr_in) = self.visit(node.expr)

        type_in = node.expr.type

        # Reading input scale and bit-width.
        bitwidth_in, scale_in, zero_in = self.getBitwidthScaleZeros(expr_in.idf)

        '''
        1.  y = ((int) (exp(((float)e) / shr1) * shr2))
        '''

        expr_out = self.getTempVar()

        # Reading / Computing output bit-width.
        bitwidth_out, scale_out, zero_out = self.getBitwidthScaleZeros(expr_out.idf)

        [I, J] = type_in.shape
        bitwidth_temp = self.getTempBitwidth(bitwidth_in, op="exp", bitwidthC=bitwidth_out)
        left_shift, shrA, nA, shrB1, nB1, shrB2, nB2 = self.getExpScaleAndZeros(scale_in, zero_in, scale_out, zero_out, bitwidth_in, bitwidth_temp, bitwidth_out)
        clamp_min, clamp_max = self.getClampValues(bitwidth_out) 
        cmd0 = IR.Comment('exp(' + expr_in.idf + ')', self.counter_inst+1)
        self.allDepths[self.counter_inst+1] = self.curDepth

        funcCall = IR.FuncCall("Exp", {
            expr_in: "A",
            expr_out: "B",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Int(left_shift): "left_shift",
            IR.Int(-zero_in): "zero_in",
            IR.Int(zero_out): "zero_out",
            IR.Int(shrA): "shrA",
            IR.Int(-nA): "nA",
            IR.Int(shrB1): "shrB",
            IR.Int(-nB1): "nB",
            IR.Int(shrB2): "shrC",
            IR.Int(-nB2): "nC",
            IR.Int(clamp_min): "clamp_min",
            IR.Int(clamp_max): "clamp_max"
        }) if not self.vbwEnabled else IR.FuncCall("Exp<uint%d_t, int%d_t, uint%d_t>"%(bitwidth_in, bitwidth_temp, bitwidth_out), {
            expr_in: "A",
            expr_out: "B",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.Float(scale_in): "scale_in",
            IR.Float(scale_out): "scale_out",
            IR.Int(left_shift): "left_shift",
            IR.Int(-zero_in): "zero_in",
            IR.Int(zero_out): "zero_out",
            IR.Int(shrA): "shrA",
            IR.Int(-nA): "nA",
            IR.Int(shrB1): "shrB",
            IR.Int(-nB1): "nB",
            IR.Int(shrB2): "shrC",
            IR.Int(-nB2): "nC",
            IR.Int(clamp_min): "clamp_min",
            IR.Int(clamp_max): "clamp_max"
        })

        self.counter_inst += 1
        self.updateLiveRange([expr_in, expr_out])

        # This method is used in the profiling floating point stage to check whether the input values are beyond a threshold.
        # Input values beyond a threshold are always mapped to zero in fixed-point code, hence these datapoints hold little use in the fixed-point mode.
        rangeCheck = IR.FuncCall("checkRange2", {
            expr_in: "A",
            IR.Int(I): "I",
            IR.Int(J): "J"
        })  if self.functionReducedProfiling and forFloat() else IR.Comment("Recommend switching on Function Reduced Profiling for sound output")

        profile = IR.FuncCall("Profile2", {
            expr_out: "Var",
            IR.Int(I): "I",
            IR.Int(J): "J",
            IR.String(expr_out): "VarName"
        })
        if forFloat():
            self.independentVars.append(expr_out.idf)
        
        debugPrint = []
        if config.zeroSkewDebug:
            debugPrint.append(IR.FuncCall("debugPrint", {
                expr_out: "expr",
                IR.Int(I): "I",
                IR.Int(J): "J",
                IR.Float(scale_out): "scale",
                IR.Int(zero_out): "zero",
                IR.String(expr_out): "varName"
            }))

        prog_exp = IR.Prog([cmd0, funcCall] + debugPrint)

        prog_out = IRUtil.concatPrograms(prog_in, prog_exp)

        # Update metadata.
        self.varDeclarations[expr_out.idf] = type_in
        self.varScales[expr_out.idf] = scale_out
        self.varZeros[expr_out.idf] = zero_out
        self.varIntervals[expr_out.idf] = (0,0)

        return (prog_out, expr_out)
    
    def getExpScaleAndZeros(self, scale_in, zero_in, scale_out, zero_out, bitwidth_in, bitwidth_temp, bitwidth_out):
        left_shift = 30 if bitwidth_temp == 32 else 62
        
        if bitwidth_temp == 32:
            shrA, nA = self.getQuantizedMultiplierLTO(scale_in, bitwidth_temp, bitwidth_in)
            nA -= (31  + 18)
            shrB1, nB1 = self.getQuantizedMultiplierLTO(1.0/scale_out, bitwidth_temp, bitwidth_in)
            nB2 = nB1 - (31 - left_shift + 7)
            nB1 -= (31 - 7)
            shrB2 = shrB1
        elif bitwidth_temp == 64:
            shrA, nA = self.getQuantizedMultiplierLTO(scale_in, bitwidth_temp, bitwidth_in)
            nA -= (63 + 18)
            shrB1, nB1 = self.getQuantizedMultiplierLTO(1.0/scale_out, bitwidth_temp, bitwidth_in)
            nB2 = nB1 - (63 - left_shift + 7)
            nB1 -= (63 - 7)
            shrB2 = shrB1
        else:
            assert False, "Only 8 and 16 bit operations supported"
        return left_shift, shrA, nA, shrB1, nB1, shrB2, nB2
    
    def memsetZeroSkew(self, expr, shape, scale, zero):
        iters_in = self.getTempIterators(len(shape))

        loopShape = []  # Contains the shape of the tensor being initialized.
        loopIters = []  # Iterators which will be used to iterate to each tensor element.

        for order in range(len(shape)):
            loopShape.append(shape[order])
            loopIters.append(iters_in[order])
        loop = IRUtil.loop(loopShape, loopIters, [
            IR.Assn(IRUtil.addIndex(expr, iters_in), 
            IR.Int(zero))
        ])
        return loop
