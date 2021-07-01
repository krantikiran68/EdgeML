# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''
Code to generate the x86 compatible Prediction code. 
'''

import numpy as np
import os

from numpy.core.numeric import _convolve_dispatcher
from numpy.lib.type_check import typename

from seedot.compiler.codegen.x86 import X86

import seedot.compiler.ir.ir as IR
import seedot.compiler.ir.irUtil as IRUtil

from functools import reduce

import seedot.compiler.type as Type
from seedot.util import *
from seedot.writer import Writer
from bokeh.plotting import figure, output_file, show

import time


class X86Posit(X86):

    def __init__(self, outputDir, generateAllFiles, printSwitch, idStr, paramInNativeBitwidth, decls, localDecls, scales, intvs, cnsts, expTables, globalVars, internalVars, floatConstants, substitutions, demotedVarsOffsets, varsForBitwidth, varLiveIntervals, notScratch, coLocatedVariables):
        super().__init__(outputDir, generateAllFiles, printSwitch, idStr, paramInNativeBitwidth, decls, localDecls, scales, intvs, cnsts, expTables, globalVars, internalVars, floatConstants, substitutions, demotedVarsOffsets, varsForBitwidth, varLiveIntervals, notScratch, coLocatedVariables)
    
    def storeFlashSize(self):
        size_full = 0
        bw = config.positBitwidth
        for var in self.globalVars:
            size = self.decls[var].shape
            size_ = 1
            size_ = reduce(lambda x, y: x*y , size)
            size_full += size_
        f = open("flashsize.txt", "w")
        f.write(str((size_full * bw)//8))
        f.close()

    def printPrefix(self):
        if self.generateAllFiles:
            self.printCincludes()

            self.printExpTables()

        self.printCHeader()
        self.storeFlashSize()
        self.computeScratchLocationsFirstFitPriority() # computeScratchLocations computeScratchLocationsFirstFit computeScratchLocationsFirstFitPriority computeScratchLocationsDLX

        self.printModelParamsWithBitwidth()

        self.printVarDecls(globalVarDecl=False)

        self.printConstDecls()

        self.out.printf('\n')

    def printCincludes(self):
        self.out.printf('#include <iostream>\n', indent=True)
        self.out.printf('#include <cstring>\n', indent=True)
        self.out.printf('#include <cmath>\n\n', indent=True)
        self.out.printf('#include "datatypes.h"\n', indent=True)
        self.out.printf('#include "predictors.h"\n', indent=True)
        self.out.printf('#include "profile.h"\n', indent=True)
        self.out.printf('#include "library_%s.h"\n' %
                        (getEncoding()), indent=True)
        self.out.printf('#include "model_%s.h"\n' %
                        (getEncoding()), indent=True)
        self.out.printf('#include "vars_%s.h"\n\n' %
                        (getEncoding()), indent=True)
        self.out.printf('using namespace std;\n', indent=True)
        self.out.printf('using namespace seedot_%s;\n' %
                        (getEncoding()), indent=True)

    def printExpTables(self):
        for exp, [table, [tableVarA, tableVarB]] in self.expTables.items():
            self.printExpTable(table[0], tableVarA)
            self.printExpTable(table[1], tableVarB)
            self.out.printf('\n')

    def printExpTable(self, table_row, var):
        self.out.printf('const MYINT %s[%d] = {\n' % (
            var.idf, len(table_row)), indent=True)
        self.out.increaseIndent()
        self.out.printf('', indent=True)
        for i in range(len(table_row)):
            self.out.printf('%d, ' % table_row[i])
        self.out.decreaseIndent()
        self.out.printf('\n};\n')

    def printCHeader(self):
        
        self.out.printf('void seedotPosit%s(float **X_temp, float* res) {\n' % (self.idStr if not self.generateAllFiles else ""), indent=True)
        self.out.increaseIndent()

    def printModelParamsWithBitwidth(self):
        for var in self.globalVars:
            if var + "idx" in self.globalVars and var + "val" in self.globalVars:
                continue
            if var[-3:] == 'idx' and not config.vbwEnabled:
                continue
            bw = self.varsForBitwidth[var] if config.vbwEnabled else config.positBitwidth
            if var[-3:] == 'idx':
                typ_str = "int%d_t"%bw
            else:
                typ_str = self.getPositType(bw)
            bitwidth_PX2_suffix = self.getPX2Suffix(bw)
            conversion_func = self.getConversionFunction(bw)

            size = self.decls[var].shape
            sizestr = ''.join(["[%d]" % (i) for i in size])

            Xindexstr = ''
            Xintstar = ''.join(["*" for i in size])

            if self.paramInNativeBitwidth or var == 'X':
                if var != 'X':
                    self.out.printf(typ_str + " " + var + sizestr + ";\n", indent = True)
                else:
                    self.out.printf(typ_str + Xintstar + " " + var + ";\n", indent = True)

                for i in range(len(size)):
                    Xindexstr += ("[i" + str(i-1) + "]" if i > 0 else "")
                    if var == 'X':
                        Xintstar = Xintstar[:-1]
                        self.out.printf("X%s = new %s%s[%d];\n" % (Xindexstr, typ_str, Xintstar, size[i]), indent=True)
                    self.out.printf("for (int i%d = 0; i%d < %d; i%d ++) {\n" % (i,i,size[i], i), indent = True)
                    self.out.increaseIndent()

                indexstr = ''.join("[i" + str(i) + "]" for i in range(len(size)))
                if var[-3:] == 'idx':
                    self.out.printf(var + indexstr + " = " + var + "_temp" + indexstr + ";\n", indent = True)
                else:
                    self.out.printf(var + indexstr + " = " + conversion_func + "(" + var + "_temp" + indexstr + bitwidth_PX2_suffix +  ")" + ";\n", indent = True)

                for i in range(len(size)):
                    self.out.decreaseIndent()
                    self.out.printf("}\n", indent = True)

    def printVarDecls(self, globalVarDecl=True):
        if self.generateAllFiles:
            varsFilePath = os.path.join(
                self.outputDir, "vars_" + getEncoding() + ".h")
            varsFile = Writer(varsFilePath)

            varsFile.printf("#pragma once\n\n")
            varsFile.printf("#include \"datatypes.h\"\n\n")
            varsFile.printf("namespace vars_%s {\n" % (getEncoding()))
            varsFile.increaseIndent()

        for decl in self.decls:
            if decl in self.globalVars:
                continue

            if decl not in self.internalVars:
                if config.vbwEnabled and decl not in self.internalVars:
                    bw = self.varsForBitwidth.get(decl, config.wordLength)
                    typ_str = self.getPositType(bw)
                else:
                    typ_str = self.getPositType(config.positBitwidth)
            else:
                typ_str = "int"

            idf_str = decl
            type = self.decls[decl]
            if Type.isInt(type):
                shape_str = ''
            elif Type.isTensor(type):
                shape_str = ''.join(['[' + str(n) + ']' for n in type.shape])

            if not config.vbwEnabled:
                self.out.printf('%s %s%s;\n', typ_str, idf_str, shape_str, indent=True)
                if self.generateAllFiles:
                    varsFile.printf('extern %s %s%s;\n', typ_str,
                                idf_str, shape_str, indent=True)
            else:
                if idf_str in self.varsForBitwidth and idf_str[:3] == "tmp":
                    if globalVarDecl:
                        for bw in config.availableBitwidths:
                            self.out.printf("%s vars_%s::%s_%d%s;\n", self.getPositType(bw), getEncoding(), idf_str, bw, shape_str, indent=True)
                    else:
                        self.out.printf("%s %s_%d%s;\n", self.getPositType(self.varsForBitwidth[idf_str]), idf_str, bw, shape_str, indent=True)
                else:
                    if globalVarDecl:
                        self.out.printf("%s vars_%s::%s%s;\n", typ_str, getEncoding(), idf_str, shape_str, indent=True)
                    else:
                        self.out.printf("%s %s%s;\n", typ_str, idf_str, shape_str, indent=True)

                if self.generateAllFiles:
                    if forFixed() and idf_str in self.varsForBitwidth and idf_str[:3] == "tmp":
                        for bw in config.availableBitwidths:
                            varsFile.printf("extern %s %s_%d%s;\n", self.getPositType(bw), idf_str, bw, shape_str, indent=True)
                    else:
                        varsFile.printf("extern %s %s%s;\n", typ_str, idf_str, shape_str, indent=True)

        self.out.printf('\n')
        if self.generateAllFiles:
            varsFile.decreaseIndent()
            varsFile.printf("}\n")
            varsFile.close()

        self.generateDebugProgram()

    def generateDebugProgram(self):
        if not self.generateAllFiles:
            return
        debugFilePath = os.path.join(self.outputDir, "debug.cpp")
        debugFile = Writer(debugFilePath)

        debugFile.printf("#include <iostream>\n\n")
        debugFile.printf("#include \"datatypes.h\"\n")
        debugFile.printf("#include \"profile.h\"\n")
        debugFile.printf("#include \"vars_fixed.h\"\n")
        debugFile.printf("#include \"vars_float.h\"\n\n")
        debugFile.printf("#include \"vars_posit.h\"\n\n")
        debugFile.printf("using namespace std;\n\n")
        debugFile.printf("void debug() {\n\n")

        if debugMode() and forFixed():
            debugFile.increaseIndent()

            for decl in self.decls:
                if decl in self.globalVars:
                    continue

                type = self.decls[decl]
                if decl not in self.scales or not isinstance(type, Type.Tensor) or type.isShapeOne():
                    continue

                scale = self.scales[decl]

                s = decl + "[0]" * type.dim
                shape_str = ''.join([str(n) + ', ' for n in type.shape])
                shape_str = shape_str.rstrip(', ')

                debugFile.printf("diff(&vars_float::%s, &vars_fixed::%s, %d, %s);\n\n" % (
                    s, s, scale, shape_str), indent=True)

            debugFile.decreaseIndent()

        debugFile.printf("}\n")
        debugFile.close()

    def printSuffix(self, expr: IR.Expr):
        self.out.printf('\n')

        if config.vbwEnabled:
            bw = self.varsForBitwidth['X']
            typ_str = "int%d_t" % bw
            size = self.decls['X'].shape
            sizestr = ''.join([("[%d]" % i) for i in size])

            Xindexstr = ''
            Xintstar = ''.join(["*" for i in size])

            for i in range(len(size)):
                Xindexstr += (("[i%d]" % (i-1)) if i > 0 else "")
                self.out.printf("for (int i%d = 0; i%d < %d; i%d ++ ){\n" % (i,i,size[i],i), indent=True)
                self.out.increaseIndent()

            for i in range(len(size)-1, -1, -1):
                self.out.decreaseIndent()
                self.out.printf("}\n", indent=True)
                self.out.printf("delete[] X%s;\n" % (Xindexstr), indent=True)
                Xindexstr = Xindexstr[:-4] if len(Xindexstr) > 0 else Xindexstr
                assert len(size) < 10, "Too simple logic for printing indices used, cannot handle 10+ Dim Tensors"

        type = self.decls[expr.idf]

        if Type.isInt(type) or (Type.isTensor(type) and type.dim == 0):
            self.out.printf('res[0] = ', indent = True)
            self.print(expr)
            self.out.printf(';\n')
        elif Type.isTensor(type):
            idfr = expr.idf
            iters = []
            resIndex = ''
            remSize = np.prod(type.shape)
            for i in range(type.dim):
                s = chr(ord('i') + i)
                remSize = remSize // type.shape[i]
                resIndex += str(s) + '*' + str(remSize) + '+'
                tempVar = IR.Var(s)
                iters.append(tempVar)
            resIndex = resIndex[:-1]
            expr_1 = IRUtil.addIndex(expr, iters)
            cmds = IRUtil.loop(type.shape, iters, [
                IR.Assn(IRUtil.addIndex(IR.Var('res'), [IR.Var(resIndex)]), IRUtil.addStrPrefixAndSuffix("convertPositToDouble(", IRUtil.addIndex(expr, iters), ")", self.varsForBitwidth[expr.idf]))
            ])
            self.print(IR.Prog(cmds))
        else:
            assert False, "Illegal type of program output"

        self.out.decreaseIndent()
        self.out.printf('}\n', indent=True)

        def isInt(a):
            try:
                int(a)
                return True
            except:
                return False

        if (int(self.printSwitch) if isInt(self.printSwitch) else -2) > -1:
            self.out.printf("const int positSwitches = %d;\n" % (int(self.printSwitch)), indent = True)
            self.out.printf('void seedotPositSwitch(int i, float **X_temp, float* res) {\n', indent=True)
            self.out.increaseIndent()
            self.out.printf('switch(i) {\n', indent = True)
            self.out.increaseIndent()
            for i in range(int(self.printSwitch)):
                self.out.printf('case %d: seedotPosit%d(X_temp, res); return;\n' % (i,i+1), indent = True)
            self.out.printf('default: res[0] = -1; return;\n', indent = True)
            self.out.decreaseIndent()
            self.out.printf('}\n', indent=True)
            self.out.decreaseIndent()
            self.out.printf('}\n', indent=True)

        getLogger().debug("Closing file after outputting cpp code: ID " + self.idStr)
        self.out.close()

    def printFor(self, ir):
        super().printFor(ir)
        
    def printFuncCall(self, ir):
        if not Config.x86MemoryOptimize or forFloat():
            self.printFuncCallOrig(ir)
        else:
            self.out.printf("{\n", indent=True)
            self.out.increaseIndent()
            self.printLocalVarDecls(ir)
            self.out.printf("%s(" % (ir.name), indent=True)
            keys = list(ir.argList)
            if config.positBitwidth not in [8, 16, 32]:
                keys.append(IR.Int(config.positBitwidth))
        
            for i in range(len(keys)):
                arg = keys[i]
                if isinstance(arg, IR.Var) and (arg.idf in self.decls.keys() or arg.idf in self.localDecls.keys()) and not arg.idf == 'X':
                    type = self.decls[arg.idf] if arg.idf in self.decls else self.localDecls[arg.idf]
                    if isinstance(type, Type.Tensor):
                        if type.dim == 0:
                            x = -1
                        else:
                            x = type.dim - len(arg.idx)
                    else:
                        x = -1
                else:
                    x = 0
                if isinstance(arg, IR.Var) and arg.idf[-3:] == 'idx': 
                    typeCast = "(int%d_t*)" % self.varsForBitwidth[arg.idf] if x > 0 else ""
                else:
                    typeCast = "(%s*)" % self.getPositType(self.varsForBitwidth[arg.idf]) if x > 0 else ""
                self.out.printf(typeCast)

                if self.currentMemMap not in self.scratchSubs or not (isinstance(arg, IR.Var) and arg.idf in self.scratchSubs[self.currentMemMap]):
                    if x != 0:
                        self.out.printf("&")
                    self.print(arg)

                    if x != 0 and x != -1:
                        self.out.printf("[0]" * x)
                else:
                    self.printVar(arg, isPointer=True)
                if i != len(keys) - 1:
                    self.out.printf(", ")

            self.out.printf(");\n")
            self.out.decreaseIndent()
            self.out.printf("}\n", indent=True)

    def getPositType(self, bw):
        if config.positBitwidth != 16:
            bw = config.positBitwidth
        if config.useUnverified:
            return "posit_2_t"
        if bw == 8:
            return "posit8_t"
        if bw == 16:
            return "posit16_t"
        if bw == 32:
            return "posit32_t"
        return "posit_2_t"
    
    def getConversionFunction(self, bw):
        if config.positBitwidth != 16:
            bw = config.positBitwidth
        if config.useUnverified:
            return "convertDoubleToPX2"
        if bw == 8:
            return "convertDoubleToP8"
        if bw == 16:
            return "convertDoubleToP16"
        if bw == 32:
            return "convertDoubleToP32"
        return "convertDoubleToPX2"

    def printFuncCallOrig(self, ir):
        self.out.printf("{\n", indent=True)
        self.out.increaseIndent()
        self.printLocalVarDecls(ir)
        self.out.printf("%s(" % self.processFuncName(ir.name), indent=True)
        keys = list(ir.argList)
        if config.positBitwidth not in [8, 16, 32]:
            keys.append(IR.Int(config.positBitwidth))
        
        for i in range(len(keys)):
            arg = keys[i]
            if isinstance(arg, IR.Var) and (arg.idf in self.decls.keys() or arg.idf in self.localDecls.keys()) and not arg.idf == 'X':
                type = self.decls[arg.idf] if arg.idf in self.decls else self.localDecls[arg.idf]
                if isinstance(type, Type.Tensor):
                    if type.dim == 0:
                        x = -1
                    else:
                        x = type.dim - len(arg.idx)
                else:
                    x = -1
            else:
                x = 0
            if x != 0:
                self.out.printf("&")
            self.print(arg)
            if x != 0 and x != -1:
                self.out.printf("[0]" * x)
            if i != len(keys) - 1:
                self.out.printf(", ")
        self.out.printf(");\n")
        self.out.decreaseIndent()
        self.out.printf("}\n", indent=True)
    
    def printConstDecls(self):
        for cnst in self.cnsts:
            var, num = cnst, self.cnsts[cnst]

            if forFloat() and var in self.floatConstants:
                self.out.printf('%s = %f;\n', var,
                                self.floatConstants[var], indent=True)
            else:
                if config.vbwEnabled and var in self.varsForBitwidth.keys() and (forX86() or forM3()):
                    float_val = self.fixedVarToFloat(var, num)
                    conversion_func = self.getConversionFunction(self.varsForBitwidth[var])
                    bitwidth_str_PX2 = ", %d"%(self.varsForBitwidth[var]) if (self.getPositType(self.varsForBitwidth[var]) == "posit_2_t") else ""
                    self.out.printf('%s_%d = %s(%f%s);\n', var, self.varsForBitwidth[var], conversion_func, float_val, bitwidth_str_PX2, indent=True)
                    # if np.iinfo(np.int16).min <= num <= np.iinfo(np.int16).max:
                    #     self.out.printf('%s_%d = %d;\n', var, self.varsForBitwidth[var], num, indent=True)
                    # elif np.iinfo(np.int32).min <= num <= np.iinfo(np.int32).max:
                    #     self.out.printf('%s_%d = %dL;\n', var, self.varsForBitwidth[var], num, indent=True)
                    # elif np.iinfo(np.int64).min <= num <= np.iinfo(np.int64).max:
                    #     self.out.printf('%s_%d = %dLL;\n', var, self.varsForBitwidth[var], num, indent=True)
                    # else:
                    #     assert False
                else:
                    float_val = self.fixedVarToFloat(var, num)
                    conversion_func = self.getConversionFunction(config.positBitwidth)
                    bitwidth_str_PX2 = ", %d"%(config.positBitwidth) if (self.getPositType(config.positBitwidth) == "posit_2_t") else ""
                    self.out.printf('%s = %s(%f%s);\n', var, conversion_func, float_val, bitwidth_str_PX2, indent=True)
                    
    def fixedVarToFloat(self, var, num):
        scale = self.scales[var]
        float_val = float(np.ldexp(float(num), scale))
        return float_val
    
    def getPX2Suffix(self, bw):
        if config.positBitwidth != 16:
            bw = config.positBitwidth
        if config.useUnverified:
            return ", %d"%(config.positBitwidth)
        if bw == 8:
            return ""
        if bw == 16:
            return ""
        if bw == 32:
            return ""
        return ", %d"%(config.positBitwidth)


    def printMemset(self, ir):
        self.out.printf('memset(', indent=True)
        # If a memory optimized mapping is available for a variable, use that else use original variable name.
        if False and Config.x86MemoryOptimize and forX86() and self.numberOfMemoryMaps in self.scratchSubs:
            self.out.printf("(scratch + %d)", self.scratchSubs[self.numberOfMemoryMaps][ir.e.idf])
        else:
            self.print(ir.e)
        typ_str = self.getPositType(self.varsForBitwidth[ir.e.idf] if config.vbwEnabled else config.positBitwidth) 
        # if config.vbwEnabled:
        #     if hasattr(self, 'varsForBitwidth'):
        #         typ_str = ("int%d_t" % (self.varsForBitwidth[ir.e.idf])) if ir.e.idf in self.varsForBitwidth else typ_str
        #     else:
        #         assert False, "Illegal state, VBW mode but no variable information present"
        self.out.printf(', 0, sizeof(%s) * %d);\n' %
                        ("float" if forFloat() else typ_str, ir.len))

    def printMemcpy(self, ir):
        # If one of the variables' offsets are used, this function computes an expression to reach the memory location including offsets.
        def printFlattenedIndices(indices, shape):
            remSize = np.prod(shape)
            for i in range(len(shape)):
                remSize //= shape[i]
                self.out.printf("%d*(", remSize)
                self.print(indices[i])
                self.out.printf(")")
                if i + 1 < len(shape):
                    self.out.printf("+")
        typ_str = self.getPositType(self.varsForBitwidth[ir.to.idf] if config.vbwEnabled else config.positBitwidth) 
        # if config.vbwEnabled:
        #     if hasattr(self, 'varsForBitwidth'):
        #         # Note ir.to and ir.start are constrained to have the same bit-width.
        #         typ_str = ("int%d_t" % (self.varsForBitwidth[ir.to.idf])) if ir.to.idf in self.varsForBitwidth else typ_str
        #     else:
        #         assert False, "Illegal state, VBW mode but no variable information present"
        typ_str = "float" if forFloat() else typ_str
        self.out.printf('memcpy(', indent=True)
        # If a memory optimized mapping is available for a variable, use that else use original variable name.
        if False and Config.x86MemoryOptimize and self.numberOfMemoryMaps in self.scratchSubs:
            for (a, b, c, d) in [(ir.to.idf, ir.toIndex, 0, ir.to.idx), (ir.start.idf, ir.startIndex, 1, ir.start.idx)]:
                self.out.printf("((scratch + %d + sizeof(%s)*(", self.scratchSubs[self.numberOfMemoryMaps][a], typ_str)
                toIndexed = IRUtil.addIndex(IR.Var(""), b)
                if len(d + b) == 0:
                    self.out.printf("0")
                elif len(d + b) == len(self.decls[a].shape):
                    printFlattenedIndices(d + b, self.decls[a].shape)
                else:
                    assert False, "Illegal state, number of offsets to memcpy should be 0 or match the original tensor dimensions"
                self.out.printf(")))")
                if c == 0:
                    self.out.printf(", ")
        else:
            toIndexed = IRUtil.addIndex(IR.Var(ir.to.idf), ir.to.idx + ir.toIndex)
            startIndexed = IRUtil.addIndex(IR.Var(ir.start.idf), ir.start.idx + ir.startIndex)
            self.out.printf("&")
            self.print(toIndexed)
            self.out.printf(", &")
            self.print(startIndexed)
        self.out.printf(', sizeof(%s) * %d);\n' % (typ_str, ir.length))
    
    def bitSizeToPositSize(self, size, bw):
        if not config.vbwEnabled:
            if config.positBitwidth == 8:
                bw = 8
            elif config.positBitwidth == 16:
                bw = 16
            else:
                bw = 32
        return (size*bw)//8
        

    def computeScratchLocationsFirstFitPriority(self):
        if not Config.x86MemoryOptimize:
            return
        else:
            varToLiveRange, decls = self.preProcessRawMemData()
            def sortkey(a):
                return (a[0][0], -a[0][1], -self.bitSizeToPositSize(a[2], a[3]))
            varToLiveRange.sort(key=sortkey)
            freeSpace = {0:-1}
            freeSpaceRev = {-1:0}
            usedSpaceMap = {}
            totalScratchSize = -1
            listOfDimensions = []
            for ([_,_], var, size, atomSize) in varToLiveRange:
                listOfDimensions.append(size)
            priorityMargin = 19200
            plot = figure(plot_width=1000, plot_height=1000)
            x = []
            y = []
            w = []
            h = []
            c = []
            visualisation = []
            i = 0
            for i in range(len(varToLiveRange)):
                ([startIns, endIns], var, size, atomSize) = varToLiveRange[i]
                if var in self.notScratch:
                    continue
                spaceNeeded = self.bitSizeToPositSize(size, atomSize) # 256 * np.ceil(size * atomSize // 8 /256)
                varsToKill = []
                for activeVar in usedSpaceMap.keys():
                    endingIns = usedSpaceMap[activeVar][0]
                    if endingIns < startIns:
                        varsToKill.append(activeVar)
                for tbk in varsToKill:
                    (st, en) = usedSpaceMap[tbk][1]
                    en += 1
                    freeSpace[st] = en
                    freeSpaceRev[en] = st
                    if en in freeSpace.keys():
                        freeSpace[st] = freeSpace[en]
                        freeSpaceRev[freeSpace[st]] = st
                        del freeSpace[en]
                        del freeSpaceRev[en]
                    if st in freeSpaceRev.keys():
                        freeSpaceRev[freeSpace[st]] = freeSpaceRev[st]
                        freeSpace[freeSpaceRev[st]] = freeSpace[st]
                        del freeSpace[st]
                        del freeSpaceRev[st]
                    del usedSpaceMap[tbk]
                potentialStart = -1
                potentialEnd = -1
                offset = 0
                for j in range(i+1, len(varToLiveRange)):
                    ([startIns_, endIns_], var_, size_, atomSize_) = varToLiveRange[j]
                    if var_ in self.notScratch:
                        continue
                    if startIns_ > endIns:
                        break
                    spaceNeeded_ = self.bitSizeToPositSize(size_, atomSize_)
                    if spaceNeeded_ >= priorityMargin and spaceNeeded < priorityMargin:
                    # if spaceNeeded_ > spaceNeeded or (spaceNeeded_ == spaceNeeded and spaceNeeded < priorityMargin and (endIns_ - startIns_ > endIns - startIns)):
                        offset = max(offset, spaceNeeded_)

                if offset not in freeSpace.keys() and offset > 0:
                    j = 0
                    for key in sorted(freeSpace.keys()):
                        j = key
                        if freeSpace[key] > offset:
                            break
                    if key < offset:
                        st = j
                        en = freeSpace[j]
                        freeSpace[st] = offset
                        freeSpace[offset] = en
                        freeSpaceRev[en] = offset
                        freeSpaceRev[offset] = st

                for start in sorted(freeSpace.keys()):
                    if start < offset:
                        continue
                    end = freeSpace[start]
                    if end - start >= spaceNeeded or end == -1:
                        potentialStart = start
                        potentialEnd = potentialStart + spaceNeeded - 1
                        break
                    else:
                        continue

                usedSpaceMap[var] = (endIns, (potentialStart, potentialEnd))
                freeSpaceEnd = freeSpace[potentialStart]
                del freeSpace[potentialStart]
                if potentialEnd + 1 != freeSpaceEnd:
                    freeSpace[potentialEnd + 1] = freeSpaceEnd
                freeSpaceRev[freeSpaceEnd] = potentialEnd + 1
                if freeSpaceEnd == potentialEnd + 1:
                    del freeSpaceRev[freeSpaceEnd]
                totalScratchSize = max(totalScratchSize, potentialEnd)
                if self.numberOfMemoryMaps not in self.scratchSubs.keys():
                    self.scratchSubs[self.numberOfMemoryMaps] = {}
                self.scratchSubs[self.numberOfMemoryMaps][var] = potentialStart
                varf = var
                if not Config.faceDetectionHacks:
                    while varf in self.coLocatedVariables:
                        varf = self.coLocatedVariables[varf]
                        self.scratchSubs[self.numberOfMemoryMaps][varf] = potentialStart
                x.append((endIns + 1 + startIns) / 2)
                w.append(endIns - startIns + 1)
                y.append((usedSpaceMap[var][1][0] + usedSpaceMap[var][1][1]) / 20000)
                h.append((usedSpaceMap[var][1][1] - usedSpaceMap[var][1][0]) / 10000)
                c.append("#" + ''.join([str(int(j)) for j in 10*np.random.rand(6)]))
                visualisation.append((startIns, var, endIns, usedSpaceMap[var][1][0], usedSpaceMap[var][1][1]))
            plot.rect(x=x, y=y, width=w, height=h, color=c, width_units="data", height_units="data")
            if not forX86():
                show(plot)
            if not forM3():
                self.out.printf("char scratch[%d];\n"%(totalScratchSize+1), indent=True)
            self.out.printf("/* %s */"%(str(self.scratchSubs)))
            f = open("writingPositBW.txt", "w")
            f.write(str(totalScratchSize + 1))
            f.close()
            return totalScratchSize + 1

    def preProcessRawMemData(self):
        varToLiveRange = []
        todelete = []
        decls = dict(self.decls)
        for var in decls.keys():
            if var in todelete:
                continue
            if var not in self.varLiveIntervals:
                todelete.append(var)
                continue
            if hasattr(self, 'floatConstants'):
                if var in self.floatConstants:
                    todelete.append(var)
                    continue
            if hasattr(self, 'intConstants'):
                if var in self.intConstants:
                    todelete.append(var)
                    continue
            if hasattr(self, 'internalVars'):
                if var in self.internalVars:
                    todelete.append(var)
                    continue
            size = np.prod(decls[var].shape)
            if not Config.faceDetectionHacks:
                varf = var
                # Two co located variables can use the same memory location. Hence, the live range of one is
                # updated and the other variable is ignored during memory allocation.
                while varf in self.coLocatedVariables:
                    variableToBeRemoved = self.coLocatedVariables[varf]
                    if self.varLiveIntervals[var][1] == self.varLiveIntervals[variableToBeRemoved][0]:
                        self.varLiveIntervals[var][1] = self.varLiveIntervals[variableToBeRemoved][1]
                        todelete.append(variableToBeRemoved)
                    else:
                        del self.coLocatedVariables[varf]
                        break
                    varf = variableToBeRemoved
            else:
                if var in self.coLocatedVariables.keys():
                    inp = var
                    out = self.coLocatedVariables[var]
                    if self.varLiveIntervals[inp][1] == self.varLiveIntervals[out][0]:
                        self.varLiveIntervals[inp][1] -= 1
            varToLiveRange.append((self.varLiveIntervals[var], var, size, self.varsForBitwidth[var]))
        for var in todelete:
            del decls[var]
        return varToLiveRange, decls

    def printVar(self, ir, isPointer=False):
        # If floating point mode is used or VBW mode is off, then the variable is printed normally (see else branch).
        if config.vbwEnabled:
            # varsForBitwidth variable must be populated during VBW mode.
            if hasattr(self, "varsForBitwidth"):
                # If target code memory optimization is enabled then variables would be renamed to explicit addresses (scratch + N) else they use names like tmpN.
                if False and Config.x86MemoryOptimize:
                    # scratchSubs contains the exact offsets each variable is mapped to, must be present if memory optimizations enabled.
                    if hasattr(self, 'scratchSubs'):
                        # If variable has a offset to which it is mapped to, then replace them with addresses, else use original variable names.
                        if self.numberOfMemoryMaps in self.scratchSubs and ir.idf in self.scratchSubs[self.numberOfMemoryMaps]:
                            type = self.decls[ir.idf]
                            offset = self.scratchSubs[self.numberOfMemoryMaps][ir.idf]
                            # Only Tensors are included in memory optimization, scalars are left to use original variable name.
                            if Type.isTensor(type):
                                resIndex = ' '
                                remSize = np.prod(type.shape)
                                if forM3():
                                    typeCast = "(Q%d_T*)" % (self.varsForBitwidth[ir.idf] - 1)
                                    if isPointer:
                                        self.out.printf("(scratch + %d +" % (offset))
                                    else:
                                        self.out.printf("*(%s(&(scratch[%d + " % (typeCast, offset))
                                else:
                                    typeCast = self.getPositType(self.varsForBitwidth[ir.idf])
                                    if isPointer:
                                        self.out.printf("(scratch + %d +" % (offset))
                                    else:
                                        self.out.printf("%s(scratch[%d + " % (typeCast, offset))
                                self.out.printf("%d * ("% (self.varsForBitwidth[ir.idf] // 8))
                                for i in range(type.dim):
                                    if i >= len(ir.idx):
                                        break
                                    remSize = remSize // type.shape[i]
                                    self.print(ir.idx[i])
                                    self.out.printf("*%d" % remSize)
                                    self.out.printf("+")
                                self.out.printf("0")
                                if forM3():
                                    if isPointer:
                                        self.out.printf("))")
                                    else:
                                        self.out.printf(")])))")
                                else:
                                    if isPointer:
                                        self.out.printf("))")
                                    else:
                                        self.out.printf(")])")
                                return
                            else:
                                pass
                        else:
                            pass
                    else:
                        assert False, "Illegal state, scratchSubs variable should be present if memory optimisation enabled"

                if ir.idf in self.varsForBitwidth and ir.idf[:3] == "tmp" and ir.idf in self.decls:
                    self.out.printf("%s_%d", ir.idf, self.varsForBitwidth[ir.idf])
                else:
                    self.out.printf("%s", ir.idf)
            else:
                assert False, "Illegal state, codegenBase must have variable bitwidth info for VBW mode"
        else:
            self.out.printf("%s", ir.idf)
        for e in ir.idx:
            self.out.printf('[')
            self.print(e)
            self.out.printf(']')

                
