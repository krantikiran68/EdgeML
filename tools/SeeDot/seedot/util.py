# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


import numpy as np
import platform
import ctypes, math

import seedot.config as config
import logging

'''
This code contains the Configuration information to control
the exploration of SeeDot.
Additionally it contains the helper utility functions that are used by 
the entire program. 
'''


class Config:

    expBigLength = 6
        # This parameter is used if exp below is set to "table".
        # Please refer to old SeeDot (PLDI'19) paper Section 5.3.1.
        # In the fixed point mode, the top 2 * expBigLength bits are used to compute e^x, the remaining bits are discarded.
    exp = "math"  # "table" "math" "new table"
        # This parameter controls the type of exponentiation to be used in the fixed point code.
        # "table": Uses the method used in old SeeDot (PLDI '19).
        # "new table": Uses the method used in new SeeDot (OOPSLA '20).
        # "math": Uses floating point implementation of math.h.
    codegen = "funcCall"  # "funcCall"
        # Type of codegen: as a sequence of library function calls or inlined (no longer supported).
    debug = False
        # Enable debug mode of the generated fixed-point/floating-point C++ inference code.
        # This should be always set to False.
    saturateArithmetic = False
        # Enable saturating arithmetic in the generated fixed-point code.
    fastApproximations = False
        # Enable fast approximations in the generated fixed-point code, like:
        #   -> In multiplication, truncate first and multiply instead of multiply first in higher bitwidth and then truncate.
        #   -> Similarly in multiplication-like functions convolution, hadamard product etc.
    x86MemoryOptimize = True
    x86MemoryOptimize = False if not config.vbwEnabled else x86MemoryOptimize
        # Enable memory optimization in the generated fixed-point code in x86, arduino or m3 codegen.
    memoryLimit = 200000
        # The maximum memory present on the target device. Used if memory optimizations are enabled in the target codegen.
    largeVariableLimit = 50000
        # Any variable with more elements than this are prioritized for demotion to 8 bits.
    defragmentEnabled = False
        # Enable defragmentation. Currently not supported, so must be kept to False.
    faceDetectionHacks = False
        # Quick fixes for face detection model. This parameter must always be kept to False apart from debugging purposes.


def isSaturate():
    return Config.saturateArithmetic

def isfastApprox():
    return Config.fastApproximations

def windows():
    return platform.system() == "Windows"

def linux():
    return platform.system() == "Linux"

def getAlgo():
    return Config.algo

def setAlgo(algo: str):
    Config.algo = algo

def getEncoding():
    return Config.encoding

def setEncoding(encoding: str):
    Config.encoding = encoding

def forFixed():
    return Config.encoding == config.Encoding.fixed

def forFloat():
    return Config.encoding == config.Encoding.floatt

def forPosit():
    return Config.encoding == config.Encoding.posit

def getTarget():
    return Config.target

def setTarget(target: str):
    Config.target = target

def forArduino():
    return Config.target == config.Target.arduino

def forM3():
    return Config.target == config.Target.m3

def forHls():
    return Config.target == config.Target.Hls

def forVerilog():
    return Config.target == config.Target.Verilog

def forX86():
    return Config.target == config.Target.x86

def getProfileLogFile():
    return Config.profileLogFile

def setProfileLogFile(file):
    Config.profileLogFile = file

def getExpBitLength():
    return Config.expBigLength

def getMaxScale():
    return Config.maxScale

def setMaxScale(x: int):
    Config.maxScale = x

def getShrType():
    # "shr" "shr+" "div" "negate"
    return "div"

def useMathExp():
    return Config.exp == "math"

def useTableExp():
    return Config.exp == "table"

def useNewTableExp():
    return Config.exp == "new table"

def genFuncCalls():
    return Config.codegen == "funcCall"

def debugMode():
    return Config.debug

def copy_dict(dict_src: dict, diff={}):
    dict_res = dict(dict_src)
    dict_res.update(diff)
    return dict_res

# set number of workers for FPGA sparseMUL
def setNumWorkers(WorkerThreads):
    Config.numWorkers = WorkerThreads

def getNumWorkers():
    return Config.numWorkers

# z = [y1,y2,..] = [[x1,..], [x2,..], ..] --> [x1,.., x2,.., ..]
def flatten(z: list):
    return [x for y in z for x in y]

def computeScalingFactor(val):
    '''
    The scale computation algorithm is different while generating function calls and while generating inline code.
    The inline code generation uses an extra padding bit for each parameter and is less precise.
    The scales computed while generating function calls uses all bits.
    '''
    if genFuncCalls():
        return computeScalingFactorForFuncCalls(val)
    else:
        return computeScalingFactorForInlineCodegen(val)

def computeScalingFactorForFuncCalls(val):
    l = np.log2(val)
    if int(l) == l:
        c = l + 1
    else:
        c = np.ceil(l)
    return -int((config.wordLength - 1) - c)

def computeScalingFactorForInlineCodegen(val):
    return int(np.ceil(np.log2(val) - np.log2((1 << (config.wordLength - 2)) - 1)))


# Logging Section
def getLogger():
    log =  logging.getLogger()    
    return log

# get bitwidth of index array in case of Sparse Matrices
def getIdxBitwidth(bitwidth):
    if bitwidth not in [8, 16, 32]:
        return config.wordLength

# Calculating ULP error of two floating-point values
def calculateULPError(val, baseVal):
    val = (ctypes.c_uint.from_buffer(ctypes.c_float(float(val)))).value
    baseVal = (ctypes.c_uint.from_buffer(ctypes.c_float(float(baseVal)))).value
    return abs(val - baseVal)

# Calculating Relative error of two floating-point values
def calculateRelativeError(val, baseVal):
    val = float(val)
    baseVal = float(baseVal)
    if baseVal == 0.0:
        return math.fabs(val - baseVal)
    error = math.fabs((val  - baseVal) / baseVal)
    return error

# Compution first fit priority memory requirement
def preProcessRawMemData(decls, coLocatedVariables, varLiveIntervals, varsForBitwidth, floatConstants, intConstants, internalVars):
    varToLiveRange = []
    todelete = []
    decls = dict(decls)
    for var in decls.keys():
        if var in todelete:
            continue
        if var not in varLiveIntervals:
            todelete.append(var)
            continue
        if floatConstants is not None:
            if var in floatConstants:
                todelete.append(var)
                continue
        if intConstants is not None:
            if var in intConstants:
                todelete.append(var)
                continue
        if internalVars is not None:
            if var in internalVars:
                todelete.append(var)
                continue
        size = np.prod(decls[var].shape)
        varf = var
        # Two co located variables can use the same memory location. Hence, the live range of one is
        # updated and the other variable is ignored during memory allocation.
        while varf in coLocatedVariables:
            variableToBeRemoved = coLocatedVariables[varf]
            if varLiveIntervals[var][1] == varLiveIntervals[variableToBeRemoved][0]:
                varLiveIntervals[var][1] = varLiveIntervals[variableToBeRemoved][1]
                todelete.append(variableToBeRemoved)
            else:
                del coLocatedVariables[varf]
                break
            varf = variableToBeRemoved
        varToLiveRange.append((varLiveIntervals[var], var, size, varsForBitwidth[var]))
    for var in todelete:
        del decls[var]
    return varToLiveRange, decls, coLocatedVariables

def computeScratchLocationsFirstFitPriority(decls, coLocatedVariables, varLiveIntervals, notScratch, varsForBitwidth, floatConstants, intConstants, internalVars):
    varToLiveRange, decls, coLocatedVariables = preProcessRawMemData(decls, coLocatedVariables, varLiveIntervals, varsForBitwidth, floatConstants, intConstants, internalVars)
    def sortkey(a):
        return (a[0][0], -a[0][1], -(a[2]*a[3])//8)
    varToLiveRange.sort(key=sortkey)
    freeSpace = {0:-1}
    freeSpaceRev = {-1:0}
    usedSpaceMap = {}
    totalScratchSize = -1
    listOfDimensions = []
    for ([_,_], var, size, atomSize) in varToLiveRange:
        listOfDimensions.append(size)
    priorityMargin = 19200
    x = []
    y = []
    w = []
    h = []
    c = []
    i = 0
    for i in range(len(varToLiveRange)):
        ([startIns, endIns], var, size, atomSize) = varToLiveRange[i]
        if var in notScratch:
            continue
        spaceNeeded = size * atomSize // 8 # 256 * np.ceil(size * atomSize // 8 /256)
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
            if var_ in notScratch:
                continue
            if startIns_ > endIns:
                break
            spaceNeeded_ = (size_ * atomSize_) // 8
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
        
        varf = var
        if not Config.faceDetectionHacks:
            while varf in coLocatedVariables:
                varf = coLocatedVariables[varf]
        x.append((endIns + 1 + startIns) / 2)
        w.append(endIns - startIns + 1)
        y.append((usedSpaceMap[var][1][0] + usedSpaceMap[var][1][1]) / 20000)
        h.append((usedSpaceMap[var][1][1] - usedSpaceMap[var][1][0]) / 10000)
        c.append("#" + ''.join([str(int(j)) for j in 10*np.random.rand(6)]))
    return totalScratchSize + 1