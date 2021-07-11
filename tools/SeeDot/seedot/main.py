# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


import argparse
import datetime
from distutils.dir_util import copy_tree
import os
import shutil
import sys
import operator
import tempfile
import traceback
from typing_extensions import final
from numpy.lib.histograms import _histogram_bin_edges_dispatcher
from numpy.testing._private.utils import memusage
from tqdm import tqdm
import numpy as np
import math
import ast

from seedot.compiler.converter.converter import Converter

import seedot.config as config
from seedot.compiler.compiler import Compiler
from seedot.predictor import Predictor
import seedot.util as Util

'''
Overall compiler logic is maintained in this file. Please refer to architecture.md for a 
detailed explanation of how the various modules interact with each other.
'''


class Main:

    def __init__(self, algo, encoding, target, trainingFile, testingFile, modelDir, sf, metric, dataset, numOutputs, source):
        self.algo, self.encoding, self.target = algo, encoding, target
        self.trainingFile, self.testingFile, self.modelDir = trainingFile, testingFile, modelDir
        self.sf = sf
            # MaxScale factor. Used in the original version of SeeDot.
            # Refer to PLDI'19 paper: maxscale parameter P.
        self.dataset = dataset
            # Dataset which is being evaluated.
        self.accuracy = {}
            # SeeDot examines accuracy of multiple codes.
            # This variable contains a map from code ID -> corresponding accuracy.
        self.metric = metric
            # This can be accuracy, disagreements (see OOPSLA'20 paper: disagreement ratio) or
            # reduced disagreement (disagreement ratio for only those parameters where float model prediction is correct).
        self.numOutputs = numOutputs
            # Number of outputs, it is 1 for a single-class prediction. For n simultaneous predictions, it is n.
        self.source = source
            # SeeDot or ONNX or TensorFlow.
        self.variableSubstitutions = {}
            # Evaluated during profiling code run (runForFloat). During compilation, variable names get substituted
            # into other names, which is stored in this variable. It is required in the case that an attribute
            # computed for some variable might have to be propagated to other variables, like scales and bitwidths.
        self.scalesForX = {}
            # Populated for multiple code generation (performSearch: if vbwEnabled is True). SeeDot carries out an
            # exploration (OOPSLA'20 paper, Section 6.2) where multiple codes are compiled and evaluated. This variable
            # stored a map from code ID -> corresponding scale of input variable 'X'.
        self.scalesForY = {}
            # Populated for multiple code generation (performSearch: if vbwEnabled is True). SeeDot carries out an
            # exploration (OOPSLA'20 paper, Section 6.2) where multiple codes are compiled and evaluated. This variable
            # stored a map from code ID -> corresponding scale of output variable 'Y' (if the problem is regression,
            # for classification problems 'Y' is already an integer hence does not need to be scaled).
        self.problemType = config.ProblemType.default
            # Edited when converter module determines whether problem is regression or classification.
        self.variableToBitwidthMap = {}
            # This variable holds the bit-width assignment for all variables in the code. The keys (variable names)
            # are identified during profiling run (runForFloat) and the values (bitwidths) are evaluated during
            # multiple code generation (performSearch: if vbwEnabled is True). By default, the bit-widths are set to 16.
        self.sparseMatrixSizes = {}
            # Populated during profiling code run (runForFloat). For sparse matrix multiplication, the matrices
            # in the generated code have different sizes than in the input code due to CSR represenation, and
            # the generated matrix sizes are stored here.
        self.varDemoteDetails = []
            # Populated during variable demotion in VBW mode. This stores the resultant code performance
            # when a subset of variables is demoted.
        self.flAccuracy = -1
            # Populated during profiling code run. Accuracy of the floating point code.
        self.allScales = {}
            # This stores the final scale assignment of every variable in the code (considering their
            # bit-width assignments). This variable is updated after every code generated, so eventually it
            # is populated with the bit-width assignment of the final generated code.
        self.demotedVarsList = []
            # Populated in VBW mode after exploration is completed. After the compiler determines the variables
            # to be demoted, this variable is populated with a list of those variables.
        self.demotedVarsOffsets = {}
            # Populated in VBW mode after exploration is completed. After the compiler determines the variables
            # to be demoted, this variable is populated with a map of variables to the scale offset which gives the best accuracy.
        self.biasShifts = {}
            # For simplifying bias addition, populated after every code run, used for M3 codegen.
            # In operations like WX + B, B is mostly used once in the code. So all the fixed point computations are clubbed into one.
        self.varSizes = {}
            # Map from a variable to number of elements it holds. Populated in floating point mode.
        self.errorHeatMap = {}
            # Map to store the error_map of previous iteration of Haunter for comparision with previous iteration
        self.varLiveIntervals = {}
            # Map that is used to calculate the scratch buffer sizes
        self.notScratch = []
            # List of variables that have to be excluded from memory calculation
        self.coLocatedVariables = {}
            # Map of coLocated Variables that can use the same memory location
        self.intConstants, self.floatConstants, self.internalVars = [], [], []
            # Constant information for use in computing memory 
        self.decls = []
            # Declarations, for use in computing memory usage
        self.secondChance = []
            # Variables that have been given a second chance 
        self.doNotPromote = []
            # Variables that have been demoted to 8-bit and shouldn't be promoted again
        self.configurationAccMap = []

    # This function is invoked right at the beginning for moving around files into the working directory.
    def setup(self):
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        copy_tree(os.path.join(curr_dir, "Predictor"), os.path.join(config.tempdir, "Predictor"))

        if self.target == config.Target.arduino:
            for fileName in ["arduino.ino", "config.h", "predict.h"]:
                srcFile = os.path.join(curr_dir, "arduino", fileName)
                destFile = os.path.join(config.outdir, fileName)
                shutil.copyfile(srcFile, destFile)
        elif self.target == config.Target.m3:
            for fileName in ["datatypes.h", "mbconv.h", "utils.h"]:
                srcFile = os.path.join(curr_dir, "..", "..", "..",  "c_reference", "include", "quantized_%s"%fileName)
                destFile = os.path.join(config.outdir, "quantized_%s"%fileName)
                shutil.copyfile(srcFile, destFile)
            for fileName in ["mbconv.c", "utils.c"]:
                srcFile = os.path.join(curr_dir, "..", "..", "..",  "c_reference", "src", "quantized_%s"%fileName)
                destFile = os.path.join(config.outdir, "quantized_%s"%fileName)
                shutil.copyfile(srcFile, destFile)
            for fileName in ["main.c", "predict.h", "Makefile"]:
                srcFile = os.path.join(curr_dir, "m3", fileName)
                destFile = os.path.join(config.outdir, fileName)
                shutil.copyfile(srcFile, destFile)

    def get_input_file(self):
        if self.source == config.Source.seedot:
            return os.path.join(self.modelDir, "input.sd")
        elif self.source == config.Source.onnx:
            return os.path.join(self.modelDir, "input.onnx")
        else:
            return os.path.join(self.modelDir, "input.pb")

    # Generates one particular fixed-point or floating-point code.
    # Arguments:
    #   encoding:                float or fixed
    #   target:                 target device (x86, arduino or m3)
    #   sf:                     maxScale factor (check description above)
    #   The next three parameters are used to control how many candidate codes are built at once. Generating
    #   multiple codes at once and building them simultaneously helps avoid multiple build overheads as well
    #   as saves data processing overhead at runtime.
    #   generateAllFiles:       if True, it generates multiple files like datasets, configuration etc. if False,
    #                           only generates the inference code.
    #   id:                     Multiple inference codes are designated by a code ID 'N' (function names are
    #                           seedot_fixed_'N').
    #   printSwitch:            whether or not to print a switch between multiple inference codes (only needs to
    #                           be True at the last code being generated as the switch is print right after that).
    #   scaleForX:              scale of input X for the particular fixed-point code.
    #   variableToBitwidthMap:  bitwidth assignments for the particular fixed-point code.
    #   demotedVarsList:        set of variables using 8-bits in the particular fixed-point code.
    #   demotedVarsOffsets:     map from variables to scale offsets for particular fixed-point code.
    #   paramInNativeBitwidth:  if False, it means model parameters are stored as 8-bit/16-bit integers mixed.
    #                           If True, it means model parameters are stored as 16-bit integers only (16 is native bit-width).
    def compile(self, encoding, target, sf, generateAllFiles=True, id=None, printSwitch=-1, scaleForX=None, variableToBitwidthMap=None, demotedVarsList=[], demotedVarsOffsets={}, paramInNativeBitwidth=True):
        Util.getLogger().debug("Generating code...")

        if variableToBitwidthMap is None:
            variableToBitwidthMap = dict(self.variableToBitwidthMap)

        # Set input and output files.
        inputFile = self.get_input_file()
        profileLogFile = os.path.join(
            config.tempdir, "Predictor", "output", "float", "profile.txt")

        logDir = os.path.join(config.outdir, "output")
        os.makedirs(logDir, exist_ok=True)
        if encoding == config.Encoding.floatt:
            outputLogFile = os.path.join(logDir, "log-float.txt")
        elif encoding == config.Encoding.posit:
            outputLogFile = os.path.join(logDir, "log-posit.txt")
        else:
            if config.ddsEnabled:
                outputLogFile = os.path.join(logDir, "log-fixed-" + str(abs(scaleForX)) + ".txt")
            else:
                outputLogFile = os.path.join(logDir, "log-fixed-" + str(abs(sf)) + ".txt")

        if target == config.Target.arduino:
            outdir = os.path.join(config.outdir, str(config.wordLength), self.algo, self.dataset)
            os.makedirs(outdir, exist_ok=True)
            outputDir = os.path.join(outdir)
        elif target == config.Target.m3:
            outdir = os.path.join(config.outdir)
            os.makedirs(outdir, exist_ok=True)
            outputDir = os.path.join(outdir)
        elif target == config.Target.x86:
            outputDir = os.path.join(config.tempdir, "Predictor")

        obj = Compiler(self.algo, encoding, target, inputFile, outputDir,
                        profileLogFile, sf, self.source, outputLogFile,
                        generateAllFiles, id, printSwitch, self.variableSubstitutions,
                        scaleForX,
                        variableToBitwidthMap, self.sparseMatrixSizes, demotedVarsList, demotedVarsOffsets,
                        paramInNativeBitwidth)
        res, state = obj.run()
        self.biasShifts = obj.biasShifts
        self.allScales = dict(obj.varScales)
        if encoding == config.Encoding.floatt:
            self.variableSubstitutions = obj.substitutions
            self.variableToBitwidthMap = dict.fromkeys(obj.independentVars, config.wordLength if config.wordLength == config.positBitwidth else config.positBitwidth) 
            self.varSizes = obj.varSizes
            self.extendedVariableToBitwidthMap = state[11]
        else:
            self.varLiveIntervals = state[12]
            self.notScratch = state[13]
            self.coLocatedVariables = state[14]
            self.decls = state[0]
            self.intConstants = state[4]
            self.floatConstants = state[8]
            self.internalVars = state[7]
            self.extendedVariableToBitwidthMap = state[11]

        self.problemType = obj.problemType
        if id is None:
            self.scaleForX = obj.scaleForX
            self.scaleForY = obj.scaleForY
        else:
            self.scalesForX[id] = obj.scaleForX
            self.scalesForY[id] = obj.scaleForY

        Util.getLogger().debug("Completed")
        return True

    # Runs the converter project to generate the input files using reading the training model.
    # Arguments:
    #   version:                float or fixed.
    #   datasetType:            train or test.
    #   target:                 target device (x86, arduino or m3).
    #   varsForBitwidth:        bitwidth assignments used to generate model files. If none,
    #                           default bitwidth 16 used for all variables.
    #   demotedVarsOffsets:     Keys are list of variables which use 8 bits.
    def convert(self, encoding, datasetType, target, varsForBitwidth={}, demotedVarsOffsets={}):
        Util.getLogger().debug("Generating input files for %s %s dataset..." %
              (encoding, datasetType))

        # Create output dirs.
        if target == config.Target.arduino:
            outputDir = os.path.join(config.outdir, "input")
            datasetOutputDir = outputDir
        elif target == config.Target.m3:
            outputDir = os.path.join(config.outdir, "input")
            datasetOutputDir = outputDir
        elif target == config.Target.x86:
            outputDir = os.path.join(config.tempdir, "Predictor")
            datasetOutputDir = os.path.join(config.tempdir, "Predictor", "input")
        else:
            assert False

        os.makedirs(datasetOutputDir, exist_ok=True)
        os.makedirs(outputDir, exist_ok=True)

        inputFile = self.get_input_file()

        try:
            varsForBitwidth = dict(varsForBitwidth)
            for var in demotedVarsOffsets:
                varsForBitwidth[var] = config.wordLength // 2
            obj = Converter(self.algo, encoding, datasetType, target, self.source,
                            datasetOutputDir, outputDir, varsForBitwidth, self.allScales, self.numOutputs, self.biasShifts, self.scaleForY if hasattr(self, "scaleForY") else None)
            obj.setInput(inputFile, self.modelDir,
                         self.trainingFile, self.testingFile)
            obj.run()
            if encoding == config.Encoding.floatt:
                self.sparseMatrixSizes = obj.sparseMatrixSizes
        except Exception as e:
            traceback.print_exc()
            return False

        Util.getLogger().debug("Done")
        return True

    # Build and run the Predictor project.
    def predict(self, encoding, datasetType, shadow=False, counted=False):
        outputDir = os.path.join("output", encoding)

        curDir = os.getcwd()
        os.chdir(os.path.join(config.tempdir, "Predictor"))

        obj = Predictor(self.algo, encoding, datasetType,
                        outputDir, self.scaleForX, self.scalesForX, self.scaleForY, self.scalesForY, self.problemType, self.numOutputs, shadow, counted)
        execMap = obj.run()

        os.chdir(curDir)
        return execMap

    # Compile and run the generated code once for a given scaling factor.
    # The arguments are explain in the description of self.compile().
    # The function is named partial compile as in one C++ output file multiple inference codes are generated.
    # One invocation of partialCompile generates only one of the multiple inference codes.
    def partialCompile(self, encoding, target, scale, generateAllFiles, id, printSwitch, variableToBitwidthMap=None, demotedVarsList=[], demotedVarsOffsets={}, paramInNativeBitwidth=True):
        if config.ddsEnabled:
            res = self.compile(encoding, target, None, generateAllFiles, id, printSwitch, scale, variableToBitwidthMap, demotedVarsList, demotedVarsOffsets, paramInNativeBitwidth)
        else:
            res = self.compile(encoding, target, scale, generateAllFiles, id, printSwitch, None, variableToBitwidthMap, demotedVarsList, demotedVarsOffsets, paramInNativeBitwidth)
        if res == False:
            return False
        else:
            return True

    # Runs the C++ file which contains multiple inference codes. Reads the output of all inference codes,
    # arranges them and returns a map of inference code descriptor to performance.
    def  runAll(self, encoding, datasetType, codeIdToScaleFactorMap, demotedVarsToOffsetToCodeId=None, doNotSort=False, printAlso=False):
        execMap = self.predict(encoding, datasetType)
        if execMap == None:
            return False, True

        # Used by test module.
        if self.algo == config.Algo.test:
            for codeId, sf in codeIdToScaleFactorMap.items():
                self.accuracy[sf] = execMap[str(codeId)]
                Util.getLogger().debug("The 95th percentile error for sf" + str(sf) + "with respect to dataset is " + str(execMap[str(codeId)][0]) + "%.")
                Util.getLogger().debug("The 95th percentile error for sf" + str(sf) + "with respect to float execution is " + str(execMap[str(codeId)][1]) + "%.\n")
            return True, False

        # During the third exploration phase, when multiple codes are generated at once, codeIdToScaleFactorMap
        # is populated with the codeID to the code description (bitwidth assignments of different variables).
        # After executing the code, print out the accuracy of the code against the code ID.
        if codeIdToScaleFactorMap is not None:
            for codeId, sf in codeIdToScaleFactorMap.items():
                if encoding == config.Encoding.posit and sf == None:
                    sf = 0
                    self.sf = 0 if self.sf == None else self.sf
                self.accuracy[sf] = execMap[str(codeId)]
                if printAlso:
                    print("Accuracy at scale factor %d is %.3f%%, Disagreement Count is %d, Reduced Disagreement Count is %d" % (sf, execMap[str(codeId)][0], execMap[str(codeId)][1], execMap[str(codeId)][2]))
                else:
                    Util.getLogger().info("Accuracy at scale factor %d is %.3f%%, Disagreement Count is %d, Reduced Disagreement Count is %d\n" % (sf, execMap[str(codeId)][0], execMap[str(codeId)][1], execMap[str(codeId)][2]))
                if datasetType == config.DatasetType.testing and self.target == config.Target.arduino:
                    outdir = os.path.join(config.outdir, str(config.wordLength), self.algo, self.dataset)
                    os.makedirs(outdir, exist_ok=True)
                    file = open(os.path.join(outdir, "res"), "w")
                    file.write("Demoted Vars:\n")
                    file.write(str(self.demotedVarsOffsets) if hasattr(self, 'demotedVarsOffsets') else "")
                    file.write("\nAll scales:\n")
                    file.write(str(self.allScales))
                    file.write("\nAccuracy at scale factor %d is %.3f%%, Disagreement Count is %d, Reduced Disagreement Count is %d\n" % (sf, execMap[str(codeId)][0], execMap[str(codeId)][1], execMap[str(codeId)][2]))
                    file.close()
        else:
            # During fourth exploration phase, when the accuracy drops of every variable is known, the variables are cumulatively demoted
            # in order of better accuracy/disagreement count which is handled in this block.
            def getMetricValue(a):
                if self.metric == config.Metric.accuracy:
                    return (a[1][0], -a[1][1], -a[1][2])
                elif self.metric == config.Metric.disagreements:
                    return (-a[1][1], -a[1][2], a[1][0])
                elif self.metric == config.Metric.reducedDisagreements:
                    return (-a[1][2], -a[1][1], a[1][0])
            allVars = []
            for demotedVars in demotedVarsToOffsetToCodeId:
                offsetToCodeId = demotedVarsToOffsetToCodeId[demotedVars]
                Util.getLogger().debug("Demoted vars: %s\n" % str(demotedVars))

                x = [(i, execMap[str(offsetToCodeId[i])]) for i in offsetToCodeId]
                x.sort(key=getMetricValue, reverse=True)
                allVars.append(((demotedVars, x[0][0]), x[0][1]))

                for offset in offsetToCodeId:
                    codeId = offsetToCodeId[offset]
                    Util.getLogger().debug("Offset %d (Code ID %d): Accuracy %.3f%%, Disagreement Count %d, Reduced Disagreement Count %d\n" %(offset, codeId, execMap[str(codeId)][0], execMap[str(codeId)][1], execMap[str(codeId)][2]))
            self.varDemoteDetails += allVars
            # For the sec
            if not doNotSort:
                self.varDemoteDetails.sort(key=getMetricValue, reverse=True)
        return True, False

    def clearHeatMapLog(self):
        outputDir = os.path.join(config.tempdir, "Predictor")
        file = os.path.join(outputDir, "debugLog")
        open(file, "w").close()
        return

    def initializeErrorMap(self):
        for var in self.variableToBitwidthMap.keys():
            self.errorHeatMap[var] = 0

    def performSearch(self):
        # Stage 0: Initialise error_heat_map:
        self.initializeErrorMap()

        # Stage I pre-processing: 
        # Compute 16-bit accuracy data to used as a basis for heat_map generation
        for var in self.variableToBitwidthMap.keys():
            self.variableToBitwidthMap[var] = 16
        
        self.partialCompile(self.encoding, config.Target.x86, None, True, None, 0, dict(self.variableToBitwidthMap), [], {})
        self.clearHeatMapLog()
        valMap16Bit = self.runAndCreateVarMap(self.encoding, config.DatasetType.training, saveConfig=False)

        self.clearHeatMapLog()
        
        # Stage II: 0th iteration
        # Computed the 8-bit accuracy data and compares it to 
        # base_line for heat_map generation
        for var in self.variableToBitwidthMap.keys():
            self.variableToBitwidthMap[var] = 8

        self.partialCompile(self.encoding, config.Target.x86, None, True, None, 0, dict(self.variableToBitwidthMap), [], {})
        valMap8Bit = self.runAndCreateVarMap(self.encoding, config.DatasetType.training)
        self.previousMemUsage = self.computMemoryUsage(self.variableToBitwidthMap, self.extendedVariableToBitwidthMap)
        self.previousVarConfig = dict(self.variableToBitwidthMap)
        self.previousResConfig = dict(self.variableToBitwidthMap)
        heat_map = self.createHeatMap(valMap8Bit, valMap16Bit)
        self.variableToBitwidthMap = dict(self.runAttackingAlgorithm(self.encoding, config.DatasetType.training, heat_map, isFirstIter = True))

        while True:
            self.clearHeatMapLog()
            self.partialCompile(self.encoding, config.Target.x86, None, True, None, 0, dict(self.variableToBitwidthMap), [], {})
            valMapvbW = self.runAndCreateVarMap(self.encoding, config.DatasetType.training)
            print("Run Completed")
            heat_map = self.createHeatMap(valMapvbW, valMap16Bit)
            print("Heat Map Created")
            self.variableToBitwidthMap = dict(self.runAttackingAlgorithm(self.encoding, config.DatasetType.training, heat_map))
            if self.previousResConfig == self.variableToBitwidthMap:
                print("Termination condition reached at : " , str(self.variableToBitwidthMap))
                # maxAcc = 0
                # for bwConfig in self.configurationAccMap:
                #     bwConfig, acc = bwConfig 
                #     if acc > maxAcc:
                #         maxAcc = acc
                #         self.variableToBitwidthMap = dict(bwConfig)
                return
            else:
                self.previousResConfig = dict(self.variableToBitwidthMap)
            pass

    def promote(self, varName):
        self.variableToBitwidthMap[varName] = 16
        return True
    def findMinNumPromote(self, heat_map):
        min_num_promote = 0
        for i in range(len(heat_map)):
            error = heat_map[i][1][0]
            if error > self.movingErrorThreshold:
                min_num_promote += 1
            else:
                break
        return min_num_promote

    def runAttackingAlgorithm(self, encoding, datasetType, heat_map, isFirstIter = False):
        min_num_promote = self.findMinNumPromote(heat_map) if not isFirstIter else 0
        promoted_count = 0
        promoted_in_this_iter = []
        for i in range(len(heat_map)):
            var = heat_map[i][0]
            if self.variableToBitwidthMap[var] == 8:
                if var in self.doNotPromote:
                    continue
                promoted_flag = self.promote(var)
                if not promoted_flag:
                    continue
                promoted_count += 1
                promoted_in_this_iter.append(var)
                memUsage = self.checkMemoryThreshold()
                if memUsage > config.memoryLimit:
                    memThreshold = False
                else: 
                    memThreshold = True
                if memThreshold:
                    print("Case 0")
                    self.previousMemUsage = memUsage
                    self.previousVarConfig = dict(self.variableToBitwidthMap)
                    self.movingErrorThreshold = heat_map[i][1][0]
                    continue
                else:
                    if promoted_count > min_num_promote:
                        print("Case 1")
                        self.movingErrorThreshold = heat_map[i-1][1][0]
                        return self.previousVarConfig
                    elif not self.defending(encoding, datasetType, heat_map, promoted_in_this_iter):
                        # Attempting to demote variables to make room failed 
                        print("Case 2")
                        self.movingErrorThreshold = heat_map[i-1][1][0]
                        return self.previousVarConfig
                    else:
                        # Successfully demoted variables
                        print("Case 3")
                        continue
        return self.variableToBitwidthMap

    def checkMemoryThreshold(self):
        memUsage = self.computMemoryUsage(self.variableToBitwidthMap, self.extendedVariableToBitwidthMap)
        return memUsage

    def defending(self, encoding, datasetType, heat_map, promoted_in_this_iter):
        cool_map = list(heat_map)
        heat_map_vars = [var[0] for var in heat_map]
        cool_map.reverse()
        codeIdToDemotedVars = {}
        codeIdToDemotedVars[0] = ""
        self.partialCompile(self.encoding, config.Target.x86, None, True, None, -1, dict(self.variableToBitwidthMap), [], {})
        demoteableVars = []
        initialSecondChance = list(self.secondChance)
        for var in cool_map:
            var = var[0]
            if var in promoted_in_this_iter:
                continue
            if (self.variableToBitwidthMap[var] in [12, 16]) and (var in heat_map_vars):
                demoteableVars.append(var)
        numCodes = len(demoteableVars)
        if numCodes == 0:
            return False

        for i in range(len(demoteableVars)):
            var = demoteableVars[i]
            newbitwidthMap = dict(self.variableToBitwidthMap)
            newbitwidthMap[var] = 12 if newbitwidthMap[var] == 16 else 8 
            self.partialCompile(self.encoding, config.Target.x86, None, False, i+1, -1 if i != (numCodes - 1) else numCodes, dict(newbitwidthMap), [], {})
            codeIdToDemotedVars[i+1] = var

        allowedDemotions = self.runAccComputation(encoding, datasetType, codeIdToDemotedVars)

        newbitwidthMap = dict(self.variableToBitwidthMap)
        for _ in range(2):
            prospective_do_not_promote = []
            for i in range(len(cool_map)):
                var = cool_map[i][0]
                if var not in allowedDemotions:
                    continue
                if newbitwidthMap[var] == 12:
                    if var not in self.secondChance:
                        self.secondChance.append(var)
                        continue
                newbitwidthMap[var] = 12 if newbitwidthMap[var] == 16 else 8
                if newbitwidthMap[var] == 8:
                    prospective_do_not_promote.append(var)
                memUsage = self.computMemoryUsage(newbitwidthMap, self.extendedVariableToBitwidthMap)
                if memUsage < config.memoryLimit:
                    self.variableToBitwidthMap = dict(newbitwidthMap)
                    self.doNotPromote.extend(prospective_do_not_promote)
                    return True
            finalSecondChance = list(self.secondChance)
            if promoted_in_this_iter == []:
                if finalSecondChance != initialSecondChance:
                    continue
            else:
                break
        return False

    def runAccComputation(self, encoding, datasetType, codeIdToDemotedVarMap):
        execMap = self.predict(encoding, datasetType, counted = True)
        if execMap == None:
            assert False, "Accuracy computation run failed in runAccComputation()"
        
        allowedDemotions = []
        baseAcc = float(execMap['default'][2])

        for codeId in codeIdToDemotedVarMap.keys():
            if codeId == 0:
                continue
            acc = float(execMap[str(codeId)][2])
            var = codeIdToDemotedVarMap[codeId]
            if self.variableToBitwidthMap[var] == 16:
                if math.fabs(acc - baseAcc) < config.accThreshold:
                    allowedDemotions.append(var)
            elif self.variableToBitwidthMap[var] == 12:
                if math.fabs(acc - baseAcc) < config.accThreshold:
                    allowedDemotions.append(var)
            return allowedDemotions

    def createHeatMap(self, newMap, baseMap):
        heat_map = {}
        heat_map_sizes = {}
        assert newMap.keys() == baseMap.keys(), "Different keys in Heat Map creation"

        for var in newMap.keys():
            values = newMap[var]
            baseVals = baseMap[var]
            assert len(values) == len(baseVals), \
                    "Same number of values expected for variable %s"%(var)
            for i in range(len(values)):
                error = Util.calculateRelativeError(values[i], baseVals[i])
                # print(error)
                if var in heat_map.keys():
                    heat_map[var] = max(heat_map[var], error)
                else:
                    heat_map[var] = error
        
        for var in heat_map.keys():
            if var not in self.errorHeatMap.keys():
                self.errorHeatMap[var] = 0
            heat_map_sizes[var] = (heat_map[var], self.varSizes[var], self.errorHeatMap[var])

        heat_map_list = self.createHeatMapWeightedList(heat_map_sizes)
        return heat_map_list

    def createHeatMapWeightedList(self, heat_map):
        def sortErrorWeight(errorWeight):
            return 0.1 * errorWeight[1][1]  / (0.8 * errorWeight[1][0] + 0.2 * errorWeight[1][2])
        for var in heat_map.keys():
            self.errorHeatMap[var] = 0.8 * heat_map[var][0] + 0.2 * heat_map[var][2]
        heat_map_list = [(var, heat_map[var]) for var in heat_map.keys()]
        heat_map_list.sort(key=sortErrorWeight)
        return heat_map_list

    def runAndCreateVarMap(self, encoding, datasetType, saveConfig=True):
        execMap = self.predict(encoding, datasetType, shadow=True)
        if saveConfig:
            self.configurationAccMap.append((self.variableToBitwidthMap, float(execMap['default'][0]))) 
        return self.createVarValueMap()
    
    def createVarValueMap(self):
        outputDir = os.path.join(config.tempdir, "Predictor")
        file = os.path.join(outputDir, "debugLog")
        f = open(file).read().split("**********")

        f = [data.split('\n') for data in f]
        f = [[line.lstrip().rstrip() for line in data] for data in f]

        varValueMap = {}
        varName = None
        for data in f:
            for line in data:
                if line == '':
                    continue
                elif line[:3] == 'tmp' or (line in self.decls):
                    varName = line
                else:
                    line = line.split(' ')
                    if varName in varValueMap.keys():
                        for value in line:
                            if value == '':
                                continue
                            varValueMap[varName].append(value)
                    else:
                        line = [val for val in line if val != '']
                        varValueMap[varName] = line
        return varValueMap


    # Reverse sort the accuracies, print the top 5 accuracies and return the
    # best scaling factor.
    def getBestScale(self):
        def getMaximisingMetricValue(a):
            if self.metric == config.Metric.accuracy:
                return (a[1][0], -a[1][1], -a[1][2]) if not config.higherOffsetBias else (a[1][0], -a[0])
            elif self.metric == config.Metric.disagreements:
                return (-a[1][1], -a[1][2], a[1][0]) if not config.higherOffsetBias else (-max(5, a[1][1]), -a[0])
            elif self.metric == config.Metric.reducedDisagreements:
                return (-a[1][2], -a[1][1], a[1][0]) if not config.higherOffsetBias else (-max(5, a[1][2]), -a[0])
            elif self.algo == config.Algo.test:
                # Minimize regression error.
                return (-a[1][0])

        x = [(i, self.accuracy[i]) for i in self.accuracy]
        x.sort(key=getMaximisingMetricValue, reverse=True)
        sorted_accuracy = x[:5]
        Util.getLogger().info(sorted_accuracy)
        return sorted_accuracy[0][0]

    # Find the scaling factor which works best on the training dataset and
    # predict on the testing dataset.
    def findBestConfiguration(self):
        print("-------------------------")
        print("Performing Exploration...")
        print("-------------------------\n")

        # Generate input files for training dataset.
        # Even Posit encoding will accept conversion to fixed encoding
        res = self.convert(self.encoding,
                           config.DatasetType.training, config.Target.x86)
        if res == False:
            return False

        # Search for the best scaling factor.
        res = self.performSearch()
        if res == False:
            return False

        return True

    # After exploration is completed, this function is invoked to show the performance of the final quantised code on a testing dataset,
    # which is ideally different than the training dataset on which the bitwidth and scale tuning was done.
    def runOnTestingDataset(self):
        print("\n-------------------------------")
        print("Prediction on testing dataset")
        print("-------------------------------\n")
        f = open("demotedVarsList", "w")
        newbitwidhts = {}
        for var in self.variableToBitwidthMap.keys():
            if self.variableToBitwidthMap[var] in [8, 12]:
                newbitwidhts[var] = self.variableToBitwidthMap[var]
        f.write(str(newbitwidhts))
        f.close()
        if self.encoding != config.Encoding.posit:
            Util.getLogger().debug("Setting max scaling factor to %d\n" % (self.sf))

        if config.vbwEnabled:
            Util.getLogger().debug("Demoted Vars with Offsets: %s\n" % (str(self.demotedVarsOffsets)))

        # Generate files for the testing dataset.
        res = self.convert(self.encoding,
                           config.DatasetType.testing, config.Target.x86)
        if res == False:
            return False

        # Compile and run code using the best scaling factor.
        if config.vbwEnabled:
            compiled = self.partialCompile(self.encoding, config.Target.x86, self.sf, True, None, 0, dict(self.variableToBitwidthMap), list(self.demotedVarsList), dict(self.demotedVarsOffsets))
        else:
            compiled = self.partialCompile(self.encoding, config.Target.x86, self.sf, True, None, 0)
        if compiled == False:
            return False

        res, exit = self.runAll(self.encoding, config.DatasetType.testing, {"default" : self.sf}, printAlso=True)
        if res == False:
            return False

        return True

    # This function is invoked before the exploration to obtain floating point accuracy, as well as profiling each variable
    # in the floating point code to compute their ranges and consequently their fixed-point ranges.
    def collectProfileData(self):
        Util.getLogger().info("-----------------------\n")
        Util.getLogger().info("Collecting profile data\n")
        Util.getLogger().info("-----------------------\n")

        res = self.convert(config.Encoding.floatt,
                           config.DatasetType.training, config.Target.x86)
        if res == False:
            return False

        res = self.compile(config.Encoding.floatt, config.Target.x86, self.sf)
        if res == False:
            return False

        execMap = self.predict(config.Encoding.floatt, config.DatasetType.training)
        if execMap == None:
            return False

        self.flAccuracy = execMap["default"][0]
        Util.getLogger().info("Accuracy is %.3f%%\n" % (execMap["default"][0]))
        Util.getLogger().info("Disagreement is %.3f%%\n" % (execMap["default"][1]))
        Util.getLogger().info("Reduced Disagreement is %.3f%%\n" % (execMap["default"][2]))

    # Generate code for Arduino.
    def compileFixedForTarget(self):
        print("------------------------------")
        print("Generating code for %s..." % (self.target))
        print("------------------------------\n")

        demotedVarsOffsets = dict(self.demotedVarsOffsets) if hasattr(self, 'demotedVarsOffsets') else {}
        variableToBitwidthMap = dict(self.variableToBitwidthMap) if hasattr(self, 'variableToBitwidthMap') else {}
        res = self.convert(config.Encoding.fixed,
                           config.DatasetType.testing, self.target, variableToBitwidthMap, demotedVarsOffsets)
        if res == False:
            return False

        # Copy files.
        if self.target == config.Target.arduino:
            srcFile = os.path.join(config.outdir, "input", "model_fixed.h")
            destFile = os.path.join(config.outdir, str(config.wordLength), self.algo, self.dataset, "model.h")
            os.makedirs(os.path.join(config.outdir, str(config.wordLength), self.algo, self.dataset), exist_ok=True)
        elif self.target == config.Target.m3:
            srcFile = os.path.join(config.outdir, "input", "model_fixed.h")
            destFile = os.path.join(config.outdir, "model.h")
            os.makedirs(os.path.join(config.outdir), exist_ok=True)
            shutil.copyfile(srcFile, destFile)
            srcFile = os.path.join(config.outdir, "input", "scales.h")
            destFile = os.path.join(config.outdir, "scales.h")
        shutil.copyfile(srcFile, destFile)

        # Copy library.h file.
        curr_dir = os.path.dirname(os.path.realpath(__file__))

        if self.target == config.Target.arduino:
            srcFile = os.path.join(curr_dir, self.target, "library", "library_fixed.h")
            destFile = os.path.join(config.outdir, "library.h")
            shutil.copyfile(srcFile, destFile)

        modifiedBitwidths = dict.fromkeys(self.variableToBitwidthMap.keys(), config.wordLength) if hasattr(self, 'variableToBitwidthMap') else {}
        if hasattr(self, 'demotedVarsList'):
            for i in self.demotedVarsList:
                modifiedBitwidths[i] = config.wordLength // 2
        res = self.partialCompile(config.Encoding.fixed, self.target, self.sf, True, None, 0, dict(modifiedBitwidths), list(self.demotedVarsList) if hasattr(self, 'demotedVarsList') else [], dict(demotedVarsOffsets))
        if res == False:
            return False

        return True

    def runEncoded(self):
        # Collect runtime profile.
        res = self.collectProfileData()
        if res == False:
            return False

        # Obtain best scaling factor.
        if self.sf == None:
            res = self.findBestConfiguration()
            if res == False:
                return False

        res = self.runOnTestingDataset()
        if res == False:
            return False
        else:
            self.testingAccuracy = self.accuracy[self.sf]

        # Generate code for target.
        if self.target != config.Target.x86:
            self.compileFixedForTarget()

            print("%s sketch dumped in the folder %s\n" % (self.target, config.outdir))

        return True

    # Generate Arduino floating point code.
    def compileFloatForTarget(self):
        assert self.target == config.Target.arduino, "Floating point code supported for Arduino only"

        print("------------------------------")
        print("Generating code for %s..." % (self.target))
        print("------------------------------\n")

        res = self.convert(config.Encoding.floatt,
                           config.DatasetType.testing, self.target)
        if res == False:
            return False

        res = self.compile(config.Encoding.floatt, self.target, self.sf)
        if res == False:
            return False

        # Copy model.h.
        srcFile = os.path.join(config.outdir, "Streamer", "input", "model_float.h")
        destFile = os.path.join(config.outdir, self.target, "model.h")
        shutil.copyfile(srcFile, destFile)

        # Copy library.h file.
        srcFile = os.path.join(config.outdir, self.target, "library", "library_float.h")
        destFile = os.path.join(config.outdir, self.target, "library.h")
        shutil.copyfile(srcFile, destFile)

        return True

    # Floating point x86 code.
    def runForFloat(self):
        Util.getLogger().info("---------------------------")
        Util.getLogger().info("Executing for X86 target...")
        Util.getLogger().info("---------------------------\n")

        res = self.convert(config.Encoding.floatt,
                           config.DatasetType.testing, config.Target.x86)
        if res == False:
            return False

        res = self.compile(config.Encoding.floatt, config.Target.x86, self.sf)
        if res == False:
            return False

        execMap = self.predict(config.Encoding.floatt, config.DatasetType.testing)
        if execMap == None:
            return False
        else:
            self.testingAccuracy = execMap["default"][0]

        print("Accuracy is %.3f%%\n" % (self.testingAccuracy))

        if self.target == config.Target.arduino:
            self.compileFloatForTarget()
            print("\nArduino sketch dumped in the folder %s\n" % (config.outdir))

        return True

    def run(self):
        sys.setrecursionlimit(10000)
        self.setup()

        if self.encoding == config.Encoding.floatt:
            return self.runForFloat()
        else:
            return self.runEncoded()

    def computMemoryUsage(self, vbwMap = None, extendedVBWMap = None):
        if vbwMap == None:
            vbwMap = dict(self.variableToBitwidthMap)
        else:
            vbwMap = dict(vbwMap)
        if extendedVBWMap != None:
            for var in extendedVBWMap.keys():
                if var in vbwMap.keys():
                    continue
                else:
                    vbwMap[var] = extendedVBWMap[var]
        return Util.computeScratchLocationsFirstFitPriority(self.decls, self.coLocatedVariables, self.varLiveIntervals, self.notScratch, vbwMap, self.floatConstants, self.intConstants, self.internalVars)