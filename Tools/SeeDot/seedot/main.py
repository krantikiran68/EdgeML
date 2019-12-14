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

from seedot.compiler.converter.converter import Converter

import seedot.config as config
from seedot.compiler.compiler import Compiler
from seedot.predictor import Predictor
import seedot.util as Util


class Main:

    def __init__(self, algo, version, target, trainingFile, testingFile, modelDir, sf, maximisingMetric):
        self.algo, self.version, self.target = algo, version, target
        self.trainingFile, self.testingFile, self.modelDir = trainingFile, testingFile, modelDir
        self.sf = sf
        self.accuracy = {}
        self.maximisingMetric = maximisingMetric

    def setup(self):
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        
        copy_tree(os.path.join(curr_dir, "Predictor"), os.path.join(config.tempdir, "Predictor"))

        for fileName in ["arduino.ino", "config.h", "predict.h"]:
            srcFile = os.path.join(curr_dir, "arduino", fileName)
            destFile = os.path.join(config.outdir, fileName)
            shutil.copyfile(srcFile, destFile)

    # Generate the fixed-point code using the input generated from the
    # Converter project
    def compile(self, version, target, sf, generateAllFiles=True, id=None, printSwitch=-1):
        print("Generating code...", end='')

        # Set input and output files
        inputFile = os.path.join(self.modelDir, "input.sd")
        profileLogFile = os.path.join(
            config.tempdir, "Predictor", "output", "float", "profile.txt")

        logDir = os.path.join(config.outdir, "output")
        os.makedirs(logDir, exist_ok=True)
        if version == config.Version.floatt:
            outputLogFile = os.path.join(logDir, "log-float.txt")
        else:
            outputLogFile = os.path.join(
                logDir, "log-fixed-" + str(abs(sf)) + ".txt")

        if target == config.Target.arduino:
            outputDir = os.path.join(config.outdir, "arduino")
        elif target == config.Target.x86:
            outputDir = os.path.join(config.tempdir, "Predictor")

        try:
            obj = Compiler(self.algo, version, target, inputFile, outputDir,
                           profileLogFile, sf, outputLogFile, generateAllFiles, id, printSwitch)
            obj.run()
        except:
            print("failed!\n")
            #traceback.print_exc()
            return False

        self.scaleForX = obj.scaleForX

        print("completed")
        return True

    # Run the converter project to generate the input files using reading the
    # training model
    def convert(self, version, datasetType, target):
        print("Generating input files for %s %s dataset..." %
              (version, datasetType), end='')

        # Create output dirs
        if target == config.Target.arduino:
            outputDir = os.path.join(config.outdir, "input")
            datasetOutputDir = outputDir
        elif target == config.Target.x86:
            outputDir = os.path.join(config.tempdir, "Predictor")
            datasetOutputDir = os.path.join(config.tempdir, "Predictor", "input")
        else:
            assert False

        os.makedirs(datasetOutputDir, exist_ok=True)
        os.makedirs(outputDir, exist_ok=True)

        inputFile = os.path.join(self.modelDir, "input.sd")

        try:
            obj = Converter(self.algo, version, datasetType, target,
                            datasetOutputDir, outputDir)
            obj.setInput(inputFile, self.modelDir,
                         self.trainingFile, self.testingFile)
            obj.run()
        except Exception as e:
            traceback.print_exc()
            return False

        print("done\n")
        return True

    # Build and run the Predictor project
    def predict(self, version, datasetType):
        outputDir = os.path.join("output", version)

        curDir = os.getcwd()
        os.chdir(os.path.join(config.tempdir, "Predictor"))

        obj = Predictor(self.algo, version, datasetType,
                        outputDir, self.scaleForX)
        execMap = obj.run()

        os.chdir(curDir)

        return execMap

    # Compile and run the generated code once for a given scaling factor
    def partialCompile(self, version, target, sf, generateAllFiles, id, printSwitch):
        res = self.compile(version, target, sf, generateAllFiles, id, printSwitch)
        if res == False:
            return False
        else:
            return True

    def runAll(self, version, datasetType, codeIdToScaleFactorMap):
        execMap = self.predict(version, datasetType)
        if execMap == None:
            return False, True

        for codeId, sf in codeIdToScaleFactorMap.items():
            self.accuracy[sf] = execMap[str(codeId)]
            print("Accuracy at scale factor %d is %.3f%%, Disagreement Count is %d, Reduced Disagreement Count is %d\n" % (sf, execMap[str(codeId)][0], execMap[str(codeId)][1], execMap[str(codeId)][2]))

        return True, False

    # Iterate over multiple scaling factors and store their accuracies
    def performSearch(self):
        start, end = config.maxScaleRange

        highestValidScale = start
        firstCompileSuccess = False
        while firstCompileSuccess == False:
            if highestValidScale == end:
                assert False, "Compilation not possible for any Scale Factor. Abort"
            firstCompileSuccess = self.partialCompile(config.Version.fixed, config.Target.x86, highestValidScale, True, None, 0)
            if firstCompileSuccess:
                break
            highestValidScale -= 1
            
        lowestValidScale = end + 1
        firstCompileSuccess = False
        while firstCompileSuccess == False:
            firstCompileSuccess = self.partialCompile(config.Version.fixed, config.Target.x86, lowestValidScale, True, None, 0)
            if firstCompileSuccess:
                break
            lowestValidScale += 1
            
        #Ignored
        self.partialCompile(config.Version.fixed, config.Target.x86, lowestValidScale, True, None, -1)

        # The iterator logic is as follows:
        # Search begins when the first valid scaling factor is found (runOnce returns True)
        # Search ends when the execution fails on a particular scaling factor (runOnce returns False)
        # This is the window where valid scaling factors exist and we
        # select the one with the best accuracy
        numCodes = highestValidScale - lowestValidScale + 1
        codeId = 0
        codeIdToScaleFactorMap = {}
        for i in range(highestValidScale, lowestValidScale - 1, -1):
            print("Testing with max scale factor of " + str(i))

            codeId += 1
            compiled = self.partialCompile(
                config.Version.fixed, config.Target.x86, i, False, codeId, -1 if codeId != numCodes else codeId)
            if compiled == False:
                return False
            codeIdToScaleFactorMap[codeId] = i

        res, exit = self.runAll(config.Version.fixed, config.DatasetType.training, codeIdToScaleFactorMap)

        if exit == True or res == False:
            return False

        print("\nSearch completed\n")
        print("----------------------------------------------")
        print("Best performing scaling factors with accuracy, disagreement, reduced disagreement:")

        self.sf = self.getBestScale()

        return True

    # Reverse sort the accuracies, print the top 5 accuracies and return the
    # best scaling factor
    def getBestScale(self):
        def getMaximisingMetricValue(a):
            if self.maximisingMetric == config.MaximisingMetric.accuracy:
                return a[1][0]
            elif self.maximisingMetric == config.MaximisingMetric.disagreements:
                return -a[1][1]
            elif self.maximisingMetric == config.MaximisingMetric.reducedDisagreements:
                return -a[1][2]
        x = [(i, self.accuracy[i]) for i in self.accuracy]
        x.sort(key=getMaximisingMetricValue, reverse=True)
        sorted_accuracy = x[:5]
        print(sorted_accuracy)
        return sorted_accuracy[0][0]

    # Find the scaling factor which works best on the training dataset and
    # predict on the testing dataset
    def findBestScalingFactor(self):
        print("-------------------------------------------------")
        print("Performing search to find the best scaling factor")
        print("-------------------------------------------------\n")

        # Generate input files for training dataset
        res = self.convert(config.Version.fixed,
                           config.DatasetType.training, config.Target.x86)
        if res == False:
            return False

        # Search for the best scaling factor
        res = self.performSearch()
        if res == False:
            return False

        print("Best scaling factor = %d" % (self.sf))

        return True

    def runOnTestingDataset(self):
        print("\n-------------------------------")
        print("Prediction on testing dataset")
        print("-------------------------------\n")

        print("Setting max scaling factor to %d\n" % (self.sf))

        # Generate files for the testing dataset
        res = self.convert(config.Version.fixed,
                           config.DatasetType.testing, config.Target.x86)
        if res == False:
            return False

        # Compile and run code using the best scaling factor
        compiled = self.partialCompile(
            config.Version.fixed, config.Target.x86, self.sf, True, None, 0)
        if compiled == False:
            return False
            
        res, exit = self.runAll(config.Version.fixed, config.DatasetType.testing, {"default" : self.sf})
        if res == False:
            return False

        return True

    # Generate files for training dataset and perform a profiled execution
    def collectProfileData(self):
        print("-----------------------")
        print("Collecting profile data")
        print("-----------------------")

        res = self.convert(config.Version.floatt,
                           config.DatasetType.training, config.Target.x86)
        if res == False:
            return False

        res = self.compile(config.Version.floatt, config.Target.x86, self.sf)
        if res == False:
            return False

        execMap = self.predict(config.Version.floatt, config.DatasetType.training)
        if execMap == None:
            return False

        print("Accuracy is %.3f%%\n" % (execMap["default"][0]))

    # Generate code for Arduino
    def compileFixedForTarget(self):
        print("------------------------------")
        print("Generating code for %s..." % (self.target))
        print("------------------------------\n")

        res = self.convert(config.Version.fixed,
                           config.DatasetType.testing, self.target)
        if res == False:
            return False

        # Copy file
        srcFile = os.path.join(config.outdir, "input", "model_fixed.h")
        destFile = os.path.join(config.outdir, "model.h")
        shutil.copyfile(srcFile, destFile)

        # Copy library.h file
        srcFile = os.path.join(config.outdir, self.target, "library", "library_fixed.h")
        destFile = os.path.join(config.outdir, self.target, "library.h")
        shutil.copyfile(srcFile, destFile)

        res = self.compile(config.Version.fixed, self.target, self.sf)
        if res == False:
            return False

        return True

    def runForFixed(self):
        # Collect runtime profile
        res = self.collectProfileData()
        if res == False:
            return False

        # Obtain best scaling factor
        if self.sf == None:
            res = self.findBestScalingFactor()
            if res == False:
                return False

        res = self.runOnTestingDataset()
        if res == False:
            return False
        else:
            self.testingAccuracy = self.accuracy[self.sf]

        # Generate code for target
        if self.target == config.Target.arduino:
            self.compileFixedForTarget()

            print("\nArduino sketch dumped in the folder %s\n" % (config.outdir))

        return True

    def compileFloatForTarget(self):
        print("------------------------------")
        print("Generating code for %s..." % (self.target))
        print("------------------------------\n")

        res = self.convert(config.Version.floatt,
                           config.DatasetType.testing, self.target)
        if res == False:
            return False

        res = self.compile(config.Version.floatt, self.target, self.sf)
        if res == False:
            return False

        # Copy model.h
        srcFile = os.path.join(config.outdir, "Streamer", "input", "model_float.h")
        destFile = os.path.join(config.outdir, self.target, "model.h")
        shutil.copyfile(srcFile, destFile)

        # Copy library.h file
        srcFile = os.path.join(config.outdir, self.target, "library", "library_float.h")
        destFile = os.path.join(config.outdir, self.target, "library.h")
        shutil.copyfile(srcFile, destFile)

        return True

    def runForFloat(self):
        print("---------------------------")
        print("Executing for X86 target...")
        print("---------------------------\n")

        res = self.convert(config.Version.floatt,
                           config.DatasetType.testing, config.Target.x86)
        if res == False:
            return False

        res = self.compile(config.Version.floatt, config.Target.x86, self.sf)
        if res == False:
            return False

        execMap = self.predict(config.Version.floatt, config.DatasetType.testing)
        if execMap == None:
            return False
        else:
            self.testingAccuracy = execMap["default"][0]

        print("Accuracy is %.3f%%\n" % (acc))

        if self.target == config.Target.arduino:
            self.compileFloatForTarget()
            print("\nArduino sketch dumped in the folder %s\n" % (config.outdir))

        return True

    def run(self):

        sys.setrecursionlimit(10000)

        self.setup()

        if self.version == config.Version.fixed:
            return self.runForFixed()
        else:
            return self.runForFloat()
