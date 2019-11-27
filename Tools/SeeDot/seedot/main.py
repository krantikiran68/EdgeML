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

import seedot.common as Common
from seedot.compiler.compiler import Compiler
from seedot.predictor import Predictor
import seedot.util as Util


class Main:

    def __init__(self, algo, version, target, trainingFile, testingFile, modelDir, sf):
        self.algo, self.version, self.target = algo, version, target
        self.trainingFile, self.testingFile, self.modelDir = trainingFile, testingFile, modelDir
        self.sf = sf
        self.accuracy = {}

    def setup(self):
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        
        copy_tree(os.path.join(curr_dir, "Predictor"), os.path.join(Common.tempdir, "Predictor"))

        for fileName in ["arduino.ino", "config.h", "predict.h"]:
            srcFile = os.path.join(curr_dir, "arduino", fileName)
            destFile = os.path.join(Common.outdir, fileName)
            shutil.copyfile(srcFile, destFile)

    # Generate the fixed-point code using the input generated from the
    # Converter project
    def compile(self, version, target, sf):
        print("Generating code...", end='')

        # Set input and output files
        inputFile = os.path.join(self.modelDir, "input.sd")
        profileLogFile = os.path.join(
            Common.tempdir, "Predictor", "output", "float", "profile.txt")

        logDir = os.path.join(Common.outdir, "output")
        os.makedirs(logDir, exist_ok=True)
        if version == Common.Version.floatt:
            outputLogFile = os.path.join(logDir, "log-float.txt")
        else:
            outputLogFile = os.path.join(
                logDir, "log-fixed-" + str(abs(sf)) + ".txt")

        if target == Common.Target.arduino:
            outputDir = os.path.join(Common.outdir, "arduino")
        elif target == Common.Target.x86:
            outputDir = os.path.join(Common.tempdir, "Predictor")

        try:
            obj = Compiler(self.algo, version, target, inputFile, outputDir,
                           profileLogFile, sf, outputLogFile)
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
        if target == Common.Target.arduino:
            outputDir = os.path.join(Common.outdir, "input")
            datasetOutputDir = outputDir
        elif target == Common.Target.x86:
            outputDir = os.path.join(Common.tempdir, "Predictor")
            datasetOutputDir = os.path.join(Common.tempdir, "Predictor", "input")
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
        os.chdir(os.path.join(Common.tempdir, "Predictor"))

        obj = Predictor(self.algo, version, datasetType,
                        outputDir, self.scaleForX)
        acc = obj.run()

        os.chdir(curDir)

        return acc

    # Compile and run the generated code once for a given scaling factor
    def runOnce(self, version, datasetType, target, sf):
        res = self.compile(version, target, sf)
        if res == False:
            return False, False

        acc = self.predict(version, datasetType)
        if acc == None:
            return False, True

        self.accuracy[sf] = acc
        print("Accuracy is %.3f%%\n" % (acc))

        return True, False

    # Iterate over multiple scaling factors and store their accuracies
    def performSearch(self):
        start, end = Common.maxScaleRange
        searching = False

        for i in range(start, end, -1):
            print("Testing with max scale factor of " + str(i))

            res, exit = self.runOnce(
                Common.Version.fixed, Common.DatasetType.training, Common.Target.x86, i)

            if exit == True:
                return False

            # The iterator logic is as follows:
            # Search begins when the first valid scaling factor is found (runOnce returns True)
            # Search ends when the execution fails on a particular scaling factor (runOnce returns False)
            # This is the window where valid scaling factors exist and we
            # select the one with the best accuracy
            if res == True:
                searching = True
            elif searching == True:
                # break
                pass

        # If search didn't begin at all, something went wrong
        if searching == False:
            return False

        print("\nSearch completed\n")
        print("----------------------------------------------")
        print("Best performing scaling factors with accuracy:")

        self.sf = self.getBestScale()

        return True

    # Reverse sort the accuracies, print the top 5 accuracies and return the
    # best scaling factor
    def getBestScale(self):
        sorted_accuracy = dict(
            sorted(self.accuracy.items(), key=operator.itemgetter(1), reverse=True)[:5])
        print(sorted_accuracy)
        return next(iter(sorted_accuracy))

    # Find the scaling factor which works best on the training dataset and
    # predict on the testing dataset
    def findBestScalingFactor(self):
        print("-------------------------------------------------")
        print("Performing search to find the best scaling factor")
        print("-------------------------------------------------\n")

        # Generate input files for training dataset
        res = self.convert(Common.Version.fixed,
                           Common.DatasetType.training, Common.Target.x86)
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
        res = self.convert(Common.Version.fixed,
                           Common.DatasetType.testing, Common.Target.x86)
        if res == False:
            return False

        # Compile and run code using the best scaling factor
        res = self.runOnce(
            Common.Version.fixed, Common.DatasetType.testing, Common.Target.x86, self.sf)
        if res == False:
            return False

        return True

    # Generate files for training dataset and perform a profiled execution
    def collectProfileData(self):
        print("-----------------------")
        print("Collecting profile data")
        print("-----------------------")

        res = self.convert(Common.Version.floatt,
                           Common.DatasetType.training, Common.Target.x86)
        if res == False:
            return False

        res = self.compile(Common.Version.floatt, Common.Target.x86, self.sf)
        if res == False:
            return False

        acc = self.predict(Common.Version.floatt, Common.DatasetType.training)
        if acc == None:
            return False

        print("Accuracy is %.3f%%\n" % (acc))

    # Generate code for Arduino
    def compileFixedForTarget(self):
        print("------------------------------")
        print("Generating code for %s..." % (self.target))
        print("------------------------------\n")

        res = self.convert(Common.Version.fixed,
                           Common.DatasetType.testing, self.target)
        if res == False:
            return False

        # Copy file
        srcFile = os.path.join(Common.outdir, "input", "model_fixed.h")
        destFile = os.path.join(Common.outdir, "model.h")
        shutil.copyfile(srcFile, destFile)

        # Copy library.h file
        srcFile = os.path.join(Common.outdir, self.target, "library", "library_fixed.h")
        destFile = os.path.join(Common.outdir, self.target, "library.h")
        shutil.copyfile(srcFile, destFile)

        res = self.compile(Common.Version.fixed, self.target, self.sf)
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
        if self.target == Common.Target.arduino:
            self.compileFixedForTarget()

            print("\nArduino sketch dumped in the folder %s\n" % (Common.outdir))

        return True

    def compileFloatForTarget(self):
        print("------------------------------")
        print("Generating code for %s..." % (self.target))
        print("------------------------------\n")

        res = self.convert(Common.Version.floatt,
                           Common.DatasetType.testing, self.target)
        if res == False:
            return False

        res = self.compile(Common.Version.floatt, self.target, self.sf)
        if res == False:
            return False

        # Copy model.h
        srcFile = os.path.join(Common.outdir, "Streamer", "input", "model_float.h")
        destFile = os.path.join(Common.outdir, self.target, "model.h")
        shutil.copyfile(srcFile, destFile)

        # Copy library.h file
        srcFile = os.path.join(Common.outdir, self.target, "library", "library_float.h")
        destFile = os.path.join(Common.outdir, self.target, "library.h")
        shutil.copyfile(srcFile, destFile)

        return True

    def runForFloat(self):
        print("---------------------------")
        print("Executing for X86 target...")
        print("---------------------------\n")

        res = self.convert(Common.Version.floatt,
                           Common.DatasetType.testing, Common.Target.x86)
        if res == False:
            return False

        res = self.compile(Common.Version.floatt, Common.Target.x86, self.sf)
        if res == False:
            return False

        acc = self.predict(Common.Version.floatt, Common.DatasetType.testing)
        if acc == None:
            return False
        else:
            self.testingAccuracy = acc

        print("Accuracy is %.3f%%\n" % (acc))

        if self.target == Common.Target.arduino:
            self.compileFloatForTarget()
            print("\nArduino sketch dumped in the folder %s\n" % (Common.outdir))

        return True

    def run(self):

        sys.setrecursionlimit(10000)

        self.setup()

        if self.version == Common.Version.fixed:
            return self.runForFixed()
        else:
            return self.runForFloat()
