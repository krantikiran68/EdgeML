# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import argparse
import datetime
from itertools import product
import json
import numpy as np
import os
import shutil
import tempfile

import seedot.common as common
import seedot.main as main
import seedot.util as util

import seedot.compiler.converter.converter as Converter
import seedot.compiler.converter.bonsai as Bonsai
import seedot.compiler.converter.protonn as Protonn


class Dataset:
    common = ["cifar-binary", "cr-binary", "cr-multiclass", "curet-multiclass",
              "letter-multiclass", "mnist-binary", "mnist-multiclass",
              "usps-binary", "usps-multiclass", "ward-binary"]
    extra = ["cifar-multiclass", "eye-binary", "dsa", "farm-beats",
             "interactive-cane", "spectakom", "usps10", "whale-binary"]
    default = common
    all = common + extra


class MainDriver:

    def __init__(self):
        self.driversAll = ["compiler", "converter", "predictor"]

    def parseArgs(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("-a", "--algo", choices=common.Algo.all,
                            default=common.Algo.default, metavar='', help="Algorithm to run")
        parser.add_argument("-v", "--version", choices=common.Version.all,
                            default=common.Version.all, metavar='', help="Floating point code or fixed point code")
        parser.add_argument("-d", "--dataset", choices=Dataset.all,
                            default=Dataset.default, metavar='', help="Dataset to run")

        parser.add_argument("-dt", "--datasetType", choices=common.DatasetType.all, default=[
                            common.DatasetType.default], metavar='', help="Training dataset or testing dataset")
        parser.add_argument("-t", "--target", choices=common.Target.all, default=[
                            common.Target.default], metavar='', help="Desktop code or Arduino code or Fpga HLS code")

        parser.add_argument("-sf", "--max-scale-factor", type=int,
                            metavar='', help="Max scaling factor for code generation")
        parser.add_argument("--load-sf", action="store_true",
                            help="Verify the accuracy of the generated code")

        parser.add_argument("--convert", action="store_true",
                            help="Pass through the converter")

        parser.add_argument("--tempdir", metavar='',
                            help="Scratch directory for intermediate files")
        parser.add_argument("-o", "--outdir", metavar='',
                            help="Directory to output the generated Arduino sketch")

        parser.add_argument("--driver", choices=self.driversAll,
                            metavar='', help="Driver to use")

        self.args = parser.parse_args()

        if not isinstance(self.args.algo, list):
            self.args.algo = [self.args.algo]
        if not isinstance(self.args.version, list):
            self.args.version = [self.args.version]
        if not isinstance(self.args.dataset, list):
            self.args.dataset = [self.args.dataset]
        if not isinstance(self.args.datasetType, list):
            self.args.datasetType = [self.args.datasetType]
        if not isinstance(self.args.target, list):
            self.args.target = [self.args.target]

        if self.args.tempdir is not None:
            assert os.path.isdir(
                self.args.tempdir), "Scratch directory doesn't exist"
            common.tempdir = self.args.tempdir
        else:
            # common.tempdir = os.path.join(tempfile.gettempdir(
            # ), "SeeDot", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            common.tempdir = "temp"
            if os.path.exists(common.tempdir):
                shutil.rmtree(common.tempdir)
            os.makedirs(common.tempdir, exist_ok=True)

        if self.args.outdir is not None:
            assert os.path.isdir(
                self.args.outdir), "Output directory doesn't exist"
            common.outdir = self.args.outdir
        else:
            common.outdir = os.path.join(common.tempdir, "arduino")
            os.makedirs(common.outdir, exist_ok=True)

    def checkMSBuildPath(self):
        found = False
        for path in common.msbuildPathOptions:
            if os.path.isfile(path):
                found = True
                common.msbuildPath = path

        if not found:
            raise Exception("Msbuild.exe not found at the following locations:\n%s\nPlease change the path and run again" % (
                common.msbuildPathOptions))

    def setGlobalFlags(self):
        np.seterr(all='warn')

    def run(self):
        if util.windows():
            self.checkMSBuildPath()

        self.setGlobalFlags()

        if self.args.driver is None:
            self.runMainDriver()
        elif self.args.driver == "compiler":
            self.runCompilerDriver()
        elif self.args.driver == "converter":
            self.runConverterDriver()
        elif self.args.driver == "predictor":
            self.runPredictorDriver()

    def runMainDriver(self):

        results = self.loadResultsFile()

        for iter in product(self.args.algo, self.args.version, self.args.dataset, self.args.target):
            algo, version, dataset, target = iter

            print("\n========================================")
            print("Executing on %s %s %s %s" %
                  (algo, version, dataset, target))
            print("========================================\n")

            if self.args.convert:
                datasetDir = os.path.join(
                    "..", "datasets", "datasets", dataset)
                modelDir = os.path.join("..", "model", dataset)

                if algo == common.Algo.bonsai:
                    modelDir = os.path.join(
                        modelDir, "BonsaiResults", "Params")
                elif algo == common.Algo.lenet:
                    modelDir = os.path.join(modelDir, "LenetModel")
                else:
                    modelDir = os.path.join(modelDir, "ProtoNNResults")

                trainingInput = os.path.join(datasetDir, "training-full.tsv")
                testingInput = os.path.join(datasetDir, "testing.tsv")

                datasetOutputDir = os.path.join(
                    "temp", "dataset-processed", algo, dataset)
                modelOutputDir = os.path.join(
                    "temp", "model-processed", algo, dataset)

                os.makedirs(datasetOutputDir, exist_ok=True)
                os.makedirs(modelOutputDir, exist_ok=True)

                if algo == common.Algo.bonsai:
                    obj = bonsai(trainingInput, testingInput,
                                 modelDir, datasetOutputDir, modelOutputDir)
                    obj.run()
                elif algo == common.Algo.protonn:
                    obj = protonn(trainingInput, testingInput,
                                  modelDir, datasetOutputDir, modelOutputDir)
                    obj.run()

                trainingInput = os.path.join(datasetOutputDir, "train.npy")
                testingInput = os.path.join(datasetOutputDir, "test.npy")
                modelDir = modelOutputDir
            else:
                datasetDir = os.path.join(
                    "..", "datasets", algo, dataset)

                trainingInput = os.path.join(datasetDir, "train.npy")
                testingInput = os.path.join(datasetDir, "test.npy")
                modelDir = os.path.join("..", "model", algo, dataset)

            try:
                if version == common.Version.floatt:
                    key = 'float32'
                elif common.wordLength == 8:
                    key = 'int8'
                elif common.wordLength == 16:
                    key = 'int16'
                elif common.wordLength == 32:
                    key = 'int32'
                else:
                    assert False

                curr = results[algo][key][dataset]

                expectedAcc = curr['accuracy']
                if version == common.Version.fixed:
                    bestScale = curr['sf']
                else:
                    bestScale = results[algo]['int16'][dataset]['sf']

            except Exception as _:
                assert self.args.load_sf == False
                expectedAcc = 0

            if self.args.load_sf:
                sf = bestScale
            else:
                sf = self.args.max_scale_factor

            obj = main.Main(algo, version, target, trainingInput,
                       testingInput, modelDir, sf)
            obj.run()

            acc = obj.testingAccuracy
            if acc != expectedAcc:
                print("FAIL: Expected accuracy %f%%" % (expectedAcc))
                #return
            elif version == common.Version.fixed and obj.sf != bestScale:
                print("FAIL: Expected best scale %d" % (bestScale))
                #return
            else:
                print("PASS")

    def runCompilerDriver(self):
        for iter in product(self.args.algo, self.args.version, self.args.target):
            algo, version, target = iter

            print("\nGenerating code for " + algo + " " + target + "...")

            inputFile = os.path.join("input", algo + ".sd")
            #inputFile = os.path.join("input", algo + ".pkl")
            profileLogFile = os.path.join("input", "profile.txt")

            outputDir = os.path.join("output")
            os.makedirs(outputDir, exist_ok=True)

            outputFile = os.path.join(outputDir, algo + "-fixed.cpp")
            obj = main.Main(algo, version, target, inputFile, outputFile,
                       profileLogFile, self.args.max_scale_factor, self.args.workers)
            obj.run()

    def runConverterDriver(self):
        for iter in product(self.args.algo, self.args.version, self.args.dataset, self.args.datasetType, self.args.target):
            algo, version, dataset, datasetType, target = iter

            print("\nGenerating input files for \"" + algo + " " + version +
                  " " + dataset + " " + datasetType + " " + target + "\"...")

            outputDir = os.path.join(
                "Converter", "output", algo + "-" + version + "-" + datasetType, dataset)
            os.makedirs(outputDir, exist_ok=True)

            datasetDir = os.path.join("..", "datasets", "datasets", dataset)
            modelDir = os.path.join("..", "model", dataset)

            if algo == common.Algo.bonsai:
                modelDir = os.path.join(modelDir, "BonsaiResults", "Params")
            elif algo == common.Algo.lenet:
                modelDir = os.path.join(modelDir, "LenetModel")
            else:
                modelDir = os.path.join(modelDir, "ProtoNNResults")

            trainingInput = os.path.join(datasetDir, "training-full.tsv")
            testingInput = os.path.join(datasetDir, "testing.tsv")

            obj = Converter(algo, version, datasetType, target,
                            outputDir, outputDir, self.args.workers)
            obj.setInput(modelDir, trainingInput, testingInput)
            obj.run()

    def runPredictorDriver(self):
        for iter in product(self.args.algo, self.args.version, self.args.dataset, self.args.datasetType):
            algo, version, dataset, datasetType = iter

            print("\nGenerating input files for \"" + algo + " " +
                  version + " " + dataset + " " + datasetType + "\"...")

            #outputDir = os.path.join("..", "Predictor", algo, version + "-testing")
            #datasetOutputDir = os.path.join("..", "Predictor", algo, version + "-" + datasetType)

            if version == common.Version.fixed:
                outputDir = os.path.join(
                    "..", "Predictor", "seedot_fixed", "testing")
                datasetOutputDir = os.path.join(
                    "..", "Predictor", "seedot_fixed", datasetType)
            elif version == common.Version.floatt:
                outputDir = os.path.join(
                    "..", "Predictor", self.algo + "_float", "testing")
                datasetOutputDir = os.path.join(
                    "..", "Predictor", self.algo + "_float", datasetType)

            os.makedirs(datasetOutputDir, exist_ok=True)
            os.makedirs(outputDir, exist_ok=True)

            datasetDir = os.path.join("..", "datasets", "datasets", dataset)
            modelDir = os.path.join("..", "model", dataset)

            if algo == common.Algo.bonsai:
                modelDir = os.path.join(modelDir, "BonsaiResults", "Params")
            elif algo == common.Algo.lenet:
                modelDir = os.path.join(modelDir, "LenetModel")
            else:
                modelDir = os.path.join(modelDir, "ProtoNNResults")

            trainingInput = os.path.join(datasetDir, "training-full.tsv")
            testingInput = os.path.join(datasetDir, "testing.tsv")

            obj = Converter(algo, version, datasetType, common.Target.x86,
                            datasetOutputDir, outputDir, self.args.workers)
            obj.setInput(modelDir, trainingInput, testingInput)
            obj.run()

            print("Building and executing " + algo + " " +
                  version + " " + dataset + " " + datasetType + "...")

            outputDir = os.path.join(
                "..", "Predictor", "output", algo + "-" + version)

            curDir = os.getcwd()
            os.chdir(os.path.join("..", "Predictor"))

            obj = Predictor(algo, version, datasetType, outputDir)
            acc = obj.run()

            os.chdir(curDir)

            if acc != None:
                print("Accuracy is %.3f" % (acc))

    def loadResultsFile(self):
        with open(os.path.join("Results", "Results.json")) as data:
            return json.load(data)


if __name__ == "__main__":
    obj = MainDriver()
    obj.parseArgs()
    obj.run()
