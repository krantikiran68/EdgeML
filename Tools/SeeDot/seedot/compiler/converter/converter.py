# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os

from seedot.compiler.converter.quantizer import *
from seedot.compiler.converter.util import *

import seedot.config as config

# Main file which sets the configurations and creates the corresponding object


class Converter:

    def __init__(self, algo, version, datasetType, target, datasetOutputDir, outputDir, varsForBitwidth={}, allScales={}):
        setAlgo(algo)
        setVersion(version)
        setDatasetType(datasetType)
        setTarget(target)

        # Set output directories
        setDatasetOutputDir(datasetOutputDir)
        setOutputDir(outputDir)

        self.sparseMatrixSizes = {}
        self.varsForBitwidth = varsForBitwidth
        self.allScales = allScales

    def setInput(self, inputFile, modelDir, trainingInput, testingInput):
        setInputFile(inputFile)
        setModelDir(modelDir)
        setDatasetInput(trainingInput, testingInput)

        self.inputSet = True

    def run(self):
        if self.inputSet != True:
            raise Exception("Set input paths before running Converter")

        if getVersion() == config.Version.fixed:
            obj = QuantizerFixed(self.varsForBitwidth, self.allScales)
        elif getVersion() == config.Version.floatt:
            obj = QuantizerFloat()

        obj.run()

        self.sparseMatrixSizes = obj.sparseMatSizes
