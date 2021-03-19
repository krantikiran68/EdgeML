# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import helpermethods
import tensorflow as tf
import numpy as np
import sys, os

from edgeml_tf.trainer.fastTrainer import FastTrainer
from edgeml_tf.graph.rnn import FastGRNNCell
from edgeml_tf.graph.rnn import FastRNNCell
from edgeml_tf.graph.rnn import UGRNNLRCell
from edgeml_tf.graph.rnn import GRULRCell
from edgeml_tf.graph.rnn import LSTMLRCell


def main():
    # Fixing seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    tf.compat.v1.disable_eager_execution()


    # Hyper Param pre-processing
    args = helpermethods.getArgs()

    dataDir = args.data_dir
    cell = args.cell
    inputDims = args.input_dim
    hiddenDims = args.hidden_dim

    totalEpochs = args.epochs
    learningRate = args.learning_rate
    outFile = args.output_file
    batchSize = args.batch_size
    decayStep = args.decay_step
    decayRate = args.decay_rate

    wRank = args.wRank
    uRank = args.uRank

    sW = args.sW
    sU = args.sU

    update_non_linearity = args.update_nl
    gate_non_linearity = args.gate_nl

    (dataDimension, numClasses, Xtrain, Ytrain, Xtest, Ytest,
     mean, std) = helpermethods.preProcessData(dataDir)

    assert dataDimension % inputDims == 0, "Infeasible per step input, " + \
        "Timesteps have to be integer"

    X = tf.compat.v1.placeholder(
        "float", [None, int(dataDimension / inputDims), inputDims])
    Y = tf.compat.v1.placeholder("float", [None, numClasses])

    currDir = helpermethods.createTimeStampDir(dataDir, cell)

    helpermethods.dumpCommand(sys.argv, currDir)
    helpermethods.saveMeanStd(mean, std, currDir)

    if cell == "FastGRNN":
        FastCell = FastGRNNCell(hiddenDims,
                                gate_non_linearity=gate_non_linearity,
                                update_non_linearity=update_non_linearity,
                                wRank=wRank, uRank=uRank)
    elif cell == "FastRNN":
        FastCell = FastRNNCell(hiddenDims,
                               update_non_linearity=update_non_linearity,
                               wRank=wRank, uRank=uRank)
    elif cell == "UGRNN":
        FastCell = UGRNNLRCell(hiddenDims,
                               update_non_linearity=update_non_linearity,
                               wRank=wRank, uRank=uRank)
    elif cell == "GRU":
        FastCell = GRULRCell(hiddenDims,
                             update_non_linearity=update_non_linearity,
                             wRank=wRank, uRank=uRank)
    elif cell == "LSTM":
        FastCell = LSTMLRCell(hiddenDims,
                              update_non_linearity=update_non_linearity,
                              wRank=wRank, uRank=uRank)
    else:
        sys.exit('Exiting: No Such Cell as ' + cell)

    FastCellTrainer = FastTrainer(
        FastCell, X, Y, sW=sW, sU=sU,
        learningRate=learningRate, outFile=outFile)

    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.global_variables_initializer())

    FastCellTrainer.train(batchSize, totalEpochs, sess, Xtrain, Xtest,
                          Ytrain, Ytest, decayStep, decayRate,
                          dataDir, currDir)

    
    idx = 0
    if FastCell._num_weight_matrices[0] == 1:
        # Vars.append(self.W)
        idx = idx + 1
        FastCell.W = FastCellTrainer.FastParams[0]
    else:
        # Vars.extend([self.W1, self.W2])
        idx = idx + 2
        FastCell.W1 = FastCellTrainer.FastParams[0]
        FastCell.W2 = FastCellTrainer.FastParams[1]


    if FastCell._num_weight_matrices[1] == 1:
        # Vars.append(self.U)
        idx = idx + 1
        FastCell.U = FastCellTrainer.FastParams[1]
    else:
        # Vars.extend([self.U1, self.U2])
        idx = idx + 2
        FastCell.U1 = FastCellTrainer.FastParams[2]
        FastCell.U2 = FastCellTrainer.FastParams[3]

    FastCell.bias_gate = FastCellTrainer.FastParams[idx]
    FastCell.bias_update = FastCellTrainer.FastParams[idx + 1]

    FastCell.zeta = FastCellTrainer.FastParams[idx + 2]
    FastCell.nu = FastCellTrainer.FastParams[idx + 3]

    model = tf.keras.Sequential(FastCell)

    tf.saved_model.save(model, 'model')


if __name__ == '__main__':
    main()
