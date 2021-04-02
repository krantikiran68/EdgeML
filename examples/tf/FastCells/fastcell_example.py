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

from tensorflow.python.ops import math_ops
import numpy as np

class FastGRNNPredictor(tf.keras.Model):

    def __init__(self, model, FC, FCbias):
        super(FastGRNNPredictor, self).__init__(name='FastGRNNPredictor')
        self.model = model

        self.FC = FC
        self.FCbias = FCbias

    def call(self, x):
        H = tf.zeros_initializer()(shape=[1,100], dtype=tf.float32)
        
        cond = lambda i, x, H: i<99
        def while_body(i, x, H):

            x = tf.reshape(x, [1, 3168])
            XX = x[:, 32*i:32*(i+1)]
            XX = tf.reshape(XX, [1,32])
            a = math_ops.matmul( math_ops.matmul(XX, self.model.W1), self.model.W2)
            b = math_ops.matmul( math_ops.matmul(H, self.model.U1), self.model.U2)
            c = a + b
            g = math_ops.sigmoid( c + self.model.bias_gate)
            h = math_ops.tanh(c + self.model.bias_update)
            H = (g * H) + ((math_ops.sigmoid(self.model.zeta) * (1.0 - g)) + (math_ops.sigmoid(self.model.nu) * h))
            # H = tf.reshape(H, [1, 100])
            return i+1, x, H
        
        i = 0
        _, _, H = tf.while_loop(cond, while_body, loop_vars=[i, x, H])
        score = math_ops.matmul(H, self.FC) + self.FCbias

        return score # tf.math.argmax(score, output_type=tf.dtypes.int64)
            
        



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
    
   
    my_dir = currDir
    os.chdir(my_dir)
    idx = 0
    if FastCell._num_weight_matrices[0] == 1:
        # Vars.append(self.W)
        idx = idx + 1
        FastCell.W = FastCellTrainer.FastParams[0]
    else:
        # Vars.extend([self.W1, self.W2])
        idx = idx + 2
        FastCell.W1 = tf.convert_to_tensor(np.load("W1.npy"), dtype=tf.float32)# FastCellTrainer.FastParams[0]
        FastCell.W2 = tf.convert_to_tensor(np.load("W2.npy"), dtype=tf.float32)# FastCellTrainer.FastParams[1]


    if FastCell._num_weight_matrices[1] == 1:
        # Vars.append(self.U)
        idx = idx + 1
        FastCell.U = FastCellTrainer.FastParams[1]
    else:
        # Vars.extend([self.U1, self.U2])
        idx = idx + 2
        FastCell.U1 = tf.convert_to_tensor(np.load("U1.npy"), dtype=tf.float32)# FastCellTrainer.FastParams[2]
        FastCell.U2 = tf.convert_to_tensor(np.load("U2.npy"), dtype=tf.float32)# FastCellTrainer.FastParams[3]

    FastCell.bias_gate = tf.convert_to_tensor(np.load("Bg.npy"), dtype=tf.float32)# FastCellTrainer.FastParams[idx]
    FastCell.bias_update = tf.convert_to_tensor(np.load("Bh.npy"), dtype=tf.float32)# FastCellTrainer.FastParams[idx + 1]

    FastCell.zeta = tf.convert_to_tensor(np.load("zeta.npy"), dtype=tf.float32)# FastCellTrainer.FastParams[idx + 2]
    FastCell.nu = tf.convert_to_tensor(np.load("nu.npy"), dtype=tf.float32)# FastCellTrainer.FastParams[idx + 3]


    model = FastGRNNPredictor(FastCell, tf.convert_to_tensor(np.load("FC.npy"), dtype=tf.float32), tf.convert_to_tensor(np.load("FCbias.npy"), dtype=tf.float32))
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy())

    model.predict(x=np.float32(Xtest[0].reshape([1, 3168])), batch_size=1, verbose=1)

    model.save('model')

    converter = tf.lite.TFLiteConverter.from_saved_model('model')
    tflite_model = converter.convert()

    with open('../../../fastgrnn.tflite', 'wb') as f:
        f.write(tflite_model)   


if __name__ == '__main__':
    main()

