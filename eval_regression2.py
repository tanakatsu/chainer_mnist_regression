#!/usr/bin/env python
from __future__ import print_function
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers
import numpy as np


# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_units),  # n_units -> n_units
            l3=L.Linear(None, 1),  # n_units -> n_out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--model', '-m', default='result/model_iter_12000',
                        help='Trained model')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(MLP(args.unit))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    serializers.load_npz(args.model, model)

    # Load the MNIST dataset
    print('load MNIST dataset')
    train, test = chainer.datasets.get_mnist()

    # Predict
    for i in range(10):
        img = train[i][0]
        label = train[i][1]
        # img = test[i][0]
        # label = test[i][1]

        y = model.predictor(img.reshape(-1, 784))
        loss = F.mean_squared_error(y, chainer.Variable(np.array([[label]], dtype=np.float32)))
        print(label, y.data[0], loss.data)

if __name__ == '__main__':
    main()
