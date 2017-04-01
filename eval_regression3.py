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

    def __init__(self, train=True):
        super(MLP, self).__init__(
            conv1=L.Convolution2D(1, 20, 5),
            conv2=L.Convolution2D(20, 50, 5),
            l1=L.Linear(800, 500),  # 50 * 4 * 4 = 800
            l2=L.Linear(500, 1),
        )
        self.train = train

    def __call__(self, x):
        # x = (1, 28, 28)
        # conv1, pool1
        # F=5, S=1, P=0, (28 - 5 + 0) / 1 + 1 = 24
        # (1, 28, 28) -> (20, 24, 24) # conv
        # (20, 24, 24) -> (20, 12, 12) # maxpool
        # conv2, pool2
        # F=5, S=1, P=0, (12 - 5 + 0) / 1 + 1 = 8
        # (20, 12, 12) -> (50, 8, 8) # conv
        # (50, 8, 8) -> (50, 4, 4) # maxpool

        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.dropout(F.relu(self.l1(h)), train=self.train)
        y = self.l2(h)
        return y


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
    model.train = False
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

        y = model.predictor(img.reshape(-1, 1, 28, 28))
        loss = F.mean_squared_error(y, chainer.Variable(np.array([[label]], dtype=np.float32)))
        print(label, y.data[0], loss.data)

if __name__ == '__main__':
    main()
