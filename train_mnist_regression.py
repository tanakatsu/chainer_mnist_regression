#!/usr/bin/env python
from __future__ import print_function
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers

from chainer import cuda
import numpy as np
import data
import six
import time
import csv

# code reference: http://hirotaka-hachiya.hatenablog.com/entry/2016/07/31/234817


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
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--initmodel', '-m', default=None,
                        help='Initialize the model from given file')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(MLP(args.unit), lossfun=F.mean_squared_error)
    model.compute_accuracy = False
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Init/Resume
    if args.initmodel:
        print('Load model from', args.initmodel)
        serializers.load_npz(args.initmodel, model)
    if args.resume:
        print('Load optimizer state from', args.resume)
        serializers.load_npz(args.resume, optimizer)

    print('load MNIST dataset')
    mnist = data.load_mnist_data()
    mnist['data'] = mnist['data'].astype(np.float32)
    mnist['data'] /= 255
    mnist['target'] = mnist['target'].astype(np.int32)
    mnist['target'] = mnist['target'].astype(np.float32).reshape(len(mnist['target']), 1)

    # Split data to train and test
    N = 60000
    x_train, x_test = np.split(mnist['data'],   [N])
    y_train, y_test = np.split(mnist['target'], [N])
    N_test = y_test.size

    # Setup gpu
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    xp = np if args.gpu < 0 else cuda.cupy

    # Training loop with epoch
    test_loss = []
    for epoch in six.moves.range(1, args.epoch + 1):
        print('epoch', epoch)

        # Training Loop with minibatch
        perm = np.random.permutation(N)
        sum_loss = 0
        start = time.time()
        for i in six.moves.range(0, N, args.batchsize):
            x = chainer.Variable(xp.asarray(x_train[perm[i:i + args.batchsize]]))
            t = chainer.Variable(xp.asarray(y_train[perm[i:i + args.batchsize]]))

            # Pass the loss function (Classifier defines it) and its arguments
            optimizer.update(model, x, t)

            sum_loss += float(model.loss.data) * len(t.data)

        # Compute throughput
        end = time.time()
        elapsed_time = end - start
        throughput = N / elapsed_time
        print('train mean loss={}, throughput={} images/sec'.format(
            sum_loss / N, throughput))

        # evaluation
        sum_loss = 0

        for i in six.moves.range(0, N_test, args.batchsize):
            x = chainer.Variable(xp.asarray(x_test[i:i + args.batchsize]),
                                 volatile='on')
            t = chainer.Variable(xp.asarray(y_test[i:i + args.batchsize]),
                                 volatile='on')
            # Compute loss and accuracy
            loss = model(x, t)

            sum_loss += float(loss.data) * len(t.data)

        print('test  mean loss={}'.format(
            sum_loss / N_test))

        # Record test loss
        test_loss.append([sum_loss / N_test])

    # Save model, optimizer, loss and accuracy
    print('save the model')
    fname = 'mnist_regression.model'
    serializers.save_npz(fname, model)

    print('save the optimizer')
    fname = 'mnist_regression.state'
    serializers.save_npz(fname, optimizer)

    print('save the losses')
    fname = 'mnist_regression_loss.csv'
    f = open(fname, 'w')
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['loss'])
    writer.writerows(test_loss)
    f.close()

if __name__ == '__main__':
    main()
