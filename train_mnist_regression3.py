#!/usr/bin/env python
from __future__ import print_function
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset
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
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports mean sqaured error loss at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(MLP(args.unit), lossfun=F.mean_squared_error)
    model.compute_accuracy = False
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    print('load MNIST dataset')
    train, test = chainer.datasets.get_mnist()

    x_train = [x[0].reshape(1, 28, 28) for x in train]
    y_train = [x[1].astype(np.float32).reshape(1) for x in train]  # scalar -> (1,)
    x_test = [x[0].reshape(1, 28, 28) for x in test]
    y_test = [x[1].astype(np.float32).reshape(1) for x in test]  # scalar -> (1,)

    train = tuple_dataset.TupleDataset(x_train, y_train)
    test = tuple_dataset.TupleDataset(x_test, y_test)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'), trigger=(args.epoch, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

if __name__ == '__main__':
    main()
