import glob
import math
import numpy as np

import chainer
from chainer import backend
from chainer import backends
from chainer.backends import cuda
from chainer import Function, FunctionNode, gradient_check, report, training, utils, Variable
from chainer import datasets, initializers, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList, Sequential
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import chainerx 

import argparse
import os
import warnings
import itertools
import sys

from PIL import Image

class Block(chainer.Chain):

    def __init__(self, n_feats, kernel_size, block_feats, res_scale=1, act=F.relu()):
        super(Block, self).__init__()
        self.res_scale = res_scale
        body = []
        body.append(
            F.BatchNormalization(F.Convolution2D(n_feats, block_feats, kernel_size, padding=kernel_size//2)))
        body.append(act)
        body.append(
            F.BatchNormalization(F.Convolution2D(block_feats, n_feats, kernel_size, padding=kernel_size//2)))

        self.body = chainer.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res
