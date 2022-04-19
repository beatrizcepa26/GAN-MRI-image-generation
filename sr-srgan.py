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

    def __init__(self):
        super(Block, self).__init__()
        self.res_scale = res_scale
        block = []

        block.append(
            F.BatchNormalization(L.Convolution2D(64, 64, 3, stride=1)))
        block.append(F.prelu)
        block.append(
            F.BatchNormalization(L.Convolution2D(64, 64, 3, stride=1)))

        self.block = chainer.Sequential(*block)

    def forward(self, x):
        res = self.block(x)
        outblock = F.add(res,x)
        return outblock




class MODEL(chainer.Chain):
    def __init__(self, args):
        super(MODEL, self).__init__()
        # hyper-params
        self.args = args
        scale = args.scale[0]
        n_resblocks = args.n_resblocks


        # HEAD
        head = []
        head.append(L.Convolution2D(1, 64, 9, stride=1))
        head.append(F.prelu)



        # BODY
        body = []
        for i in range(n_resblocks):
            body.append(Block())
        body.append(
            F.BatchNormalization(L.Convolution2D(64, 64, 3, stride=1)))



        # TAIL
        tail = []
        tail.append(
            F.BatchNormalization(L.Convolution2D(64, 256, 3, stride=1)))
        tail.append(F.depth2space(2))
        tail.append(F.prelu)
        tail.append(
            F.BatchNormalization(L.Convolution2D(256, 256, 3, stride=1)))
        tail.append(F.depth2space(2))
        tail.append(F.prelu)
        tail.append(L.Convolution2D(256, 1, 9, stride=1))



        # make object members
        self.head = chainer.Sequential(*head)
        self.body = chainer.Sequential(*body)
        self.tail = chainer.Sequential(*tail)
    

    def forward(self, lr):
        out1 = self.head(lr)
        out2 = self.body(out1)
        outbody = F.add(out1, out2)
        sr = self.tail(outbody)
        
        return sr



class Updater(chainer.training.updaters.StandardUpdater):
    
    def __init__(self, *args, **kwargs):
        self.imgSR = kwargs.pop('model') 
        super(Updater, self).__init__(*args, **kwargs)

        
    def loss_func(self, imgSR, x_real, sr_img):

        # content loss (L1 or MAE)
        
        content_loss = F.mean_absolute_error()
        loss = content_loss(original_img, sr_img)
        
        chainer.report({'loss': loss}, imgSR)
        
        return loss


    def update_core(self):

        sr_optimizer = self.get_optimizer('imgSR')
        
        batch = self.get_iterator('main').next()
        
        device = self.device
            
        x_real = Variable(self.converter(batch, device)) / 255. 

        imgSR = self.imgSR
        batchsize = len(batch)
        
        sr_img = imgSR(x)

        sr_optimizer.update(self.lossfunc, imgSR, x_real, sr_img)



def out__generated_image(imgSR, dst):
    @chainer.training.make_extension() # make a new extension
    
    def make_image(trainer):
        
        with chainer.using_config('train', False):
            x = imgSR()
        
        x = chainer.backends.cuda.to_cpu(x.array)

        preview_dir = '{}/preview'.format(dst) # '%s/preview' %dst
        preview_path = preview_dir +\
            '/image{:0>8}.dcm'.format(trainer.updater.iteration) # DICOM image
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(x).save(preview_path)
    return make_image



def main():
    parser = argparse.ArgumentParser(description='Img SR model')
    parser.add_argument('--batchsize', '-b', type=int, default=50,
                        help='Number of images in each mini-batch')
    parser.add_argument('--communicator', type=str,
                        default='pure_nccl', help='Type of communicator')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', action='store_true',
                        help='Use GPU')
    parser.add_argument('--dataset', '-i',
                        help='Directory of image files')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    
    parser.add_argument('--n_resblocks', '-n', type=int,
                        help='Number of residual blocks on the body)')

    parser.add_argument('--sr_model', '-r', default='',
                        help='Use pre-trained sr model for training')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    args = parser.parse_args()

    
    
# Prepare ChainerMN communicator
if args.gpu:
    if args.communicator == 'naive':
        print('Error: \'naive\' communicator does not support GPU.\n')
        exit(-1)
    
    #chainermn.create_communicator() creates a communicator (is in charge of communication between workers)
    comm = chainermn.create_communicator(args.communicator)

    '''Workers in a node have to use different GPUs. For this purpose, intra_rank property of communicators is useful. 
    Each worker in a node is assigned a unique intra_rank starting from zero.'''
    device = comm.intra_rank
else:
    if args.communicator != 'naive':
        print('Warning: using naive communicator '
                'because only naive supports CPU-only execution')
    comm = chainermn.create_communicator('naive')
    device = -1


    if comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(comm.size))
        if args.gpu:
            print('Using GPUs')
        print('Using {} communicator'.format(args.communicator))
        print('Num hidden unit: {}'.format(args.n_hidden))
        print('Num Minibatch-size: {}'.format(args.batchsize))
        print('Num epoch: {}'.format(args.epoch))
        print('==========================================')


# Set up a neural network to train
imgSR = MODEL(args=args)


if device >= 0:
    # Make a specified GPU current
    chainer.cuda.get_device_from_id(device).use()
    imgSR.to_gpu()  # Copy the model to the GPU


# Setup an optimizer
def make_optimizer(model, comm, alpha=0.0002, beta1=0.5):

    # Create a multi node optimizer from a standard Chainer optimizer.

    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.Adam(alpha=alpha, beta1=beta1), comm)

    optimizer.setup(model)
    
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
    return optimizer
    

# make an optimizer the model
opt_SR = make_optimizer(imgSR, comm)


# Split and distribute the dataset. Only worker 0 loads the whole dataset.
# Datasets of worker 0 are evenly split and distributed to all workers.
if comm.rank == 0:
    if args.dataset == '':
        print('Provide a valid directory of input images')
    else:
        # 256x256 -> Users\user\Desktop\imgs\1
        all_files = os.listdir(args.dataset)
        image_files = [f for f in all_files if ('dcm' in f)] # DICOM images
        print('{} contains {} image files'
                .format(args.dataset, len(image_files)))
        train = chainer.datasets\
            .ImageDataset(paths=image_files, root=args.dataset)
else:
    train = None

train = chainermn.scatter_dataset(train, comm)

# Setup an iterator
train_iter = chainer.iterators.SerialIterator(train, args.batchsize)


# Setup an updater
updater = Updater(
    model=imgSR,
    iterator=train_iter,
    optimizer=sr_optimizer,
    device=device)

# Setup a trainer
trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)


'''Some display and output extensions are necessary only for one worker, otherwise, there would just be repeated outputs'''
if comm.rank == 0:
    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    
    
    trainer.extend(extensions.snapshot_object(
        imgSR, 'imgSR_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)

    trainer.extend(extensions.LogReport(trigger=display_interval))

    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'imgSR/loss', 'elapsed_time',
    ]), trigger=display_interval)

    trainer.extend(extensions.ProgressBar(update_interval=10))
    
    
    trainer.extend(
        out_generated_image(imgSR, args.out),
        trigger=snapshot_interval)

# Start the training using a pre-trained model, saved by snapshot_object
if args.sr_model:
    chainer.serializers.load_npz(args.sr_model, imgSR)


# Run the training
trainer.run()

if __name__ == '__main__':
    main()