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

from PIL import Image



def add_noise(device, h, sigma=0.2):
    # Add some noise to every intermediate outputs of D before giving them to the next layers
    
    if chainer.config.train: 
        xp = device.xp 
       
        # TODO(niboshi): Support random.randn in ChainerX (?)
        
        if device.xp is chainerx:
            fallback_device = device.fallback_device
            with chainer.using_device(fallback_device): 
                randn = device.send(fallback_device.xp.random.randn(*h.shape))
        else:
            randn = xp.random.randn(*h.shape)
        return h + sigma * randn
    else:
        return h



class Generator(chainer.Chain):
    
    def __init__(self, n_hidden, bottom_width=4, ch=1024, wscale=0.02):
        
        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.ch = ch 
        self.bottom_width = bottom_width 

        with self.init_scope(): 
            w = chainer.initializers.Normal(wscale)
            
            model= chainer.Sequential(
                L.Linear(self.n_hidden, bottom_width * bottom_width * ch,
                               initialW=w),
                L.BatchNormalization(bottom_width * bottom_width * ch)
                F.relu,
                F.reshape(len(z), self.ch, self.bottom_width, self.bottom_width),
                L.Deconvolution2D(ch, ch // 2, 4, 2, 1, initialW=w),
                L.BatchNormalization(ch // 2),
                F.relu,
                L.Deconvolution2D(ch // 2, ch // 4, 4, 2, 1, initialW=w),
                L.BatchNormalization(ch // 4),
                F.relu,
                L.Deconvolution2D(ch // 4, ch // 8, 4, 2, 1, initialW=w),
                L.BatchNormalization(ch // 8),
                F.relu,
                L.Deconvolution2D(ch // 8, 1, 3, 1, 1, initialW=w), # out_channels = 1 -> grayscale
                F.tanh            
            )

    
    
    def make_hidden(self, batchsize):
        
        dtype = chainer.get_dtype() 
        return np.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1))\
            .astype(dtype)

    
    
    def forward(self, z):
        # in this function, each layer is called and followed by relu, except the last layer (tanh)
        x = model(z)

        return x



class Discriminator(chainer.Chain):

    def __init__(self, bottom_width=4, ch=1024, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(Discriminator, self).__init__()
        
        with self.init_scope():
        
            self.c0_0 = L.Convolution2D(1, ch // 8, 3, 1, 1, initialW=w) # in_channels = 1 -> grayscale
            self.c0_1 = L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w)
            self.c1_0 = L.Convolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w)
            self.c1_1 = L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w)
            self.c2_0 = L.Convolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w)
            self.c2_1 = L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w)
            self.c3_0 = L.Convolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w)
            
            self.l4 = L.Linear(bottom_width * bottom_width * ch, 1, initialW=w)
            
            self.bn0_1 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn1_0 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn1_1 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn2_0 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn2_1 = L.BatchNormalization(ch // 1, use_gamma=False)
            self.bn3_0 = L.BatchNormalization(ch // 1, use_gamma=False)

    def forward(self, x):
        device = self.device
        h = add_noise(device, x)
        h = F.leaky_relu(add_noise(device, self.c0_0(h)))
        h = F.leaky_relu(add_noise(device, self.bn0_1(self.c0_1(h))))
        h = F.leaky_relu(add_noise(device, self.bn1_0(self.c1_0(h))))
        h = F.leaky_relu(add_noise(device, self.bn1_1(self.c1_1(h))))
        h = F.leaky_relu(add_noise(device, self.bn2_0(self.c2_0(h))))
        h = F.leaky_relu(add_noise(device, self.bn2_1(self.c2_1(h))))
        h = F.leaky_relu(add_noise(device, self.bn3_0(self.c3_0(h))))
        return self.l4(h)




class DCGANUpdater(chainer.training.updaters.StandardUpdater):
    
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models') 
        super(DCGANUpdater, self).__init__(*args, **kwargs)

        
    def loss_dis(self, dis, y_fake, y_real):
        
        batchsize = len(y_fake)
        
        L1 = F.sum(F.softplus(-y_real)) / batchsize # loss of the real samples
        L2 = F.sum(F.softplus(y_fake)) / batchsize # loss of the synthetic samples
        loss = L1 + L2
        
        chainer.report({'loss': loss}, dis)
        
        return loss

    
    
    def loss_gen(self, gen, y_fake):        
        batchsize = len(y_fake)
        loss = F.sum(F.softplus(-y_fake)) / batchsize
        chainer.report({'loss': loss}, gen)
        return loss

    
    
    def update_core(self):
        
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')
        
        batch = self.get_iterator('main').next()
        
        
        device = self.device
        
        x_real = Variable(self.converter(batch, device)) / 255. 

        gen, dis = self.gen, self.dis
        batchsize = len(batch)
        
        y_real = dis(x_real)

        z = Variable(device.xp.asarray(gen.make_hidden(batchsize)))
        
        x_fake = gen(z)
        
        y_fake = dis(x_fake)
    
        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
        gen_optimizer.update(self.loss_gen, gen, y_fake)




def out_generated_image(gen, dis, rows, cols, seed, dst):
    @chainer.training.make_extension() # make a new extension
    
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        xp = gen.xp 
        z = Variable(xp.asarray(gen.make_hidden(n_images)))
        
        with chainer.using_config('train', False):
            x = gen(z)
        
        x = chainer.backends.cuda.to_cpu(x.array)
        
        np.random.seed()

        x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
    
        _, _, H, W = x.shape
        
        x = x.reshape((rows, cols, 1, H, W)) # 1 colour channel -> grayscale
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((rows * H, cols * W, 1)) # 1 colour channel -> grayscale

        preview_dir = '{}/preview'.format(dst) # '%s/preview' %dst
        preview_path = preview_dir +\
            '/image{:0>8}.dcm'.format(trainer.updater.iteration) # DICOM image
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(x).save(preview_path)
    return make_image





def main():
    parser = argparse.ArgumentParser(description='ChainerMN example: DCGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=50,
                        help='Number of images in each mini-batch')
    parser.add_argument('--communicator', type=str,
                        default='pure_nccl', help='Type of communicator')
    parser.add_argument('--epoch', '-e', type=int, default=1000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', action='store_true',
                        help='Use GPU')
    parser.add_argument('--dataset', '-i', default='',
                        help='Directory of image files. Default is cifar-10.')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--gen_model', '-r', default='',
                        help='Use pre-trained generator for training')
    parser.add_argument('--dis_model', '-d', default='',
                        help='Use pre-trained discriminator for training')
    parser.add_argument('--n_hidden', '-n', type=int, default=100,
                        help='Number of hidden units (z)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
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

   

    # Set up a neural network to train -> making the instances of the generator and the discriminator
    gen = Generator(n_hidden=args.n_hidden)
    dis = Discriminator()


    if device >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(device).use()
        gen.to_gpu()  # Copy the model to the GPU
        dis.to_gpu()

   

    # Setup an optimizer
    def make_optimizer(model, comm, alpha=0.0002, beta1=0.5):

        # Create a multi node optimizer from a standard Chainer optimizer.

        optimizer = chainermn.create_multi_node_optimizer(
            chainer.optimizers.Adam(alpha=alpha, beta1=beta1), comm)

        optimizer.setup(model)
        
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer
        
    
    # make an optimizer for each model
    opt_gen = make_optimizer(gen, comm)
    opt_dis = make_optimizer(dis, comm)

    
    # Split and distribute the dataset. Only worker 0 loads the whole dataset.
    # Datasets of worker 0 are evenly split and distributed to all workers.
    if comm.rank == 0:
        if args.dataset == '':
            train, _ = chainer.datasets.get_cifar10(withlabel=False,
                                                    scale=255.)
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
    updater = DCGANUpdater(
        models=(gen, dis),
        iterator=train_iter,
        optimizer={
            'gen': opt_gen, 'dis': opt_dis},
        device=device)

    
    
    # Setup a trainer
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    
   
   
    '''Some display and output extensions are necessary only for one worker, otherwise, there would just be repeated outputs'''
    if comm.rank == 0:
        snapshot_interval = (args.snapshot_interval, 'iteration')
        display_interval = (args.display_interval, 'iteration')
        
        
        trainer.extend(extensions.snapshot_object(
            gen, 'gen_iter_{.updater.iteration}.npz'),
            trigger=snapshot_interval)
        trainer.extend(extensions.snapshot_object(
            dis, 'dis_iter_{.updater.iteration}.npz'),
            trigger=snapshot_interval)

        trainer.extend(extensions.LogReport(trigger=display_interval))

        trainer.extend(extensions.PrintReport([
            'epoch', 'iteration', 'gen/loss', 'dis/loss', 'elapsed_time',
        ]), trigger=display_interval)

        trainer.extend(extensions.ProgressBar(update_interval=10))
        
        
        trainer.extend(
            out_generated_image(
                gen, dis,
                10, 10, args.seed, args.out),
            trigger=snapshot_interval)



    # Start the training using pre-trained model, saved by snapshot_object
    if args.gen_model:
        chainer.serializers.load_npz(args.gen_model, gen)
    if args.dis_model:
        chainer.serializers.load_npz(args.dis_model, dis)
   
    
    # Run the training
    trainer.run()

if __name__ == '__main__':
    main()