import numpy as np

import chainer
from chainer import backends
from chainer.backends import cuda
from chainer import Function, FunctionNode, gradient_check, report, training, utils, Variable
from chainer import datasets, initializers, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import chainermn

import argparse
import os
import warnings

from PIL import Image


# add some noise to every intermediate outputs of D before giving them to the next layers
def add_noise(h, sigma=0.2):
    xp = cuda.get_array_module(h.array)
    if chainer.config.train:
        return h + sigma * xp.random.randn(*h.shape)
    else:
        return h



# Generator architecture
class Generator(chainer.Chain):
    
    def __init__(self, n_hidden, bottom_width=4, ch=1024, wscale=0.02):

        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.ch = ch 
        self.bottom_width = bottom_width 

        with self.init_scope(): 
            w = chainer.initializers.Normal(wscale)

            self.l0 = L.Linear(self.n_hidden, bottom_width * bottom_width * ch,
                               initialW=w) 
        
            self.dc1 = L.Deconvolution2D(ch, ch // 2, 4, 2, 1, initialW=w) 
            self.dc2 = L.Deconvolution2D(ch // 2, ch // 4, 4, 2, 1, initialW=w) 
            self.dc3 = L.Deconvolution2D(ch // 4, ch // 8, 4, 2, 1, initialW=w) 
            self.dc4 = L.Deconvolution2D(ch // 8, ch // 16, 4, 2, 1, initialW=w) 
            self.dc5 = L.Deconvolution2D(ch // 16, ch // 32, 4, 2, 1, initialW=w)
            self.dc6 = L.Deconvolution2D(ch // 32, 3, 4, 2, 1, initialW=w)
            self.dc7 = L.Deconvolution2D(3, 3, 3, 1, 1, initialW=w)

            self.bn0 = L.BatchNormalization(bottom_width * bottom_width * ch)
            self.bn1 = L.BatchNormalization(ch // 2)
            self.bn2 = L.BatchNormalization(ch // 4)
            self.bn3 = L.BatchNormalization(ch // 8)
            self.bn4 = L.BatchNormalization(ch // 16)
            self.bn5 = L.BatchNormalization(ch // 32)
            self.bn6 = L.BatchNormalization(3)

    
    # uniform noise distribution Z fed to the Generator
    def make_hidden(self, batchsize):
        
        return np.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1))\
            .astype(np.float32)

    
    def forward(self, z):       
        
        h = F.reshape(self.bn0(self.l0(z)), 
                      (len(z), self.ch, self.bottom_width, self.bottom_width))
        h = F.leaky_relu(F.dropout(self.bn1(self.dc1(h)), 0.2), 0.2)
        h = F.leaky_relu(F.dropout(self.bn2(self.dc2(h)), 0.2), 0.2)
        h = F.leaky_relu(F.dropout(self.bn3(self.dc3(h)), 0.2), 0.2)
        h = F.leaky_relu(F.dropout(self.bn4(self.dc4(h)), 0.2), 0.2)
        h = F.leaky_relu(F.dropout(self.bn5(self.dc5(h)), 0.2), 0.2)
        h = F.leaky_relu(F.dropout(self.bn6(self.dc6(h)), 0.2), 0.2)
        x = F.tanh(F.dropout(self.dc7(h), 0.2))
        return x



# Discriminator architecture
class Discriminator(chainer.Chain):

    def __init__(self, bottom_width=4, ch=1024, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(Discriminator, self).__init__()
        
        with self.init_scope():
        
            self.c0_0 = L.Convolution2D(3, ch // 32, 4, 2, 1, initialW=w) # k=4
            self.c0_1 = L.Convolution2D(ch // 32, ch // 16, 4, 2, 1, initialW=w) # k=4
            self.c1_0 = L.Convolution2D(ch // 16, ch // 8, 4, 2, 1, initialW=w) # k=4
            self.c1_1 = L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w) # k=4
            self.c2_0 = L.Convolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w) # k=3
            self.c2_1 = L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w) # k=4
            self.c3_0 = L.Convolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w) # k=3
            self.c3_1 = L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w) # k=4
            self.c4_0 = L.Convolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w) #k=3

            self.l5 = L.Linear(bottom_width * bottom_width * ch, 1, initialW=w)

            self.bn0_1 = L.BatchNormalization(ch // 16, use_gamma=False)
            self.bn1_0 = L.BatchNormalization(ch // 8, use_gamma=False)
            self.bn1_1 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn2_0 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn2_1 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn3_0 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn3_1 = L.BatchNormalization(ch // 1, use_gamma=False)
            self.bn4_0 = L.BatchNormalization(ch // 1, use_gamma=False)            
           

    def forward(self, x):
        h = add_noise(x)
        h = F.relu(add_noise(self.c0_0(h)))
        h = F.relu(add_noise(self.bn0_1(self.c0_1(h))))
        h = F.relu(add_noise(self.bn1_0(self.c1_0(h))))
        h = F.relu(add_noise(self.bn1_1(self.c1_1(h))))
        h = F.relu(add_noise(self.bn2_0(self.c2_0(h))))
        h = F.relu(add_noise(self.bn2_1(self.c2_1(h))))
        h = F.relu(add_noise(self.bn3_0(self.c3_0(h))))
        h = F.relu(add_noise(self.bn3_1(self.c3_1(h))))
        h = F.sigmoid(add_noise(self.bn4_0(self.c4_0(h))))
        x = self.l5(h)
        return x



# Updater
class DCGANUpdater(chainer.training.updaters.StandardUpdater):
    
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models') 
        super(DCGANUpdater, self).__init__(*args, **kwargs)


    # loss of the Discriminator    
    def loss_dis(self, dis, y_fake, y_real):
        
        batchsize = len(y_fake)

        L1 = F.sum(F.softplus(-y_real)) / batchsize 
        L2 = F.sum(F.softplus(y_fake)) / batchsize

        loss = L1 + L2
        
        chainer.report({'loss': loss}, dis)        
        return loss


    # loss of the Generator     
    def loss_gen(self, gen, y_fake):    
    
        batchsize = len(y_fake)
        loss = F.sum(F.softplus(-y_fake)) / batchsize

        chainer.report({'loss': loss}, gen)
        return loss

    
    # generate intermediate samples
    def update_core(self):
        
        # access model optimizers
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        batch = self.get_iterator('main').next()
        x_real = Variable(self.converter(batch, self.device)) / 255.
       
        xp = cuda.get_array_module(x_real.array)

        gen, dis = self.gen, self.dis
        batchsize = len(batch)

        y_real = dis(x_real)

        z = Variable(xp.asarray(gen.make_hidden(batchsize)))
        x_fake = gen(z)
        y_fake = dis(x_fake)

        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
        gen_optimizer.update(self.loss_gen, gen, y_fake)



# visualize generated images
def out_generated_image(gen, dis, rows, cols, seed, dst):
    @chainer.training.make_extension() 
    
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        xp = gen.xp 
        z = Variable(xp.asarray(gen.make_hidden(n_images)))
        
        with chainer.using_config('train', False): 
            x = gen(z)
        
        x = chainer.cuda.to_cpu(x.array)
        
        np.random.seed()

        x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
    
        _, _, H, W = x.shape
        
        x = x.reshape((rows, cols, 3, H, W))  
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((rows * H, cols * W, 3)) 

        preview_dir = '{}/preview'.format(dst) 
        preview_path = preview_dir +\
            '/image{:0>8}.png'.format(trainer.updater.iteration) 
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)

        Image.fromarray(x).save(preview_path)
    return make_image



def main():
    parser = argparse.ArgumentParser(description='ChainerMN example: DCGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Number of images in each mini-batch')
    parser.add_argument('--communicator', type=str,
                        default='pure_nccl', help='Type of communicator')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', action='store_true',
                        help='Use GPU')
    parser.add_argument('--dataset', '-i', default='',
                        help='Directory of image files.')
    parser.add_argument('--out', '-o', default='dcgan',
                        help='Directory to output the result')
    parser.add_argument('--gen_model', '-r', default='',
                        help='Use pre-trained generator for training')
    parser.add_argument('--dis_model', '-d', default='',
                        help='Use pre-trained discriminator for training')
    parser.add_argument('--n_hidden', '-n', type=int, default=100,
                        help='Number of hidden units (z)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    parser.add_argument('--snapshot_interval', type=int, default=100,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=10,
                        help='Interval of displaying log to console')
    args = parser.parse_args()

    
    # prepare ChainerMN communicator
    if args.gpu:
        if args.communicator == 'naive':
            print('Error: \'naive\' communicator does not support GPU.\n')
            exit(-1)
        
        comm = chainermn.create_communicator(args.communicator)

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

    
    # set up a neural network by making the instances of the Generator and the Discriminator
    gen = Generator(n_hidden=args.n_hidden)
    dis = Discriminator()


    # make a specified GPU current
    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()
        gen.to_gpu()  
        dis.to_gpu()

   
    # set up a multi node optimizer
    def make_optimizer(model, comm, alpha=0.0002, beta1=0.5):

        optimizer = chainermn.create_multi_node_optimizer(
            chainer.optimizers.Adam(alpha=alpha, beta1=beta1), comm)

        optimizer.setup(model)
        
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.001), 'hook_dec')
        return optimizer
        
    
    # create an optimizer for the Generator and Discriminator
    opt_gen = make_optimizer(gen, comm)
    opt_dis = make_optimizer(dis, comm)

    
    # split and distribute the dataset (only worker 0 loads the whole dataset)
    if comm.rank == 0:
        if args.dataset == '':
            print('Please provide the dataset directory')
        else:
            all_files = os.listdir(args.dataset)
            image_files = [f for f in all_files if (f.endswith('.png'))] 
            print('{} contains {} image files'
                  .format(args.dataset, len(image_files)))
            train = chainer.datasets\
                .ImageDataset(paths=image_files, root=args.dataset)
    else:
        train = None

    # dataset of worker 0 is evenly split and distributed to all workers.
    train = chainermn.scatter_dataset(train, comm)

   
    # setup an iterator
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    
    # setup an updater
    updater = DCGANUpdater(
        models=(gen, dis),
        iterator=train_iter,
        optimizer={
            'gen': opt_gen, 'dis': opt_dis},
        device=device)

    
    # setup an trainer
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    
   
    # some display and output extensions are necessary only for one worker, otherwise 
        # there would just be repeated outputs
    if comm.rank == 0:
        snapshot_interval = (args.snapshot_interval, 'iteration')
        display_interval = (args.display_interval, 'iteration')
                
        # save only model parameters (instead of saving whole trainer module, only the network 
            # models are saved)
        trainer.extend(extensions.snapshot_object(
            gen, 'gen_iter_{.updater.iteration}.npz'),
            trigger=snapshot_interval)
        trainer.extend(extensions.snapshot_object(
            dis, 'dis_iter_{.updater.iteration}.npz'),
            trigger=snapshot_interval)

        # output the accumulated results to a log file
        trainer.extend(extensions.LogReport(trigger=display_interval))

        # print the accumulated results
        trainer.extend(extensions.PrintReport([
            'epoch', 'iteration', 'gen/loss', 'dis/loss', 'elapsed_time',
        ]), trigger=display_interval)

        # print a progress bar and recent training status
        trainer.extend(extensions.ProgressBar(update_interval=10))        
        
        trainer.extend(
            out_generated_image(
                gen, dis,
                10, 10, args.seed, args.out),
            trigger=snapshot_interval)


    # start the training using pre-trained model saved by snapshot_object
    if args.gen_model:
        chainer.serializers.load_npz(args.gen_model, gen)
    if args.dis_model:
        chainer.serializers.load_npz(args.dis_model, dis)
   
    
    # run the training
    trainer.run()
    

if __name__ == '__main__':
    main()