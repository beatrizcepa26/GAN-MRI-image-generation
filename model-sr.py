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

# Normalization parameters for pre-trained PyTorch models

'''All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images 
of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of 
[0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]'''
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform=chainer.Sequential(
            F.resize_images(Image.BICUBIC, (hr_height // 4, hr_height // 4))
            chainer.as_array()
            # falta normalizar
            # transforms.Normalize(mean, std)
        )
        self.hr_transform=chainer.Sequential(
            F.resize_images(Image.BICUBIC, (hr_height, hr_height))
            chainer.as_array()
            # falta normalizar
            # transforms.Normalize(mean, std)
        )
        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)



class FeatureExtractor(chainer.Chain):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = L.VGG19Layers()
        self.feature_extractor = chainer.Sequential(
            *list(vgg19_model.children())[:18] # (?)
        )

    def forward(self, img):
        return self.feature_extractor(img)



class ResidualBlock(chainer.Chain):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.in_features=in_features

        with self.init_scope():
            self.conv_block = chainer.Sequential(
                L.Convolution2D(in_features, in_features, 3, 1, 1),
                L.BatchNormalization(in_features, 0.8),
                L.prelu(),
                L.Convolution2D(in_features, in_features, 3, 1, 1),
                L.BatchNormalization(in_features, 0.8),
            )

    def forward(self, x):
        return x + self.conv_block(x)



class GeneratorResNet(chainer.Chain):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(GeneratorResNet, self).__init__()

        # Input layer
        self.conv1 = chainer.Sequential(
            L.Convolution2D(in_channels, 64, 9, 1, 4),
            F.prelu()
        )

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = chainer.Sequential(*res_blocks)



        # Second conv layer (post res blocks)
        self.conv1 = chainer.Sequential(
            L.Convolution2D(64, 64, 3, 1, 1),
            F.BatchNormalization(64, 0.8)
        )


        # Upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                
                L.Convolution2D(64, 256, 3, 1, 1),
                F.BatchNormalization(256),
                F.depth2space(2),                
                F.prelu(),
            ]
        self.upsampling = chainer.Sequential(*upsampling)


        # Output layer
        self.conv1 = chainer.Sequential(
            L.Convolution2D(64, 3, 9, 1, 4),
            F.tanh()
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out



class Discriminator(chainer.Chain):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(L.Convolution2D(in_filters, out_filters, 3, 1, 1))
            
            if not first_block:
                layers.append(F.BatchNormalization(out_filters))
            
            layers.append(F.leaky_relu(0.2))
            layer.append(L.Convolution2D(out_filters, out_filters, 3, 2, 1))
            layers.append(F.BatchNormalization(out_filters))
            layers.append(F.leaky_relu(0.2))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(L.Convolution2D(out_filters, 1, 3, 1, 1))
        
        self.model = chainer.Sequential(*layers)

    def forward(self, img):
        return self.model(img)



class Updater(chainer.training.updaters.StandardUpdater):
    
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models') 
        super(Updater, self).__init__(*args, **kwargs)

        
    def loss_dis(self, dis, y_fake, y_real):
        ######
        batchsize = len(y_fake)
        
        L1 = F.sum(F.softplus(-y_real)) / batchsize # loss of the real samples
        L2 = F.sum(F.softplus(y_fake)) / batchsize # loss of the synthetic samples
        loss = L1 + L2
        
        chainer.report({'loss': loss}, dis)
        
        return loss

    
    
    def loss_gen(self, gen, y_fake):
        #######        
        batchsize = len(y_fake)
        loss = F.sum(F.softplus(-y_fake)) / batchsize
        chainer.report({'loss': loss}, gen)
        return loss

    
    
    def update_core(self):

        
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        batch = self.get_iterator('main').next()
        device = self.device
                
        valid = Variable(chainer.as_array(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)))
        fake = Variable(chainer.as_array(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)))
    
        gen, dis = self.gen, self.dis
        batchsize = len(batch)

        gen_hr = generator(imgs_lr)

        #######
        
        y_real = dis(x_real)

        z = Variable(device.xp.asarray(gen.make_hidden(batchsize)))
        
        x_fake = gen(z)
        
        y_fake = dis(x_fake)
    
        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
        gen_optimizer.update(self.loss_gen, gen, y_fake)






def main():
    os.makedirs("images", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
    opt = parser.parse_args()
    print(opt)

    
    
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


    
    hr_shape = (opt.hr_height, opt.hr_width)

    
    
    # Initialize generator and discriminator
    generator = GeneratorResNet()
    discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
    feature_extractor = FeatureExtractor()

    
    # Set feature extractor to inference mode
    feature_extractor.eval()

    
    # Losses
    criterion_GAN = F.mean_squared_error()
    criterion_content = F.mean_absolutes_error()

    if device >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(device).use()
        generator.to_gpu()  # Copy the model to the GPU
        discriminator.to_gpu()
        criterion_GAN.to_gpu()
        criterion_content.to_gpu()
    

    ''' corrigir

        if opt.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load("saved_models/generator_%d.pth"))
        discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth"))'''

    


    # Setup an optimizer
    def make_optimizer(model, comm, alpha=0.0002, beta1=0.5):

        # Create a multi node optimizer from a standard Chainer optimizer.

        optimizer = chainermn.create_multi_node_optimizer(
            chainer.optimizers.Adam(alpha=alpha, beta1=beta1), comm)

        optimizer.setup(model)
        
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer
        
    
    # make an optimizer for each model
    opt_gen = make_optimizer(generator, comm)
    opt_dis = make_optimizer(discriminator, comm)

    

    # Split and distribute the dataset. Only worker 0 loads the whole dataset.
    # Datasets of worker 0 are evenly split and distributed to all workers.
    if comm.rank == 0:
        if args.dataset == '':
            all_files = os.listdir("../../data/%s" % opt.dataset_name)
            image_files = [f for f in all_files if ('dcm' in f)] 
            print('{} contains {} image files'
                  .format("../../data/%s" % opt.dataset_name, len(image_files)))
            train = chainer.datasets\
                .ImageDataset(paths=image_files, root=args.dataset)
            
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

    for i, imgs in train_iter:

            imgs_lr = Variable(imgs["lr"].type(Tensor))
            imgs_hr = Variable(imgs["hr"].type(Tensor))




# ===============================================================================================================

    # ----------
    #  Training
    # ----------

    for epoch in range(opt.epoch, opt.n_epochs):
        for i, imgs in enumerate(dataloader):

            # Configure model input
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            imgs_hr = Variable(imgs["hr"].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)

            # Adversarial loss
            loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

            # Content loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr)
            loss_content = criterion_content(gen_features, real_features.detach())

            # Total loss
            loss_G = loss_content + 1e-3 * loss_GAN

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss of real and fake images
            loss_real = criterion_GAN(discriminator(imgs_hr), valid)
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            sys.stdout.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                # Save image grid with upsampled inputs and SRGAN outputs
                imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
                gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
                imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
                img_grid = torch.cat((imgs_lr, gen_hr), -1)
                save_image(img_grid, "images/%d.png" % batches_done, normalize=False)

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
            torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)


if __name__ == '__main__':
    main()