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