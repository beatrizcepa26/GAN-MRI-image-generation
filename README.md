# Medical Image Generation
This repository contains the code and files related with the development of my dissertation entitled **"Deep Learning for Image Generation Using HPC"**.

The model developed is based on the DCGAN architecture proposed by Radford et al. and receives as input real MRI image scans of the Brain Tumor Segmentation (BraTS) 2020 Dataset and generates several stacks of 100 3-channel 2D images of size 256×256 pixels. Each one contains an axial view of a brain with a tumor where, similarly to the original T2-FLAIR MRI image scans, the grey area represents the brain tissue, and the white area represents the tumoral tissue.


### Files description

- ***data*** contains the images before the pre-processing, that is, 720 images of size 374×368;
- ***resized*** contains the model input data, i.e., 720 images resized to a size 256×256;
- ***img-gen-model.py*** is the image generation model;
- ***resize-img.py*** is the code used to resize the images from **data**.


### References

- [Radford et al.](https://arxiv.org/abs/1511.06434)
- [BraTS 2020 Dataset description](https://www.med.upenn.edu/cbica/brats2020/data.html.)
