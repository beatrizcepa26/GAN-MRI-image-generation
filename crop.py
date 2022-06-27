
# Importing Image class from PIL module
from PIL import Image
import os

def crop(height, width):
    k=1
    os.mkdir('C:/Users/user/desktop/cropped/')
    dir=r'C:/Users/user/desktop/cropped/'
    im = Image.open(r'C:/Users/user/desktop/files/result/dcgan1/preview/image00000100.png')
    imgwidth, imgheight = im.size
    # print(im.size)
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            a.save(os.path.join(dir,"IMG-%s.png" % k))
            k +=1

crop(256,256)