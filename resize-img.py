from PIL import Image
import os

dir=r'C:/Users/user/desktop/data/'
all_files = os.listdir(dir)
for f in all_files:
    if (f.endswith('.png')):
        image=Image.open(dir+f)
        new_image=image.resize((256, 256))
        new_image.save('C:/Users/user/desktop/resized/resized_'+f)
        # print(new_image.size)
