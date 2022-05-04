from PIL import Image
import os


# image = Image.open('C:/Users/user/desktop/image_00078.png')
# # (x, y, right, lower) where x and y are the starting position and right and lower are the distance in pixels from this 
#     # starting position towards the right and bottom direction. (0,0) is the upper left corner
# box = (26, 80, 282, 336) 
# new_image = image.crop(box)
# new_image.save('image_00078_cropped.png')
# new_image.show()


dir=r'C:/Users/user/desktop/data/'
all_files = os.listdir(dir)
for f in all_files:
    if (f.endswith('.png')):
        image=Image.open(dir+f)
        new_image=image.resize((256, 256))
        new_image.save('C:/Users/user/desktop/resized/resized_'+f)
        print(new_image.size)




# image = Image.open('C:/Users/user/desktop/001_img_00060.png')
# new_image=image.resize((256, 256))
# new_image.save('001_img_00060_resized.png')
# new_image.show()

# print(new_image.size) 