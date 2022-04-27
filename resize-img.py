from PIL import Image

# image = Image.open('C:/Users/user/desktop/image_00078.png')
# # (x, y, right, lower) where x and y are the starting position and right and lower are the distance in pixels from this 
#     # starting position towards the right and bottom direction. (0,0) is the upper left corner
# box = (26, 80, 282, 336) 
# new_image = image.crop(box)
# new_image.save('image_00078_cropped.png')
# new_image.show()

image = Image.open('C:/Users/user/desktop/image_00078.png')
new_image=image.resize((256, 256))
new_image.save('image_00078_resized.png')
new_image.show()

print(new_image.size) 