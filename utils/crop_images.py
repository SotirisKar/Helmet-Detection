from os import listdir
from PIL import Image

left = 0
top = 300
right = 1078
bottom = 850
path = '/home/sotiris/Downloads/VID-20211108-145245'

img_list = listdir(path)
for img in img_list:
	image = Image.open(r'{}/{}'.format(path,img))
	image_croped = image.crop((left, top, right, bottom))
	image_croped.save('{}/{}'.format(path,img))
