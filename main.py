import numpy as np
from PIL import Image

img = np.array(Image.open('my.png').convert('RGB'))
print(img)
print('____________________________')


b = np.unpackbits(img)
print(b)

np.where((b==0)|(b==1), b^1, b)

b = np.packbits(b)
b = np.split(b, 3)
print(b)

""" im = Image.fromarray(b)
im.save("your_file.jpeg") """