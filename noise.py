from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

picture = '1.png'

img = Image.open(picture)
img_arr = np.asarray(img)

img_arr.T.shape
img_bin = np.vectorize(np.binary_repr)(img_arr, width=8)
rav = img_bin.copy().ravel()
for el in range(rav.shape[0]):
    i = np.random.randint(0, 8)
    rav[el] = f'{rav[el][:i]}{int(rav[el][i]) ^ 1}{rav[el][i+1:]}'
img_err = rav.reshape(img_bin.shape)
pixels_err = np.array(list(map(lambda x: int(x, 2), list(img_err.ravel()))))
plt.axis('off')
plt.imshow(pixels_err.reshape(img_bin.shape))
plt.savefig('2.jpg')
plt.show()

with open('image.txt', 'w+') as outfile:
    for slice_2d in img_bin:
        np.savetxt(outfile, slice_2d, fmt='%s')
