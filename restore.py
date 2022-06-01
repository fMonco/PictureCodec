from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random


picture = '1.png'

img = Image.open(picture)
img_arr = np.asarray(img)


img_arr.T.shape
img_bin = np.vectorize(np.binary_repr)(img_arr, width=8)
rav = img_bin.copy().ravel()
for el in range(rav.shape[0]):
    i = np.random.randint(0, 8)
    rav[el] = f'{rav[el][:i]}{int(rav[el][i]) ^ 1}{rav[el][i+1:]}'

left = np.identity(8).astype(int)
right = np.random.randint(0, 2, size=(8, 8))
G_sys = np.hstack([left, right])

def xor(a: list):
    res = a[0]
    for el in a[1:]:
        res  = np.bitwise_xor(res, el)
    return res

d = np.array(range(2**8))
left_mat = (((d[:,None] & (1 << np.arange(8))[::-1])) > 0).astype(int)
rmat = np.zeros((2**8, 8))
for i, r in enumerate(left_mat[1:]):
    a = right[np.nonzero(r)[0], :]
    rmat[i+1] = xor(a)


rmat = rmat.astype(int)
rc_mat = np.hstack([left_mat, rmat])
res = np.hstack([left_mat, rc_mat, rc_mat.sum(axis=1).reshape(-1, 1)])
rmat = np.zeros((img_bin.ravel().shape[0], 16))
print(rmat.shape)
for i, r in enumerate(img_bin.ravel()):
    a = G_sys[np.nonzero(np.array(list(map(int, list(r)))))[0], :]
    if len(a) == 0:
        rmat[i] = np.zeros(16)
        continue
    rmat[i] = xor(a)
rmat = rmat.astype(int)



res = []
for el in rmat:
    res.append(np.array2string(el, separator='')[1:-1])
bin = np.array(res).reshape(img_bin.shape)
rav = bin.copy().ravel()


for el in range(rav.shape[0]):
    i = np.random.randint(0, 16)
    rav[el] = f'{rav[el][:i]}{int(rav[el][i]) ^ 1}{rav[el][i+1:]}'
errors = rav.reshape(img_bin.shape)

with open('bin.txt', 'w+') as outfile:
    for slice_2d in bin:
        np.savetxt(outfile, slice_2d, fmt='%s')

with open('errors.txt', 'w+') as outfile:
    for slice_2d in errors:
        np.savetxt(outfile, slice_2d, fmt='%s')


with open('errors.txt', 'r') as file :
    filedata = file.read()

a = random.randint(0, 1)

filedata = filedata.replace('100', '101')
if (a == 1):
    filedata = filedata.replace('0000000', '0000010')
else: 
    filedata = filedata.replace('000000', '010010')


with open('errors.txt', 'w') as file:
    file.write(filedata)
    


H_t_sys = np.vstack((G_sys[:, 8:], G_sys[:, :8]))

mod = np.zeros((errors.ravel().shape[0], 8))




#############

for i, r in enumerate(errors.ravel()):
    a = H_t_sys[np.nonzero(np.array(list(map(int, list(r)))))[0], :]
    if len(a) == 0:
        mod[i] = np.zeros(8)
        continue
    mod[i] = xor(a)
mod = mod.astype(int)
v = rc_mat[11]
v[2] = np.bitwise_not(v[2].astype(bool))
a = H_t_sys[np.nonzero(v)[0], :]
e = a[0]


for l in a[1:]:
    e = np.bitwise_xor(e, l)
kpc = np.hstack([H_t_sys[::-1], np.rot90(np.identity(16).astype(int))])
eer = []


for v in mod:
    m = np.vectorize(np.bitwise_xor)(v, H_t_sys[::-1]).sum(axis=1).argmin()
    e = np.rot90(np.identity(16).astype(int))[m]
    eer.append(e)
eer = np.array(eer)
eee = []


for i, e in enumerate(errors.ravel()):
    p = np.array(list(map(int, list(e))))
    k = np.bitwise_xor(p, eer[i])
    eee.append(k)
eee = np.array(eee)
res = []


for el in eee:
    res.append(np.array2string(el, separator='')[1:-1])
chisht = np.array(res).reshape(bin.shape)



from tqdm import tqdm


ier = []
for v in tqdm(eee):
    m = np.vectorize(np.bitwise_xor)(v, rc_mat).sum(axis=1).argmin()
    e = left_mat[m]
    ier.append(e)
ier = np.array(ier)
res = []



for el in ier:
    res.append(int(''.join(list(map(str, el))), 2))
rrr = np.array(res).reshape(bin.shape)
plt.imshow(rrr.reshape(bin.shape))
plt.axis('off')
plt.show()
plt.savefig('3.jpg')