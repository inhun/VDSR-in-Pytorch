import numpy as np
import os
from PIL import Image
import cv2
import random
import math
import h5py
from torchvision import datasets, transforms

def search(dirname, paths):
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        paths.append(full_filename)


def modcrop(image, scale):
    if len(image.shape) == 3:
        h, w, _ = image.shape
        if h >= w:
            h = w
        else:
            w = h
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        if h >= w:
            h = w
        else:
            w = h
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image


folder = 'dataset/original/291'

#savepath = 'train.h5'
size_input = 41
size_label = 41
stride = 41

# scale factors
scale = [2, 3, 4]
# downsizing
downsizes = [1, 0.7, 0.5]

#data = np.zeros((size_input, size_input, 1, 42708))
#label = np.zeros((size_label, size_label, 1, 42708))
count = -1
margain = 0
data = np.zeros((35369, 1, size_input, size_input))
label = np.zeros((35369, 1, size_input, size_input))
filepaths = []
search('dataset/original/291', filepaths)
num = 0

for i in range(len(filepaths)):
    for s in range(len(scale)):
        image = cv2.imread(filepaths[i])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)

        image = cv2.normalize(image[:, :, 0].astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        im_label = modcrop(image, scale[s])
        [hei, wid] = im_label.shape

        temp_im_input = cv2.resize(im_label, None, fx=1/scale[s], fy=1/scale[s], interpolation=cv2.INTER_AREA)
        im_input = cv2.resize(temp_im_input, dsize=(hei, wid), interpolation=cv2.INTER_CUBIC)

        #print(im_label.shape, im_input.shape)
        for x in range(0, hei - size_input + 1, stride):
            for y in range(0, wid - size_input + 1, stride):
                count += 1
                subim_input = im_input[x:x + size_input, y:y + size_input]
                subim_label = im_label[x:x + size_label, y:y + size_label]
                print(subim_input.shape)
                data[count, 0, :, :] = subim_input
                label[count, 0:, :] = subim_label

data = data.reshape(-1, 1, 41, 41)
label = label.reshape(-1, 1, 41, 41)

with h5py.File('train.h5', 'w') as f:
    f.create_dataset('data', data=data)
    f.create_dataset('label', data=label)
'''
print(count)
target = Image.open("./dataset/test/test1.jpg")
target_data = transforms.ToTensor()(target)
print(target_data.shape)
'''
'''

for i in range(len(filepaths)):
    for flip in range(0, 3):
        for degree in range(0, 4):
            for s in range(len(scale)):
                for downsize in range(len(downsizes)):
                    image = cv2.imread(filepaths[i])
                    print(image.shape)
                    # image = Image.open(filepaths[i]).convert("RGB")
                    if flip == 1:
                        image = np.flip(image, 1)
                    elif flip == 2:
                        image = np.flip(image, 2)

                    image = np.rot90(image, degree)
                    image = cv2.resize(image, None, fx=downsizes[downsize], fy=downsizes[downsize], interpolation=cv2.INTER_AREA)

                    if image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
                        # image = image.convert('YCbCr')
                        image = cv2.normalize(image[:, :, 0].astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
                        im_label = modcrop(image, scale[s])
                        [hei, wid] = im_label.shape

                        # print(im_label)
                        temp_im_input = cv2.resize(im_label, None, fx=1/scale[s], fy=1/scale[s], interpolation=cv2.INTER_AREA)
                        im_input = cv2.resize(temp_im_input, dsize=(hei, wid), interpolation=cv2.INTER_CUBIC)
                        # im_input = cv2.resize(cv2.resize(im_label, 1/scale[s]), (hei, wid))

                        for x in range(0, hei-size_input+1, stride):
                            for y in range(0, wid-size_input+1, stride):
                                subim_input = im_input[x:x+size_input, y:y+size_input]
                                subim_label = im_label[x:x+size_label, y:y+size_label]





                                count += 1



                                #data[:, :, 0, count] = subim_input
                                #label[:, :, 0, count] = subim_label
print(count)
'''
'''
order = []
for i in range(count):
    order.append(i)
random.shuffle(order)
print(order)

data = data.reshape(1, 41, 41, -1)
label = label.reshape(1, 41, 41, -1)
#data = data[:, :, 0, order]
#label = label[:, :, 0, order]

chunksz = 64
created_flag = False
totalct = -1
print(data.shape)
with h5py.File("train.h5", 'w') as f:
    f.create_dataset('data', data=data)
    f.create_dataset('label', data=label)
    '''
'''
    for batchno in range(math.floor(count / chunksz)):
        last_read = batchno * chunksz
        batchdata = data[:, :, 0, last_read:last_read + chunksz]
        batchlabs = label[:, :, 0, last_read :last_read + chunksz]

        startloc = {'dat': [0, 0, 0, totalct], 'lab': [0, 0, 0, totalct]}
        for k, v in startloc.items():
            f.create_dataset('dataset_' + str(batchno), startloc)

'''

