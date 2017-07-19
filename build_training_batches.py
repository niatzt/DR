from __future__ import absolute_import, division, print_function

import numpy as np
import os
import json
import skimage
import skimage.io
import skimage.transform

#from util import im_processing
#from models import processing_tools

################################################################################
# Parameters
################################################################################

image_dir = '../train1300each/'

# Saving directory
data_folder = '../trainbatch/'
data_prefix = 'train_det'

# Model Param
N = 10
neg_to_pos_ratio = 0.5
################################################################################
# Load annotations and bounding box proposals
################################################################################
training_samples = []
for i in range(5):
    label = [0,0,0,0,0]
    path = image_dir + str(i) + '/'
    print(path)
    p_files = [pos_csv for pos_csv in os.listdir(path) if pos_csv.endswith('.png')]
    label[i] = 1
    for file_name in p_files: 
        sample = (i, file_name, label)
        training_samples.append(sample)

np.random.seed(3)
perm_idx = np.random.permutation(len(training_samples))
shuffled_training_samples = [training_samples[n] for n in perm_idx]
del training_samples
print('#total sample=', len(shuffled_training_samples))

num_batch = len(shuffled_training_samples) // N
print('total batch number: %d' % num_batch)

################################################################################
# Save training samples to disk
################################################################################

imcrop_batch = np.zeros((N, 224, 224, 3), dtype=np.uint8)
label_batch = np.zeros((N, 5), dtype=np.float32)

if not os.path.isdir(data_folder):
    os.mkdir(data_folder)
for n_batch in range(num_batch):
    print('saving batch %d / %d' % (n_batch+1, num_batch))
    batch_begin = n_batch * N
    batch_end = (n_batch+1) * N
    for n_sample in range(batch_begin, batch_end):
        folder, imname, label = shuffled_training_samples[n_sample]
        im = skimage.io.imread(image_dir + str(folder) + '/' + imname)
        imcrop = skimage.img_as_ubyte(skimage.transform.resize(im[:,:,:3], [224, 224]))

        idx = n_sample - batch_begin
        imcrop_batch[idx, ...] = imcrop
        label_batch[idx, ...] = label

    np.savez(file=data_folder + data_prefix + '_' + str(n_batch) + '.npz',
        imcrop_batch=imcrop_batch,
        label_batch=label_batch)
