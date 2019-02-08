#!/usr/bin/env python
# -*- coding: utf-8 -*-

from xlearn.utils import dataProcessing, MBGD_extractor
from xlearn.segmentation import model_choose
import multiprocessing as mp
import h5py


patch_size = 32
patch_step = 5
batch_size = 100
h5path = '../test/process/{}'.format(patch_size)

nb_conv = 32
size_conv = 3
nb_down = 3
nb_gpu = 0
nb_epoch = 20


process_params = {
    'parentDir': '../test/huge_dset',
    'outpath': '../test/process/{}'.format(patch_size),
    'patch_size': patch_size,
    'patch_step': patch_step,
    'batch_size': batch_size,
}

dataProcessing(**process_params).process(ftype='h5')

# init Net
mdl = model_choose(patch_size, patch_size, nb_conv, size_conv, nb_down, nb_gpu)

with h5py.File(h5path + '.h5', 'r') as f:
    tmp = f['shape']
    total_nb_batch = tmp[:][0]

helper_params = {
    'inpath': h5path,
    'patch_size': patch_size,
    'stride': patch_step,
    'total_nb_batch':  total_nb_batch,
    'MB_size': batch_size,
    'shuffle': True,
}

train_gen = MBGD_extractor(h5path + '.h5', batch_size)

nb_cores = mp.cpu_count()
mdl.fit_generator(generator=train_gen,
                  verbose=2,
                  epochs=nb_epoch,
                  # validation_data=train_gen,
                  steps_per_epoch=total_nb_batch,
                  max_queue_size=10,
                  use_multiprocessing=True,
                  workers=nb_cores
                  )
