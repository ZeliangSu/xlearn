#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2015-2018, UChicago Argonne, LLC. All rights reserved.    #
#                                                                         #
# Copyright 2018. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################
"""
Module containing model_choose, seg_train and seg_predict routines
"""
import numpy as np
import time
import dxchange
from xlearn.utils import nor_data, extract_3d, reconstruct_patches, MBGD_helper
from xlearn.models import transformer2, transformer3_pooling

import psutil
from math import ceil, floor
from PIL import Image

__authors__ = "Xiaogang Yang, Francesco De Carlo"
__copyright__ = "Copyright (c) 2018, Argonne National Laboratory"
__version__ = "0.2.0"
__docformat__ = "restructuredtext en"
__all__ = ['model_choose',
           'seg_train',
           'seg_predict']

def model_choose(ih, iw, nb_conv, size_conv, nb_down, nb_gpu):
    if nb_down == 3:
        mdl = transformer3_pooling(ih, iw, nb_conv, size_conv, nb_gpu)
    else:
        mdl = transformer2(ih, iw, nb_conv, size_conv, nb_gpu)
    return mdl

def seg_train(img_x, img_y, patch_size = 32,
                patch_step = 1, nb_conv = 32, size_conv = 3,
                batch_size =1000, nb_epoch = 20, nb_down = 2, nb_gpu = 1):
    """
    Function description.

    Parameters
    ----------
    img_x: array, 2D or 3D
        Training input of the model. It is the raw image for the segmentation.

    img_y: array, 2D or 3D
        Training output of the model. It is the corresponding segmentation of the training input.

    patch_size: int
        The size of the small patches extracted from the input images. This size should be big enough to cover the
        features of the segmentation object.

    patch_step: int
         The pixel steps between neighbour patches. Larger steps leads faster speed, but less quality. I recommend 1
         unless you need quick test of the algorithm.

    nb_conv: int
          Number of the covolutional kernals for the first layer. This number doubles after each downsampling layer.

    size_conv: int
          Size of the convolutional kernals.

    batch_size: int
          Batch size for the training. Bigger size leads faster speed. However, it is restricted by the memory size of
          the GPU. If the user got the memory error, please decrease the batch size.

    nb_epoch: int
          Number of the epoches for the training. It can be understand as the number of iterations during the training.
          Please define this number as the actual convergence for different data.

    nb_down: int
          Number of the downsampling for the images in the model.

    nb_gpu: int
          Number of GPUs you want to use for the training.


    Returns
    -------
    mdl
        The trained CNN model for segmenation. The model can be saved for future segmentations.
    """
    # if img_x.ndim == 3:
    #     _, ih, iw = img_x.shape
    # else:
    #     ih, iw = img_x.shape
    patch_shape = (patch_size, patch_size)
    # print img_x.shape
    # print img_x.max(), img_x.min()
    img_x = nor_data(img_x)
    img_y = nor_data(img_y)
    # print img_x.shape
    # print img_x.max(), img_x.min()

    train_x = extract_3d(img_x, patch_shape, patch_step)
    train_y = extract_3d(img_y, patch_shape, patch_step)
    # print train_x.shape
    # print train_x.max(), train_x.min()
    train_x = np.reshape(train_x, (len(train_x), patch_size, patch_size, 1))
    train_y = np.reshape(train_y, (len(train_y), patch_size, patch_size, 1))
    mdl = model_choose(patch_size, patch_size, nb_conv, size_conv, nb_down, nb_gpu)
    print(mdl.summary())
    mdl.fit(train_x, train_y, batch_size=batch_size, epochs=nb_epoch)
    return mdl

def seg_predict(img, wpath, spath, patch_size = 32, patch_step = 1,
                  nb_conv=32, size_conv=3,
                  batch_size=1000, nb_down=2, nb_gpu = 1):
    """
    Function description

    Parameters
    ----------
    img : array
        The images need to be segmented.

    wpath: string
        The path where the trained weights of the model can be read.

    spath: string
        The path to save the segmented images.

    patch_size: int
        The size of the small patches extracted from the input images. This size should be big enough to cover the
        features of the segmentation object.

    patch_step: int
         The pixel steps between neighbour patches. Larger steps leads faster speed, but less quality. I recommend 1
         unless you need quick test of the algorithm.

    nb_conv: int
          Number of the covolutional kernals for the first layer. This number doubles after each downsampling layer.

    size_conv: int
          Size of the convolutional kernals.

    batch_size: int
          Batch size for the training. Bigger size leads faster speed. However, it is restricted by the memory size of
          the GPU. If the user got the memory error, please decrease the batch size.

    nb_epoch: int
          Number of the epoches for the training. It can be understand as the number of iterations during the training.
          Please define this number as the actual convergence for different data.

    nb_down: int
          Number of the downsampling for the images in the model.

    nb_gpu: int
          Number of GPUs you want to use for the training.

    Returns
    -------
    save the segmented images to the spath.

      """
    patch_shape = (patch_size, patch_size)
    img = np.float32(nor_data(img))
    mdl = model_choose(patch_size, patch_size, nb_conv, size_conv, nb_down, nb_gpu)
    # print(mdl.summary())
    mdl.load_weights(wpath)
    if img.ndim == 2:
        ih, iw = img.shape
        predict_x = extract_3d(img, patch_shape, patch_step)
        predict_x = np.reshape(predict_x, (predict_x.shape[0], patch_size, patch_size, 1))
        predict_y = mdl.predict(predict_x, batch_size=batch_size)
        predict_y = np.reshape(predict_y, (predict_y.shape[0],patch_size, patch_size))
        predict_y = reconstruct_patches(predict_y, (ih, iw), patch_step)
        fname = spath + 'prd'
        dxchange.write_tiff(predict_y, fname, dtype='float32')

    else:
        pn, ih, iw = img.shape
        for i in range(pn):
            print('Processing the %s th image' % i)
            tstart = time.time()
            predict_x = img[i]
            predict_x = extract_3d(predict_x, patch_shape, patch_step)
            predict_x = np.reshape(predict_x, (len(predict_x), patch_size, patch_size, 1))
            predict_y = mdl.predict(predict_x, batch_size=batch_size)
            predict_y = np.reshape(predict_y, (len(predict_y), patch_size, patch_size))
            predict_y = reconstruct_patches(predict_y, (ih, iw), patch_step)
            fname = spath + 'prd-' + str(i)
            dxchange.write_tiff(predict_y, fname, dtype='float32')
            print('The prediction runs for %s seconds' % (time.time() - tstart))


def MBGD_seg_train(address2X, address2y, patch_size=32,
                patch_step=1, nb_conv=32, size_conv = 3,
                nb_patch_per_MB = 100, nb_epoch=20, nb_down=2, nb_gpu=1):
    '''Mini Batch Grandient Descend without loading whole training set'''

    # calculate some params
    RAM_available = psutil.virtual_memory()[1] >> 20  # unit: MB
    one_Ximg = Image.open(address2X)
    one_yimg = Image.open(address2y)
    totalXSize = one_Ximg.shape[0] * one_Ximg.shape[0] * len(address2X)
    totalySize = one_yimg.shape[0] * one_yimg.shape[0] * len(address2y)

    if totalXSize != totalySize:
        print('Please check dimensions of images for i/o of train folder!')

    # generate a list of batch IDs
    total_nb_patch = (one_Ximg[0] - patch_size) / patch_step *\
                     (one_Ximg[1] - patch_size) / patch_step *\
                     len(address2X)


    list_MBbatchIDs = np.linspace(1, total_nb_patch, total_nb_patch)

    # cal minibatch size
    mem_per_patch = 4 ** patch_size ** 2 >> 20  # 4 bytes = 32 bits, unit: bytes
    MB_size = ceil(RAM_available * 0.8 / (mem_per_patch * nb_patch_per_MB))  #authorize 80% RAM charge

    #
    params = {'patch_size': patch_size,
              'stride': patch_step,
              'nb_conv': nb_conv,
              'size_conv': size_conv,
              'total_nb_batch': total_nb_patch,
              'MB_size': MB_size,
              'nb_epoch': nb_epoch,
              'nb_down': nb_down,
              'nb_gpu': nb_gpu,
              'shuffle': True}

    # datagen: image augmentation

    # withdraw data
    train_gen = MBGD_helper(**params)
    valid_gen = MBGD_helper(**params)

    # init Net
    mdl = model_choose(patch_size, patch_size, nb_conv, size_conv, nb_down, nb_gpu)
    print(mdl.summary())
    mdl.compile()

    # train model
    nb_cores = psutil.cpu_count()
    mdl.fit_generator(generator=train_gen,
                      validation_data=valid_gen,
                      max_queue_size=4, #for not saturating too much
                      use_multiprocessing=True,
                      workers=nb_cores)
    pass