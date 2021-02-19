# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 19:26:48 2021

@author: 20164798
"""

import os 
import SimpleITK as sitk
import numpy as np
from scrollview import ScrollView
from matplotlib import pyplot as plt


def normalization(images):
    """
    DESCRIPTION: Z-score normalization for images
    ----------
    INPUTS:
    Images: list of numpy arrays, containing the images to normalize
    -------
    OUTPUTS:
    the normalized images
    """
    mean = np.mean(np.stack(images), axis=(1,2,3), keepdims=True)
    std = np.std(np.stack(images), axis=(1,2,3), keepdims=True)
    images -= mean
    images /= std
    return images


# verschillende sets patienten
patients1 = ['p102', 'p107', 'p108', 'p109', 'p115']
patients2 = ['p116', 'p117', 'p119', 'p120', 'p125']
patients3 = ['p127', 'p128', 'p129', 'p133', 'p135']

# load images and masks
images_org = [sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(patient, "mr_bffe.mhd"))) for patient in patients1]

# normalize images
images_norm = normalization(images_org)

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(25, 5))

ScrollView(images_norm[0]).plot(ax1, cmap='gray')
ScrollView(images_norm[1]).plot(ax2, cmap='gray')
ScrollView(images_norm[2]).plot(ax3, cmap='gray')
ScrollView(images_norm[3]).plot(ax4, cmap='gray')
ScrollView(images_norm[4]).plot(ax5, cmap='gray')
ax3.set_title("genormaliseerd")

fig, (ax6, ax7, ax8, ax9, ax10) = plt.subplots(1, 5, figsize=(25, 5))

ScrollView(images_org[0]).plot(ax6, cmap='gray')
ScrollView(images_org[1]).plot(ax7, cmap='gray')
ScrollView(images_org[2]).plot(ax8, cmap='gray')
ScrollView(images_org[3]).plot(ax9, cmap='gray')
ScrollView(images_org[4]).plot(ax10, cmap='gray')
ax8.set_title("origineel")
plt.show()
