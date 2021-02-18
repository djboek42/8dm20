# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:13:03 2021

@author: 20164798
"""

import numpy as np
from metrics import mutual_information

def majority_voting(masks, threshold=0.5):
    """
    DESCRIPTION: majority voting for combining masks
    ----------
    INPUTS:
    masks: list of numpy arrays, containing the masks to use
    threshold: float, threshold on which to keep/discard voxels
    -------
    OUTPUTS:
    the resulting mask
    """
    mask_mean = np.sum(masks, axis=0)/np.shape(masks)[0] #sum all masks and divide by number of masks
    return mask_mean > threshold #apply threshold to average mask
 
def global_weighted_voting(images, masks, new_image, threshold):
    """
    DESCRIPTION: global weighted voting for combining masks
    ----------
    INPUTS:
    images: list of numpy arrays, containing the images to use
    masks: list of numpy arrays, containing the masks to use
    new_image: numpy array, containing the image to which the known images should be compared
    threshold: float, threshold on which to keep/discard voxels
    -------
    OUTPUTS:
    the resulting mask
    """
    new_images = [new_image]*len(images)
    
    #uncomment following lines for only taking in account prostate regions for similarity metric
    # images = [mask*image for mask, image in zip(masks, images)]
    # new_images = [mask*new_image for mask in masks]
    
    weights = np.array(list(map(mutual_information, images, new_images)))
    weights /= np.max(weights) #optional, normalize the weights
    weighted_masks = [mask*weight for mask, weight in zip(masks, weights)]
    mask_mean=np.sum(weighted_masks, axis=0)/np.shape(weighted_masks)[0]
    return mask_mean > threshold