# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:13:03 2021

@author: 20164798
"""

import numpy as np
from itertools import product
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
 
def get_box(x, y, z, rad, nr_idx, nr_idxs, idx):
    if nr_idx % 1000 == 0: print(f"get box of idx: {nr_idx}/{nr_idxs}")
    xes = np.array(range(-rad, rad+1))
    yes = np.array(range(-rad, rad+1))*x
    zes = np.array(range(-rad, rad+1))*x*y
    box_idx = [idx+xi+yi+zi for xi, yi, zi in product(xes, yes, zes) if ((idx+xi+yi+zi > 0) and (idx+xi+yi+zi < z*y*x))]
    return box_idx

def box_mi(im, im1, nr_idx, nr_idxs, box_idx):
    if nr_idx % 1000 == 0: print(f"get MI of box: {nr_idx}/{nr_idxs}")
    return mutual_information(im[box_idx], im1[box_idx])

def get_box_idxs(idxs, im_shape = (86, 333, 271), box_len = 10):
    z, y, x = im_shape
    rad = box_len//2 
    box_idx = map(get_box, [x]*len(idxs), [y]*len(idxs), [z]*len(idxs), [rad]*len(idxs), list(range(len(idxs))), [len(idxs)]*len(idxs), idxs)
    return list(box_idx)
    
def get_box_mi(image, new_image, idxs, box_idx): 
    im = image.flatten()
    im1 = new_image.flatten()
    
    mi = map(box_mi, [im]*len(idxs), [im1]*len(idxs), list(range(len(idxs))), [len(idxs)]*len(idxs), box_idx)    
    return list(mi)

def local_weighted_voting(images, masks, unknown_image, threshold, box_idx = None, max_idx = 100000):
    mask_mean = np.sum(masks, axis=0)/np.shape(masks)[0]
    mm = mask_mean.flatten()
    idxs = np.nonzero((mm > 0.2) & (mm < 0.8))[0]
    idxs1 = np.nonzero(mm>0.8)[0]
    weights = [ [] for _ in range(len(images))]
    print(f"{len(idxs)} indexes to calculate")
    for i in range(0, len(idxs), max_idx):
        start_idx = i
        end_idx = i + max_idx
        if end_idx > len(idxs): end_idx = len(idxs)
        print(start_idx, end_idx)
        
        #if box_idx is None:
        box_idx = get_box_idxs(idxs[start_idx:end_idx], images[0].shape)
            #with open('box_idx.pkl', 'wb') as f: pickle.dump(box_idx, f)
    
        new_weights = np.array(list(map(get_box_mi, images, [unknown_image]*len(images), [idxs[start_idx:end_idx]]*len(images), [box_idx]*len(images))))
        weights = np.hstack((weights, new_weights))
        
    weights /= np.max(weights)
    weighted_masks = [mask.flatten()[idxs]*weight for mask, weight in zip(masks, weights)]
    mean_weighted_mask = np.sum(weighted_masks, axis=0)/np.shape(weighted_masks)[0]
    thresholded_mask = mean_weighted_mask > threshold
    
    new_mask = np.zeros(masks[0].flatten().shape)
    new_mask[idxs1] = 1
    new_mask[idxs] = thresholded_mask
    
    return np.reshape(new_mask, masks[0].shape)