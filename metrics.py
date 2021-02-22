# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 16:19:17 2021

@author: 20164798
"""

import math
import numpy as np
from scipy.ndimage import morphology

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
    mean = np.mean(np.stack(images), axis=(1, 2, 3), keepdims=True)
    std = np.std(np.stack(images), axis=(1, 2, 3), keepdims=True)
    images -= mean
    images /= std
    return list(images)

def mutual_information(y_true, y_pred):
    """
    DESCRIPTION: Mutual information (MI) metric
    ----------
    INPUTS:
    y_true: numpy array, containing the fixed image
    y_pred: numpy array, containing the moved image
    -------
    OUTPUTS:
    the MI
    """
    hist_2d, _, _ = np.histogram2d(y_true.ravel(),y_pred.ravel())
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0) 
    px_py = px[:, None] * py[None, :] 

    nzs = pxy > 0 
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def rmse(im1, im2):
    """Calculates the root mean square error (RSME) between two images""" 
    errors = np.abs(im1-im2) 
    return math.sqrt(np.mean(np.square(errors)))

def dice_coef(y_true, y_pred, smooth = 0.0):
    """
    DESCRIPTION: Dice similarity coefficient (DSC) metric: 2*TP / (2*TP + FP + FN)
    ----------
    INPUTS:
    y_true: numpy array, the real label
    y_pred: numpy array, the predicted label
    smooth: 0 standard, 1 if you deal with multiple empty masks
    -------
    OUTPUTS:
    the DSC
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def sensitivity(y_true, y_pred, smooth = 0.0):
    """
    DESCRIPTION: Sensitivity metric: TP / (TP + FN)
    ----------
    INPUTS:
    y_true: numpy array, the real label
    y_pred: numpy array, the predicted label
    smooth: 0 standard, 1 if you deal with multiple empty masks
    -------
    OUTPUTS:
    the sensitivity
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (np.sum(y_true_f) + smooth)

def specificity(y_true, y_pred, smooth = 0.0):
    """
    DESCRIPTION: Specificity metric: TN / (TN + FP)
    ----------
    INPUTS:
    y_true: numpy array, the real label
    y_pred: numpy array, the predicted label
    smooth: 0 standard, 1 if you deal with multiple empty masks
    -------
    OUTPUTS:
    the specificity
    """
    y_true_f = 1 - y_true.flatten()
    y_pred_f = 1 - y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (np.sum(y_true_f) + smooth)

def MeanSurfaceDistance(maskA, maskM):
    """
    DESCRIPTION: Calculates mean surface distance (MSD) between the outer edges of two surfaces.
    For each point on the outer edge of the automatic segmentation the distance to the point closest on the outer edge
    of the manual segmentation is calculated and also vice verse; for each point on the outer edge of the manual
    segmentation the distance to the point closest on the outer edge of the automatic segmentation is calculated.
    The mean of all these distances is calculated and that is the MSD.
    ----------
    INPUTS:
    maskA = automatic segmentation
    maskB = manual segmentation
    -------
    OUTPUTS:
    MSD
    """
    input_1 = np.atleast_1d(maskA.astype(np.bool))
    input_2 = np.atleast_1d(maskM.astype(np.bool))

    conn = morphology.generate_binary_structure(input_1.ndim, 1)

    S = (input_1.astype('uint8') - (morphology.binary_erosion(input_1, conn).astype('uint8'))).astype('bool')
    Sprime = (input_2.astype('uint8') - (morphology.binary_erosion(input_2, conn).astype('uint8'))).astype('bool')

    # voxelsize die uit het artikel van Pluim komt
    sampling = [0.55, 0.55, 3]

    dta = morphology.distance_transform_edt(~S, sampling)
    dtb = morphology.distance_transform_edt(~Sprime, sampling)

    sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])
    msd = sds.mean()

    return msd


