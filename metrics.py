# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 16:19:17 2021

@author: 20164798
"""
import numpy as np
import os 
import SimpleITK as sitk

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



