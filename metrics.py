# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 16:19:17 2021

@author: 20164798
"""
import numpy as np
import os 
import SimpleITK as sitk

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

#%% Example 
#
# #specify data path & load filenames
# data_path = r"C:\Users\20164798\OneDrive - TU Eindhoven\UNI\ME 1\Q3\CS of MI\Image Registration Project\Data"
# patients = os.listdir(data_path)
#
# #load images and masks
# masks = [sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_path, patient, "prostaat.mhd"))) for patient in patients]
#
# DSC = dice_coef(masks[0], masks[1])
# SNS = sensitivity(masks[0], masks[1])
# SPC = specificity(masks[0], masks[1])
#
# print(DSC, SNS, SPC)


