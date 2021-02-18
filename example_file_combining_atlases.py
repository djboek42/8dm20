# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:07:11 2021

@author: 20164798
"""

from metrics import dice_coef, sensitivity, specificity
from Combination_methods import majority_voting, global_weighted_voting

import os 
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

#%% Example combining atlases
#specify data path & load filenames
data_path = r"C:\Users\20164798\OneDrive - TU Eindhoven\UNI\ME 1\Q3\CS of MI\Image Registration Project\Data"
patients = os.listdir(data_path)

#load images and masks
masks = [sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_path, patient, "prostaat.mhd"))) for patient in patients]
images = [sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_path, patient, "mr_bffe.mhd"))) for patient in patients if patient.find("p1")>-1]

new_mask=masks.pop()
new_image=images.pop()

mask_mean = np.sum(masks, axis=0)/np.shape(masks)[0] #only used to visualize the mean mask

m1 = majority_voting(masks, 0.5)
DSC_m1, SNS_m1, SPC_m1 = dice_coef(new_mask, m1), sensitivity(new_mask, m1), specificity(new_mask, m1)

w1 = global_weighted_voting(images, masks, new_image, 0.5)
DSC_w1, SNS_w1, SPC_w1 = dice_coef(new_mask, w1), sensitivity(new_mask, w1), specificity(new_mask, w1)

#%%
# plots for illustration
idx = 40
fig, ax = plt.subplots(1, 4, figsize=(10,5))
ax[0].imshow(new_mask[idx])
ax[0].set_title('Ground truth')
ax[1].imshow(mask_mean[idx]) #plot that shows the voting in the mean mask
ax[1].set_title('Mean mask')
ax[2].imshow(m1[idx]) #plot that shows the result after applying majority voting
ax[2].set_title(f'Majority voting \n DSC:{np.round(DSC_m1, 2)}, \nSNS:{np.round(SNS_m1, 2)}, \nSPC:{np.round(SPC_m1, 2)}')
ax[3].imshow(w1[idx]) #plot that shows the result after applying global weighted voting
ax[3].set_title(f'Weighted voting \n DSC:{np.round(DSC_w1, 2)}, \nSNS:{np.round(SNS_w1, 2)}, \nSPC:{np.round(SPC_w1, 2)}')
plt.show()
