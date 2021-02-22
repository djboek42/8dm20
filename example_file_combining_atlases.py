# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:07:11 2021

@author: 20164798
"""

from metrics import normalization, dice_coef, sensitivity, specificity, MeanSurfaceDistance, mutual_information, rmse
from Combination_methods import majority_voting, global_weighted_voting, local_weighted_voting
from time import time
from datetime import datetime

import os 
import pickle
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

def get_scores(y_true, y_predict):
    return dice_coef(y_true, y_predict), sensitivity(y_true, y_predict), specificity(y_true, y_predict), MeanSurfaceDistance(y_true, y_predict), mutual_information(y_true, y_predict), rmse(y_true, y_predict)

#%% Example combining atlases
#specify data path & load filenames
data_path = r"C:\Users\20164798\OneDrive - TU Eindhoven\UNI\ME 1\Q3\CS of MI\Image Registration Project\Data"
#data_path = r"/home/jpavboxtel/CSMI/Data"
patients = os.listdir(data_path)

#load images and masks
masks = [sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_path, patient, "prostaat.mhd"))) for patient in patients]
images_raw = [sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_path, patient, "mr_bffe.mhd"))) for patient in patients if patient.find("p1")>-1]

images = normalization(images_raw)

#specify unknown image & mask
unknown_mask=masks.pop()
unknown_image=images.pop()

#calculate mean of masks
mask_mean = np.sum(masks, axis=0)/np.shape(masks)[0] #only used to visualize the mean mask

#%%
#calculate majority voting combination of masks
st = time()
m1 = majority_voting(masks, 0.5)
d_m1 = time() - st

#calculate global weighted voting combination of masks using MI metric
st = time()
w1 = global_weighted_voting(images, masks, unknown_image, 0.5, mutual_information, False, 1)
d_w1 = time() - st

#calculate global weighted voting combination of masks using rmse metric
st = time()
w2 = global_weighted_voting(images, masks, unknown_image, 0.5, rmse, False,  5)
d_w2 = time() - st

#calculate local weighted voting combination of masks (warning, takes a long time) using MI and rsme metric
st = time()
g1, g2 = local_weighted_voting(images, masks, unknown_image, 0.5, max_idx = 50000, use_rmse = True, p = 5)
d_g1 = time() - st

scores_m1 = get_scores(unknown_mask, m1)
scores_w1 = get_scores(unknown_mask, w1)
scores_w2 = get_scores(unknown_mask, w2)
scores_g1 = get_scores(unknown_mask, g1)
scores_g2 = get_scores(unknown_mask, g2)

#save results
results = [m1, d_m1, scores_m1, w1, d_w1, scores_w1, w2, d_w2, scores_w2, g1, d_g1, scores_g1, g2, d_g1, scores_g2]
with open(f'results_{datetime.now().strftime("%Y%m%d_%H.%M.%S")}.pkl', 'wb') as f: pickle.dump(results, f)

#%% plots for illustration
# with open('results/results_20210219_15.06.46.pkl', 'rb') as f: results = pickle.load(f)
# [m1, d_m1, scores_m1, w1, d_w1, scores_w1, w2, d_w2, scores_w2, g1, d_g1, scores_g1, g2, d_g1, scores_g2] = results

metrics = ["DSC", "SNS", "SPC", "MSD", "MI", "RSME"]
nl = "\n"

idx = 43
fig, ax = plt.subplots(1, 5, figsize=(15,5))
ax[0].imshow(unknown_mask[idx])
ax[0].set_title('Ground truth')
ax[1].imshow(mask_mean[idx]) #plot that shows the voting in the mean mask
ax[1].set_title('Mean mask')
ax[2].imshow(m1[idx]) #plot that shows the result after applying majority voting
ax[2].set_title(f'Majority voting \n {nl.join([i + ": " + str(np.round(j, 3)) for i, j in zip(metrics, scores_m1)])}')
ax[3].imshow(w1[idx]) #plot that shows the result after applying global weighted voting
ax[3].set_title(f'Weighted voting MI \n {nl.join([i + ": " + str(np.round(j, 3)) for i, j in zip(metrics, scores_w1)])}')
ax[4].imshow(w2[idx]) #plot that shows the result after applying global weighted voting
ax[4].set_title(f'Weighted voting rmse \n {nl.join([i + ": " + str(np.round(j, 3)) for i, j in zip(metrics, scores_w2)])}')

plt.show()
