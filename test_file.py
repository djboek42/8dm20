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
ground_truths = []   
results_m = []
results_w1 = []
results_w2 = []
results_g1 = []
results_g2 = []
times = []
# "p102", "p107", "p108", "p109", "p115", "p116", "p117", "p119", "p120", "p125"]#, 

patient_list = ["p127", "p128", "p129", "p133", "p135"]    
start_path = r"C:\Users\20164798\OneDrive - TU Eindhoven\UNI\ME 1\Q3\CS of MI\Image Registration Project"
start_path = r"/home/jpavboxtel/CSMI"

for patient_nr in patient_list:
    print(f"Patient: {patient_nr}")
    
    #specify data path & load filenames
    original_data_path = os.path.join(start_path, "Data")
    registered_data_path = os.path.join(start_path, r"Test_Data/results" + patient_nr)
    
    patients = os.listdir(registered_data_path)
    patients = [patient for patient in patients if patient[-4:] != patient_nr]
    
    masks = [sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(registered_data_path, patient, "result.mhd"))) for patient in patients]
    images_raw = [sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(registered_data_path, patient, "result.1.mhd"))) for patient in patients]
    images = normalization(images_raw)
    
    #specify unknown image & mask
    unknown_mask= sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(original_data_path, patient_nr, "prostaat.mhd")))
    unknown_image= sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(original_data_path, patient_nr, "mr_bffe.mhd")))
    
    #calculate mean of masks
    mask_mean = np.sum(masks, axis=0)/np.shape(masks)[0] #only used to visualize the mean mask
    
    # #calculate majority voting combination of masks
    # start_t = time()
    # m1 = majority_voting(masks)
    # m1_t = start_t - time()
    
    # #calculate global weighted voting combination of masks using MI metric
    # start_t = time()
    # w1 = global_weighted_voting(images, masks, unknown_image, metric=mutual_information, only_mask=False, p=1)
    # w1_t = start_t - time()
    
    # #calculate global weighted voting combination of masks using rmse metric
    # start_t = time()
    # w2 = global_weighted_voting(images, masks, unknown_image, metric=rmse, only_mask=False, p=-1)
    # w2_t = start_t - time()
    
    #calculate local weighted voting combination of masks (warning, takes a long time) using MI and rsme metric
    start_t = time()
    g1, g2 = local_weighted_voting(images, masks, unknown_image, max_idx = 50000, use_rmse = True, p = -1)
    g1_t = start_t - time()  
    
    results_g1.append(g1), results_g2.append(g2), times.append(g1_t)#, ground_truths.append(unknown_mask); results_m.append(m1); results_w1.append(w1); results_w2.append(w2), times.append([m1_t, w1_t, w2_t])
    with open(f'results_{patient_nr}_{datetime.now().strftime("%Y%m%d_%H.%M.%S")}.pkl', 'wb') as f: pickle.dump([g1, g2, g1_t], f)

with open(f'results_{patient_list[0]}_{datetime.now().strftime("%Y%m%d_%H.%M.%S")}.pkl', 'wb') as f: pickle.dump([results_g1, results_g2, times], f)

# #%%
# thresholds = np.linspace(0.1, 0.9, 9)
# scores_tm = np.empty((len(thresholds), len(ground_truths), 6))
# scores_tw1 = scores_tm.copy()
# scores_tw2 = scores_tm.copy()

# for i, t in enumerate(thresholds):
#     print(f"threshold: {t}")
#     tm = np.asarray(results_m)>t
#     tw1 = np.asarray(results_w1)>t
#     tw2 = np.asarray(results_w2)>t
    
#     print("get scores")
#     scores_tm[i] = np.asarray(list(map(get_scores, tm, ground_truths)))
#     scores_tw1[i] = np.asarray(list(map(get_scores, tw1, ground_truths)))
#     scores_tw2[i] = np.asarray(list(map(get_scores, tw2, ground_truths)))

# #%%
# results = [ground_truths, results_m, results_w1, results_w2, times, scores_tm, scores_tw1, scores_tw2]
# with open(f'results_{datetime.now().strftime("%Y%m%d_%H.%M.%S")}.pkl', 'wb') as f: pickle.dump(results, f)
