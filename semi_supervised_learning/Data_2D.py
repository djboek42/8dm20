"""
author: 
    J.P.A. van Boxtel
    j.p.a.v.boxtel@student.tue.nl
"""
import os
import csv
import elasticdeform
import random

import numpy as np
import SimpleITK as sitk

from skimage.transform import resize


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
    mean = np.mean(np.stack(images), axis=(1, 2), keepdims=True)
    std = np.std(np.stack(images), axis=(1, 2), keepdims=True)
    images -= mean
    images /= std
    return images

def elastic_deformation(image):
    sigma = random.randint(1, 10)
    image[:,:,0] = elasticdeform.deform_random_grid(image[:,:,0], sigma=sigma, points=3)
    return image

def reshape_imgs(imgs, img_x, img_y):
    """
    DESCRIPTION: reshapes the input images to the desired shapes
    -------
    INPUTS: 
    imgs:       numpy array containing all input images
    img_rows:   target number of rows for the images
    img_cols:   target number of columns for the images
    -------
    OUTPUTS:
    imgs_p:     numpy array with the reshaped images
    """ 

    imgs_p = np.ndarray((imgs.shape[0], img_x, img_y), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_x, img_y), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

def load_data(data_path, imgs="mr_bffe.mhd", msks=None, img_x=333, img_y=271):
    ##### loading the data to numpy arrays #####
    print_func('Loading Data')
    patients = os.listdir(data_path)
    imgs_train = np.asarray([sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_path, patient, imgs))) for patient in patients if patient.find("p1")>-1])
    
    print_func('store 3D slices as 2D images')
    imgs_train = np.vstack(imgs_train)
    
    print_func(f"Reshape images to shape {img_x}x{img_y}")
    imgs_train = reshape_imgs(imgs_train, img_x, img_y)
    
    print_func("Data normalization")
    imgs_train = imgs_train.astype('float32')
    imgs_train = normalization(imgs_train)
    
    if msks is not None:
        print_func("Repeat for masks")
        msks_train = np.asarray([sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_path, patient, msks))) for patient in patients if patient.find("p1")>-1])
        msks_train = np.vstack(msks_train)
        msks_train = reshape_imgs(msks_train, img_x, img_y)
        msks_train = msks_train.astype('float32')
        msks_train /= np.max(msks_train)  # scale masks to [0, 1]
        
        return imgs_train, msks_train
    else:
        return imgs_train
    
def create_semi_supervised_data(unlabelled_images, model, division_rate, division_step):
    start_idx = division_step*len(unlabelled_images)//division_rate
    end_idx = (division_step+1)*len(unlabelled_images)//division_rate
    if len(unlabelled_images)<end_idx: end_idx = len(unlabelled_images)
    
    subset = unlabelled_images[start_idx:end_idx]
    
    print_func(f"predict for indexes {start_idx}-{end_idx}")
    predicted_masks = np.empty(subset.shape)
    
    for i, image in enumerate(subset):
        image = np.expand_dims(image, axis=0)
        predicted_masks[i] = model.predict(image)
    
    return subset, predicted_masks > 0.5

def save_results(model_name, dice, time, elab=True, file_total = 'results.csv', file_elab = 'results_elaborate.csv'):
    """
    DESCRIPTION: helper function to easily save the model results to a csv file
    -------
    INPUTS:
    model_name: string, name to identify the model with
    dice:       number or list of numbers, the dice similarity score(s) of the model(s) 
    time:       number or list of numbers, the time(s) used for training the model(s)
    elab:       boolean, to determine whether to save the results as a result per fold in K-fold cross 
                validation (elab=True) or to save results as the means and standard deviations of all folds (elab=False)
    file_total: string, name of the file to save means and standard devitations to from all folds combined
    file_elab:  string, name of the file to save results to from each fold
    -------
    OUTPUTS:
    file_elab:  csv file containing the dice score and the train time per fold of the model (if file already exists, results will be appended)
    file_total: csv file containing the mean and standard deviation of the dice score and the train time per model (if file already exists, results will be appended)
    """
    
    files = os.listdir()
    if elab:
        with open(file_elab, 'a', newline="") as file:
            writer = csv.writer(file, delimiter=';')
            if file_elab not in files: writer.writerow(["Model_name", "Dice_score", "Time"])
            writer.writerow([model_name, dice, time])
            file.close()
    else:
        with open(file_total, 'a', newline="") as file:
            writer = csv.writer(file,  delimiter=';')
            if file_total not in files: writer.writerow(["Model_name", "mean_dice_score", "std_dice_score", "mean_time", "std_time"])
            writer.writerow([model_name, np.mean(dice), np.std(dice), np.mean(time), np.std(time)])
            file.close()
        
def print_func(str_in, c = '-', n=50):
    """
    DESCRIPTION: helper function to print information clearly in filled consoles
    -------
    INPUTS:
    str_in: string, the information that needs to be printed
    c:      string, the seperator characted to use
    n:      the number of seperator characters to print
    -------
    OUTPUTS:
    prints the input information with a leading and closing line of n seperator characters
    """
    
    print(c*n)
    print(str_in)
    print(c*n)