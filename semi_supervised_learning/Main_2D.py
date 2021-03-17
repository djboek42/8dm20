import os
import random
import numpy

from time import time
from Model_2D import Unet, load_callback_list, get_generators
from Data_2D import load_data, print_func, save_results, elastic_deformation, create_semi_supervised_data

from sklearn.model_selection import KFold
from tensorflow.keras import backend as K

from tensorflow.keras.layers import BatchNormalization, LayerNormalization
#from tensorflow_addons.layers import InstanceNormalization, GroupNormalization

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

data_path = r"C:\Users\20164798\OneDrive - TU Eindhoven\UNI\ME 1\Q3\CS of MI\Image Registration Project\Data_ML\train"
save_path = r""
unlabelled_data_path = r"C:\Users\20164798\OneDrive - TU Eindhoven\UNI\ME 1\Q3\CS of MI\Image Registration Project\Data_ML\unlabelled"

train_data_gen_args = dict(rotation_range=90, width_shift_range=0.1, height_shift_range=0.1,
                           shear_range=0.2, zoom_range=0.2, horizontal_flip=True, 
                           vertical_flip=True, preprocessing_function=elastic_deformation)


def train_model(data_path, imgs="mr_bffe.mhd", msks="prostaat.mhd", model_name="model", save_path = "results", x_size = 320, y_size = 256, num_folds=5, batch_size=32, learning_rate=1e-5, nr_epochs=80, verbosity=1, up=False, start_ch=32, depth=4, inc_rate=2, kernel_size=(3, 3), activation='relu', normalization=None, dropout=0.2, augment_factor=5, semi_supervised = False, division_rate = 5, shuffle_unlabelled = True, unlabelled_data_path = None):
    
    ##### load data and optional unlabelled data #####
    images, masks = load_data(data_path, imgs, msks, x_size, y_size)
    if semi_supervised: 
        unlabelled_images = load_data(unlabelled_data_path, imgs, None, x_size, y_size)
        predicted_unlabelled_images, predicted_unlabelled_masks = None, None
        if shuffle_unlabelled: random.shuffle(unlabelled_images)        
    
    ##### save arguments for the model to dictionairy #####
    arg_dict_model = {"img_shape":(x_size, y_size, 1), "start_ch": start_ch, "depth": depth, "inc_rate": inc_rate, "kernel_size": kernel_size, "activation": activation, "normalization": eval(str(normalization)), "dropout": dropout, "learning_rate": learning_rate, "up": up}
    
    ##### prepare for k-fold cross validation #####
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_no = 1
    dice_per_fold, time_per_fold = [], []

    for train, val in kfold.split(images, masks):
        if not semi_supervised: division_rate = 1
        for division_step in range(division_rate):
            print_func(f'Training for fold {fold_no} (of {num_folds}), for self_training fold {division_step+1} (of {division_rate}) ... \nModel name: {model_name}')
            
            ##### divide images and masks into a train and a validation set ######
            train_im, train_msk, val_im, val_msk = images[train], masks[train], images[val], masks[val]
            
            if semi_supervised and division_step>0: train_im, train_msk = numpy.vstack((train_im, predicted_unlabelled_images)), numpy.vstack((train_msk, predicted_unlabelled_masks))
            train_gen = get_generators(train_data_gen_args, train_im, train_msk, batch_size)
            val_gen = get_generators({}, val_im, val_msk, batch_size)
            
            ##### load model with random initialized weights ######
            model = Unet(**arg_dict_model)
            
            ##### load callbacks #####
            save_dir = os.path.join(save_path, model_name + " K_" + str(fold_no))
            callbacks_list = load_callback_list(save_dir)
           
            ##### fit model #####
            arg_dict_fit = {"x": train_gen, "validation_data": val_gen, "epochs": nr_epochs, "verbose": verbosity, "callbacks": callbacks_list, "steps_per_epoch": (len(train_im)*augment_factor) // batch_size, "validation_steps": len(val_im)//batch_size}
            start_time = time()
            model.fit(**arg_dict_fit)
            train_time = int(time()-start_time)
                    
            ##### evaluate model #####
            scores = model.evaluate(val_im, val_msk, verbose=0)
            
            ##### save scores of fold #####
            print_func(f"Scores \nDice: {scores[1]} \nTime: {train_time}")
            save_results(model_name + f' K_{fold_no} S_{division_step}', scores[1], train_time)
            
            predicted_unlabelled_images, predicted_unlabelled_masks = create_semi_supervised_data(unlabelled_images, model, division_rate, division_step)
        
        dice_per_fold.append(scores[1]); time_per_fold.append(train_time)
        fold_no += 1 
    
    ##### save scores of model #####
    save_results(model_name, dice_per_fold, time_per_fold, False)    
    return model

if __name__ == '__main__':
    result = train_model(data_path = data_path, unlabelled_data_path = unlabelled_data_path, model_name="test", start_ch=16, batch_size=16, learning_rate=0.0001, normalization="BatchNormalization", depth=6, save_path=save_path, semi_supervised=True)
    
