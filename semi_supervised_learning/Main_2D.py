import os

from time import time
from Model_2D import Unet, load_callback_list
from Data_2D import load_data, print_func, save_results

from sklearn.model_selection import KFold
from tensorflow.keras import backend as K

# from tensorflow.keras.layers import BatchNormalization, LayerNormalization
# from tensorflow_addons.layers import InstanceNormalization, GroupNormalization

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

# data_path = r"C:\Users\20164798\OneDrive - TU Eindhoven\UNI\ME 1\Q3\CS of MI\Image Registration Project\Data"
data_path = r"/home/8dm20-7/Data"
save_path = r"/home/8dm20-7/Results"

def train_model(data_path=data_path, imgs="mr_bffe.mhd", msks="prostaat.mhd", model_name="model", save_path = "results", x_size = 320, y_size = 272, num_folds=5, batch_size=32, learning_rate=1e-5, nr_epochs=80, verbosity=1, up=False, start_ch=32, depth=4, inc_rate=2, kernel_size=(3, 3), activation='relu', normalization=None, dropout=0.2):
    
    ##### load data and optional test data #####
    images, masks = load_data(data_path, imgs, msks, x_size, y_size)
    
    ##### save arguments for the model to dictionairy #####
    arg_dict_model = {"img_shape":(x_size, y_size, 1), "start_ch": start_ch, "depth": depth, "inc_rate": inc_rate, "kernel_size": kernel_size, "activation": activation, "normalization": normalization, "dropout": dropout, "learning_rate": learning_rate, "up": up}
    
    ##### prepare for k-fold cross validation #####
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_no = 1
    dice_per_fold, time_per_fold = [], []

    for train, val in kfold.split(images, masks):
        print_func(f'Training for fold {fold_no} (of {num_folds}) ... \nModel name: {model_name}')
        
        ##### divide images and masks into a train and a validation set ######
        train_im, train_msk, val_im, val_msk = images[train], masks[train], images[val], masks[val]  
        
        ##### load model with random initialized weights ######
        model = Unet(**arg_dict_model)
        
        ##### load callbacks #####
        save_dir = os.path.join(save_path, model_name + " K_" + str(fold_no))
        callbacks_list = load_callback_list(save_dir)
       
        ##### fit model #####
        arg_dict_fit = {"x": train_im, "y": train_msk, "validation_data": (val_im, val_msk), "batch_size": batch_size, "epochs": nr_epochs, "verbose": verbosity, "shuffle": True}
        start_time = time()
        model.fit(callbacks=callbacks_list, **arg_dict_fit)
        train_time = int(time()-start_time)
                
        ##### evaluate model #####
        scores = model.evaluate(val_im, val_msk, verbose=0)
        
        ##### save scores of fold #####
        print_func(f"Scores \nDice: {scores[1]} \nTime: {train_time}")
        dice_per_fold.append(scores[1]); time_per_fold.append(train_time)
        save_results(model_name + f' K_{fold_no}', scores[1], train_time)
        
        fold_no += 1 
    
    ##### save scores of model #####
    save_results(model_name, dice_per_fold, time_per_fold, False)    
    
if __name__ == '__main__':
    result = train_model(save_path=save_path)
    
