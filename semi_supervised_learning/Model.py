from __future__ import print_function

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv3D, Conv3DTranspose, Dropout, UpSampling3D, AveragePooling3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

import numpy as np

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
smooth = 1. #smooth factor used in DSC and Precission metrics

###########################################  
def sensitivity(y_true, y_pred):
    """
    DESCRIPTION: Dice similarity coefficient (DSC) metric, which can be used on predictions from the model
    ----------
    INPUTS:
    y_true: numpy array, the real label
    y_pred: numpy array, the predicted label
    -------
    OUTPUTS:
    the sensitivity
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (np.sum(y_true_f) + smooth)

def specificity(y_true, y_pred):
    """
    DESCRIPTION: Dice similarity coefficient (DSC) metric, which can be used on predictions from the model
    ----------
    INPUTS:
    y_true: numpy array, the real label
    y_pred: numpy array, the predicted label
    -------
    OUTPUTS:
    the specificity
    """
    y_true_f = 1 - y_true.flatten()
    y_pred_f = 1 - y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (np.sum(y_true_f) + smooth)

def dice_coef_pred(y_true, y_pred):
    """
    DESCRIPTION: Dice similarity coefficient (DSC) metric, which can be used on predictions from the model
    ----------
    INPUTS:
    y_true: numpy array, the real label
    y_pred: numpy array, the predicted label
    -------
    OUTPUTS:
    the DSC
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def precision_pred(y_true, y_pred):
    """
    DESCRIPTION: the precission metric
    ----------
    INPUTS:
    y_true: numpy array, the real label
    y_pred: numpy array, the predicted label
    -------
    OUTPUTS:
    the precission
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (np.sum(y_pred_f) + smooth)

def dice_coef(y_true, y_pred):
    """
    DESCRIPTION: Dice similarity coefficient (DSC) metric, which can be used in the keras backend
    ----------
    INPUTS:
    y_true: the real labels.
    y_pred: the predicted labels via network.
    -------
    OUTPUTS:
    the DSC
    """
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
############################################
    
def dice_coef_loss(y_true, y_pred):
    """
    DESCRIPTION: Dice similarity coefficient (DSC) Loss function, which can be used in the keras backend
    ----------
    INPUTS:
    y_true: the real labels.
    y_pred: the predicted labels via network.
    -------
    OUTPUTS:
    the DSC Loss function
    """
    return -dice_coef(y_true, y_pred)

def conv_block(m, dim, shape, acti, norm, do=0):
    """
    DESCRIPTION: Convolution block used in both U-net and M-net structure
    ----------
    INPUTS:
    m:      the previous layers of the model on which to build on
    dim:    int, the number of filters to be used in the convolution layers
    shape:  int or tuple of 2 integers, the kernel size to be used in the convolution layers
    acti:   string, which activation function to use in the convolution layers
    norm:   function, normalization function. In case of Groupnormalization a tuple of the function and the desired group size
    do:     float between 0-1, the dropout rate to be used
    -------
    OUTPUTS:
    n:      the model with the new convolution block added
    """
    
    n = Conv3D(dim, shape, activation=acti, padding='same')(m)
    n = norm()(n) if norm and type(norm) != tuple else norm[0](norm[1])(n) if type(norm) == tuple else n
    n = Dropout(do)(n) if do else n
    n = Conv3D(dim, shape, activation=acti, padding='same')(n)
    n = norm()(n) if norm and type(norm) != tuple else norm[0](norm[1])(n) if type(norm) == tuple else n
    return n

def level_block_unet(m, dim, shape, depth, inc, acti, norm, do, up):
    """
    DESCRIPTION: Recursive function, used to build the U-net structure
    ----------
    INPUTS:
    m:      the previous layers of the model on which to build on
    dim:    int, the number of filters to be used in the convolution layers
    shape:  int or tuple of 2 integers, the kernel size to be used in the convolution layers
    depth:  int, the number of convolutional layers to build
    inc:    number, the factor with which the number of filters is incremented per convolutional layer
    acti:   string, which activation function to use in the convolution layers
    norm:   function, normalization function. In case of Groupnormalization a tuple of the function and the desired group size
    do:     float between 0-1, the dropout rate to be used
    up:     boolean, True for using upsampling, False for using Transposed convolution 
    -------
    OUTPUTS:
    m:      the stacked layers of the models
    """
    
    if depth > 0:
        n = conv_block(m, dim, shape, acti, norm, do)
        m = AveragePooling3D()(n)
        m = level_block_unet(m, int(inc*dim), shape, depth-1, inc, acti, norm, do, up)
        
        if up:
            m = UpSampling3D()(m)
            m = Conv3D(dim, 2, activation=acti, padding='same')(m)
        else:
            m = Conv3DTranspose(dim, shape, strides=(2, 2, 2), padding='same')(m)
        
        n = Concatenate()([n, m])
        m = conv_block(n, dim, shape, acti, norm)   
    else:
        m = conv_block(m, dim, shape, acti, norm, do)
    
    return m

def Unet(img_shape = (80, 272, 320, 1), out_ch=1, start_ch=32, depth=4, inc_rate=2, kernel_size = (3, 3, 3), activation='relu', normalization=None, dropout=0, up = False, compile_model =True, learning_rate = 1e-5):
    """
    DESCRIPTION: The U-net model
    ----------
    INPUTS:
    img_shape:      tuple, the shape of the input images
    out_ch:         int, the number of filters for the output layer
    start_ch:       int, the number of filters for the first convolutional layers
    depth:          int, the number of convolutional layers
    inc:            number, the factor with which the number of filters is incremented per convolutional layer
    kernel_size:    int or tuple of 2 integers, the kernel size to be used in the convolution layers  
    activation:     string, which activation function to use in the convolution layers
    normalization:  function, normalization function. In case of Groupnormalization a tuple of the function and the desired group size
    dropout:        float between 0-1, the dropout rate to be used
    up:             boolean, True for using upsampling, False for using Transposed convolution 
    -------
    OUTPUTS:
    model:          the compiled U-net model
    """
    
    i = Input(shape=img_shape)
    o = level_block_unet(i, start_ch, kernel_size, depth, inc_rate, activation, normalization, dropout, up)
    o = Conv3D(out_ch, (1, 1, 1), activation = 'sigmoid')(o)
    model = Model(inputs=i, outputs=o)
    
    if compile_model: model.compile(optimizer=Adam(lr=learning_rate), loss = dice_coef_loss, metrics=[dice_coef])
    return model

a = Unet()

