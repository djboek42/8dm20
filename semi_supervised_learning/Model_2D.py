from __future__ import print_function

from tensorflow.keras import backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Concatenate, Conv2D, Conv2DTranspose, Dropout, UpSampling2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
smooth = 1. #smooth factor used in DSC and Precission metrics

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
    
    n = Conv2D(dim, shape, activation=acti, padding='same')(m)
    n = norm()(n) if norm and type(norm) != tuple else norm[0](norm[1])(n) if type(norm) == tuple else n
    n = Dropout(do)(n) if do else n
    n = Conv2D(dim, shape, activation=acti, padding='same')(n)
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
        m = MaxPooling2D()(n)
        m = level_block_unet(m, int(inc*dim), shape, depth-1, inc, acti, norm, do, up)
        
        if up:
            m = UpSampling2D()(m)
            m = Conv2D(dim, 2, activation=acti, padding='same')(m)
        else:
            m = Conv2DTranspose(dim, shape, strides=(2, 2), padding='same')(m)
        
        n = Concatenate()([n, m])
        m = conv_block(n, dim, shape, acti, norm)   
    else:
        m = conv_block(m, dim, shape, acti, norm, do)
    
    return m

def Unet(img_shape = (96, 96, 1), out_ch=1, start_ch=32, depth=4, inc_rate=2, kernel_size = (3, 3), activation='relu', normalization=None, dropout=0, up = False, compile_model =True, learning_rate = 1e-5):
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
    o = Conv2D(out_ch, (1, 1), activation = 'sigmoid')(o)
    model = Model(inputs=i, outputs=o)
    
    if compile_model: model.compile(optimizer=Adam(lr=learning_rate), loss = dice_coef_loss, metrics=[dice_coef])
    return model

def load_callback_list(save_dir):
    callbacks_list = []
    callbacks_list.append(ModelCheckpoint(save_dir + ' weights.h5', monitor="val_loss", save_best_only=True))
    callbacks_list.append(CSVLogger(save_dir + ' log.out', append=True, separator=';'))
    callbacks_list.append(EarlyStopping(monitor = "val_loss", verbose = 1, min_delta = 0.0001, patience = 5, mode = 'auto', restore_best_weights = True))
    return callbacks_list

def get_generators(data_gen_args, images, masks, batch_size=32):
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    image_datagen.fit(images, augment=True, seed=seed)
    mask_datagen.fit(masks, augment=True, seed=seed)
    image_generator = image_datagen.flow(images, batch_size=batch_size, seed=seed)
    mask_generator = mask_datagen.flow(masks, batch_size=batch_size, seed=seed)

    train_generator = zip(image_generator, mask_generator)
    return train_generator