import tensorflow_addons as tfa
import tensorflow.keras as keras
import tensorflow as tf
import sys
# ==============================================================================
# =                                  networks                                  =
# ==============================================================================

def _get_norm_layer(norm):
    # Comment by K.C: selection different method of Normalization, here, were are using batch_normalization
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return keras.layers.BatchNormalization
    elif norm == 'instance_norm':
        return tfa.layers.InstanceNormalization
    elif norm == 'layer_norm':
        return tfa.layers.GroupNormalization

# Comment by K.C:
# ConvGenerator defines the attributes of a generator

def G_Unet(pretrained_weights = None, input_size = (256,256,1)):
    inputs = keras.Input(input_size)
    conv1 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = keras.layers.Dropout(0.5)(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = keras.layers.Dropout(0.5)(conv5)

    up6 = keras.layers.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(drop5))
    merge6 = keras.layers.concatenate([drop4,up6], axis = 3)
    conv6 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = keras.layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv6))
    merge7 = keras.layers.concatenate([conv3,up7], axis = 3)
    conv7 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv7))
    merge8 = keras.layers.concatenate([conv2,up8], axis = 3)
    conv8 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv8))
    merge9 = keras.layers.concatenate([conv1,up9], axis = 3)
    conv9 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = keras.layers.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = keras.layers.Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = keras.layers.Model(input = inputs, output = conv10)

    return model



def ConvGenerator_1(input_shape=(1, 1, 128),
                  output_channels=3,
                  dim=64,
                  n_upsamplings=4,
                  norm='batch_norm',
                  name='ConvGenerator_1'):
    Norm = _get_norm_layer(norm)

    # 0
    # Comment by K.C: h is the noise input with the length 128, it goes through a deconvolution pad and upgraded from 1
    # to 64*2^(n_upsampling-1), the minimize pad size is 64*8=512

    h = inputs = keras.Input(shape=input_shape)

    # 1: 1x1 -> 4x4
    # Comment by K.C: d = min(dim * 2^(n_upsampling - 1), dim * 8)
    d = min(dim * 2 ** (n_upsamplings - 1), dim * 8)
    h = keras.layers.Conv2DTranspose(d, 4, strides=1, padding='valid', use_bias=False)(h)
    # Comment by K.C: this is a deconvolution layer (Transpose convolution layer), upsamples the noise signal to at
    # most 512x512
    h = Norm()(h) # selection the Normalization tye based on the Norm function
    h = keras.layers.ReLU()(h)

    # 2: upsamplings, 4x4 -> 8x8 -> 16x16 -> ...
    for i in range(n_upsamplings - 1):
        d = min(dim * 2 ** (n_upsamplings - 2 - i), dim * 8)
        h = keras.layers.Conv2DTranspose(d, 4, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = keras.layers.ReLU()(h)

    h = keras.layers.Conv2DTranspose(output_channels, 4, strides=2, padding='same', use_bias=False)(h)
    h = keras.layers.Activation('tanh')(h)
    # Added by K.C:
    # pringt the structure of the generator
    # convG = keras.Model(inputs=inputs, outputs=h, name=name)
    # keras.utils.plot_model(convG,'output/convGenerator.png', show_shapes=True)
    return keras.Model(inputs=inputs, outputs=h, name=name)





def G_Split_Unet(input_shape = (256,256,1), name='SplitU_G', norm='batch_norm', padding='same'):
    Norm = _get_norm_layer(norm)
    # 256x256x1
    inputs = keras.Input(input_shape)
    # 256x256x64
    conv1 = Norm()(inputs)
    conv1 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(conv1)
    #conv1 = Norm()(conv1)
    conv1 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(pool1)
    #conv2 = Norm()(conv2)
    conv2 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(pool2)
    #conv3 = Norm()(conv3)
    conv3 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(pool3)
    #conv4 = Norm()(conv4)
    conv4 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(conv4)
    drop4 = keras.layers.Dropout(0.5)(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5_1 = keras.layers.Conv2D(1024, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(pool4)
    conv5_1 = keras.layers.Conv2D(1024, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(conv5_1)
    drop5_1= keras.layers.Dropout(0.5)(conv5_1)

    up6_1 = keras.layers.Conv2D(512, 2, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(drop5_1))
    merge6_1 = keras.layers.concatenate([drop4,up6_1], axis = 3)
    conv6_1 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(merge6_1)
    conv6_1 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(conv6_1)

    up7_1 = keras.layers.Conv2D(256, 2, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv6_1))
    merge7_1 = keras.layers.concatenate([conv3,up7_1], axis = 3)
    conv7_1 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(merge7_1)
    conv7_1 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(conv7_1)

    up8_1 = keras.layers.Conv2D(128, 2, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv7_1))
    merge8_1 = keras.layers.concatenate([conv2,up8_1], axis = 3)
    conv8_1 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(merge8_1)
    conv8_1 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(conv8_1)

    up9_1 = keras.layers.Conv2D(64, 2, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv8_1))
    merge9_1 = keras.layers.concatenate([conv1,up9_1], axis = 3)
    conv9_1 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(merge9_1)
    conv9_1 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(conv9_1)
    conv9_1 = keras.layers.Conv2D(2, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(conv9_1)
    conv10_1 = keras.layers.Conv2D(1, 1, activation = 'sigmoid')(conv9_1)


    conv5_2 = keras.layers.Conv2D(1024, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(pool4)
    conv5_2 = keras.layers.Conv2D(1024, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(conv5_2)
    drop5_2 = keras.layers.Dropout(0.5)(conv5_2)

    up6_2 = keras.layers.Conv2D(512, 2, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(drop5_2))
    merge6_2 = keras.layers.concatenate([drop4,up6_2], axis = 3)
    conv6_2 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(merge6_2)
    conv6_2 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(conv6_2)

    up7_2 = keras.layers.Conv2D(256, 2, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv6_2))
    merge7_2 = keras.layers.concatenate([conv3,up7_2], axis = 3)
    conv7_2 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(merge7_2)
    conv7_2 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(conv7_2)

    up8_2 = keras.layers.Conv2D(128, 2, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv7_2))
    merge8_2 = keras.layers.concatenate([conv2,up8_2], axis = 3)
    conv8_2 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(merge8_2)
    conv8_2 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(conv8_2)

    up9_2 = keras.layers.Conv2D(64, 2, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv8_2))
    merge9_2 = keras.layers.concatenate([conv1,up9_2], axis = 3)
    conv9_2 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(merge9_2)
    conv9_2 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(conv9_2)
    conv9_2 = keras.layers.Conv2D(2, 3, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(conv9_2)
    conv10_2 = keras.layers.Conv2D(1, 1, activation = 'sigmoid')(conv9_2)


    model = keras.Model(inputs=inputs, outputs = [conv10_1,conv10_2], name=name)
    #keras.utils.plot_model(model,'U_Net.png', show_shapes=True)
    return model

def ConvDiscriminator_1(input_shape=(64, 64, 3),
                      dim=64,
                      n_downsamplings=4,
                      norm='batch_norm',
                      name='ConvDiscriminator'):
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1: downsamplings, ... -> 16x16 -> 8x8 -> 4x4
    h = keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)
    h = keras.layers.LeakyReLU(alpha=0.2)(h)

    for i in range(n_downsamplings - 1):
        d = min(dim * 2 ** (i + 1), dim * 8)
        h = keras.layers.Conv2D(d, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = keras.layers.LeakyReLU(alpha=0.2)(h)

    # 3: logit
    h = keras.layers.Conv2D(1, 4, strides=1, padding='valid')(h)
    # Added by K.C:
    # Print the structure of the discriminator
    # convD = keras.Model(inputs=inputs, outputs=h, name=name)
    # keras.utils.plot_model(convD,'output/convDiscriminator.png', show_shapes=True)
    return keras.Model(inputs=inputs, outputs=h, name=name)


def ConvDiscriminator_2(input_shape=(256, 256, 1),
                      dim=64,
                      n_downsamplings=4,
                      norm='batch_norm',
                      name='ConvDiscriminator'):
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)
    h = Norm()(h)
    # 1: downsamplings, ... -> 16x16 -> 8x8 -> 4x4
    h = keras.layers.Conv2D(dim, 3, strides=2, padding='same')(h)
    h = keras.layers.LeakyReLU(alpha=0.2)(h)

    for i in range(n_downsamplings - 1):
        d = min(dim * 2 ** (i + 1), dim * 8)
        h = keras.layers.Conv2D(d, 3, strides=4, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = keras.layers.LeakyReLU(alpha=0.2)(h)

   # h = keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(h)
    #h = keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(h)
    #h = keras.layers.Dropout(0.5)(h)
    h = keras.layers.Conv2D(1,3,strides=2,activation='relu',padding='same')(h)
    # 3: logit
    #h = keras.layers.Conv2D(1, 4, strides=1, padding='valid')(h)
    # Added by K.C:
    # Print the structure of the discriminator
    convD = keras.Model(inputs=inputs, outputs=h, name=name)
    #keras.utils.plot_model(convD,'output/convDiscriminator.png', show_shapes=True)
    return convD

def ConvDiscriminator_3(input_shape=(256, 256, 1),
                      dim=64,
                      n_downsamplings=6,
                      norm='batch_norm',
                      name='ConvDiscriminator'):
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)
    #h = Norm()(h)
    # 1: downsamplings, ... -> 16x16 -> 8x8 -> 4x4
    h = keras.layers.Conv2D(dim, 3, strides=2, padding='same')(h)
    h = keras.layers.LeakyReLU(alpha=0.2)(h)

    for i in range(n_downsamplings - 1):
        d = min(dim * 2 ** (i + 1), dim * 8)
        h = keras.layers.Conv2D(d, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = keras.layers.LeakyReLU(alpha=0.2)(h)

   # h = keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(h)
    #h = keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(h)
    #h = keras.layers.Dropout(0.5)(h)
    h = keras.layers.Conv2D(1,3,strides=2,activation='relu',padding='same')(h)
    # 3: logit
    #h = keras.layers.Conv2D(1, 4, strides=1, padding='valid')(h)
    # Added by K.C:
    # Print the structure of the discriminator
    convD = keras.Model(inputs=inputs, outputs=h, name=name)
    #keras.utils.plot_model(convD,'output/convDiscriminator.png', show_shapes=True)
    return convD