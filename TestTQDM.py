import functools
import tqdm
import pylib as py
import data
import tensorflow as tf
import tensorflow.keras as keras
import module
import tf2gan as gan
import matplotlib.pyplot as plt

# adversarial_loss_functions
d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn('wgan')

#img_paths = py.glob('data/faces', '*.jpg')
#dataset, shape, len_dataset = data.make_PETCT_dataset(img_paths, 1)
#n_G_upsamplings = n_D_downsamplings = 4




#img_paths = py.glob('data/faces', '*.jpg')
img_paths = py.glob('data/STS_RGB','*.png')


dataset, shape, len_dataset = data.make_PETCT_dataset(img_paths, 5)
n_G_upsamplings = n_D_downsamplings = 4



# networks
# Comment by K.C:
# the following commands set the structure of a G model
#G = module.ConvGenerator_1(input_shape=(1, 1, args.z_dim), output_channels=shape[-1], n_upsamplings=n_G_upsamplings, name='G_%s' % args.dataset)
G = module.G_Split_Unet(input_shape=(shape[0],shape[1],1), name='G_padding_PETCT', padding='Same')

#G = module.G_Split_Unet( input_shape=(shape[0], shape[1],1), padding='valid', name='G_Split_Unet')

D1 = module.ConvDiscriminator_2(input_shape=(shape[0],shape[1],1), name='D1_PETCT' )
D2 = module.ConvDiscriminator_2(input_shape=(shape[0],shape[1],1), name='D2_PETCT' )
keras.utils.plot_model(G, 'convGenerator.png', show_shapes=True)

G.summary()
D1.summary()
D2.summary()


def take_mean(list):
    import numpy as np
    tmp = []
    for i in range(len(list)):
       tmp.append(np.mean(list[0:i+1]))
    return {'tmp': tmp}

def get_PET(x_real):
    import tensorflow as tf
    img_PET = x_real[:,:,:,0]
    img_PET = tf.expand_dims(img_PET,3)
    return img_PET

def get_CT(x_real):
    import tensorflow as tf
    img_CT = x_real[:,:,:,1]
    img_CT = tf.expand_dims(img_CT,3)
    return img_CT

def get_Mask(x_real):
    import tensorflow as tf
    img_Mask = x_real[:,:,:,2]
    img_Mask = tf.expand_dims(img_Mask,3)
    return img_Mask


@tf.function
def train_G(x_real):

    with tf.GradientTape() as t:
        #z = tf.random.normal(shape=(args.batch_size, 1, 1, args.z_dim))
        #x_fake = G(z, training=True)
        # Changed by K.C, the input signal changed from noise Z to the real PET image
        ground_truth = get_Mask(x_real)
        x1_fake, x2_fake = G(ground_truth, training=True)
        x1_fake_d_logit = D1(x1_fake, training=True)
        x2_fake_d_logit = D2(x2_fake, training=True)
        G1_loss = g_loss_fn(x1_fake_d_logit)
        G2_loss = g_loss_fn(x2_fake_d_logit)
        rate = 0.5
        G_loss = rate * G1_loss + (1-rate) * G2_loss

    #G_grad = t.gradient(G_loss, G.trainable_variables)

    return {'g_loss': G_loss}




# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)



for ep in tqdm.trange(200, desc='Epoch Loop'):
    if ep < ep_cnt:
        continue

    # update epoch counter
    ep_cnt.assign_add(1)

    # train for an epoch
    for x_real in tqdm.tqdm(dataset, desc='Inner Epoch Loop', total=len_dataset):

        x_real
        img_PET = get_PET(x_real)
        img_CT = get_CT(x_real)
        img_Mask = get_Mask(x_real)

        img_PET = tf.expand_dims(img_PET,3)
        img_PET.shape

        G_loss_dict = train_G(x_real)
        with tf.GradientTape() as t:
            ground_truth = get_Mask(x_real)
            x1_fake, x2_fake = G(ground_truth, training=True)
            x1_fake_d_logit = D1(x1_fake, training=True)
            x2_fake_d_logit = D2(x2_fake, training=True)
            G1_loss = g_loss_fn(x1_fake_d_logit)
            G2_loss = g_loss_fn(x2_fake_d_logit)
    t.gradient(G1_loss, G.trainable_variables)

'''
        print('fix all the layers in G:')
        for layer in G.layers:
            print(layer.name)
            #layer.trainable = False

        print('set layers from one branch to be trainable')
        for i in range(14,64,2):
            print(i)
            print(G.layers[i].name)
            G.layers[i].trainable = False


        print('set layers from another branch to be trainable')
        for i in range(15,65,2):
            print(i)
            print(G.layers[i].name)
            G.layers[i].trainable = False







        G_grad = t.gradient(G1_loss, G.trainable_variables)

        x1_fake, x2_fake = G(img_Mask, training=False)
        x1_fake.shape
        x1_fake = x1_fake[1,:,:,0]
        x2_fake = x2_fake[1,:,:,0]
        x_fake = tf.concat([x1_fake.numpy(), x2_fake.numpy()], 0)
        plt.figure()
        plt.imshow(x_fake.numpy())
        plt.show()


'''








