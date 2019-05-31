import functools

import data
import imlib as im
import module
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2gan as gan
import tf2lib as tl
import tqdm
import matplotlib.pyplot as plt
import sys
import numpy as np

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

# command line
py.arg('--dataset', default='PETCT', choices=['PETCT', 'cifar10', 'fashion_mnist', 'mnist', 'celeba', 'anime', 'custom'])
py.arg('--batch_size', type=int, default=5)
py.arg('--epochs', type=int, default=25)
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--n_d', type=int, default=1)  # # d updates per g update
py.arg('--PETCT_dim', type=int, default=128)
py.arg('--adversarial_loss_mode', default='wgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
py.arg('--gradient_penalty_weight', type=float, default=10.0)
py.arg('--experiment_name', default='PETCT')
py.arg('--data_rate', type=float, default=0.0)
py.arg('--training_mode', type=int, default=0)
py.arg('--padding', default='same', choices=['same', 'valid', 'full'])
py.arg('--input_size',type=int, default=256)
args = py.args()

# output_dir
if args.experiment_name == 'none':
    args.experiment_name = '%s_%s' % (args.dataset, args.adversarial_loss_mode)
    if args.gradient_penalty_mode != 'none':
        args.experiment_name += '_%s' % args.gradient_penalty_mode

#output_dir = py.join('output', '%s_BN%d_DPG%d_DataR%f' % (args.experiment_name, args.batch_size, args.n_d, args.data_rate ) )
if args.padding == 'same':
    Pad = 0
if args.padding == 'valid':
    Pad = 1
if args.padding == 'full':
    Pad = 2

output_dir = py.join('output', '%s_%s_B%dT%dP%d_%d' % (args.experiment_name,args.gradient_penalty_mode, args.batch_size, args.training_mode, Pad, args.input_size))
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)


# ==============================================================================
# =                               data and model                               =
# ==============================================================================

# setup dataset
if args.dataset in ['cifar10', 'fashion_mnist', 'mnist']:  # 32x32
    dataset, shape, len_dataset = data.make_32x32_dataset(args.dataset, args.batch_size)
    n_G_upsamplings = n_D_downsamplings = 3

elif args.dataset == 'celeba':  # 64x64
    img_paths = py.glob('data/img_align_celeba', '*.jpg')
    dataset, shape, len_dataset = data.make_celeba_dataset(img_paths, args.batch_size)
    n_G_upsamplings = n_D_downsamplings = 4

elif args.dataset == 'anime':  # 64x64
    img_paths = py.glob('data/faces', '*.jpg')
    dataset, shape, len_dataset = data.make_anime_dataset(img_paths, args.batch_size)
    n_G_upsamplings = n_D_downsamplings = 4

elif args.dataset == 'PETCT': # Added by K.C: Construt two datasets: PET & CT 256x256 -> 512x512
    img_paths = py.glob('data/faces', '*.jpg')
    dataset, shape, len_dataset = data.make_PETCT_dataset(img_paths, args.batch_size, args.input_size)
    n_G_upsamplings = n_D_downsamplings = 4

elif args.dataset == 'custom':
    # ======================================
    # =               custom               =
    # ======================================
    img_paths = ...  # image paths of custom dataset
    dataset, shape, len_dataset = data.make_custom_dataset(img_paths, args.batch_size)
    n_G_upsamplings = n_D_downsamplings = ...  # 3 for 32x32 and 4 for 64x64
    # ======================================
    # =               custom               =
    # ======================================

# setup the normalization function for discriminator
if args.gradient_penalty_mode == 'none':
    d_norm = 'batch_norm'
if args.gradient_penalty_mode in ['dragan', 'wgan-gp']:  # cannot use batch normalization with gradient penalty
    # TODO(Lynn)
    # Layer normalization is more stable than instance normalization here,
    # but instance normalization works in other implementations.
    # Please tell me if you find out the cause.
    d_norm = 'layer_norm'

# networks
# Comment by K.C:
# the following commands set the structure of a G model
#G = module.ConvGenerator_1(input_shape=(1, 1, args.z_dim), output_channels=shape[-1], n_upsamplings=n_G_upsamplings, name='G_%s' % args.dataset)
G = module.G_Split_Unet( input_shape=(shape[0], shape[1],1), padding=args.padding, name='SplitU_G_%s' % args.dataset )
D1 = module.ConvDiscriminator_2(input_shape=(shape[0],shape[1],1), name='D1_%s' % args.dataset, norm=d_norm)
D2 = module.ConvDiscriminator_2(input_shape=(shape[0],shape[1],1), name='D2_%s' % args.dataset, norm=d_norm)
py.mkdir('%s/summaries' % output_dir)
keras.utils.plot_model(G,'%s/summaries/convGenerator.png' % output_dir, show_shapes=True)
keras.utils.plot_model(D1,'%s/summaries/convDiscriminator1.png' % output_dir, show_shapes=True)
keras.utils.plot_model(D2,'%s/summaries/convDiscriminator2.png' % output_dir, show_shapes=True)
G.summary()
D1.summary()
D2.summary()

# adversarial_loss_functions
d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)

G_optimizer = keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta_1)
D_optimizer = keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta_1)
#Unet_optimizer = keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta_1)

# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

def take_mean(list):
    import numpy as np
    tmp = []
    for i in range(len(list)):
       tmp.append(np.mean(list[0:i+1]))
    return {'tmp': tmp}

def get_PET(x_real):
    import tensorflow as tf
    img_PET = x_real[:,:,:,2]
    img_PET = tf.expand_dims(img_PET,3)
    return img_PET

def get_CT(x_real):
    import tensorflow as tf
    img_CT = x_real[:,:,:,1]
    img_CT = tf.expand_dims(img_CT,3)
    return img_CT

def get_Mask(x_real):
    import tensorflow as tf
    img_Mask = x_real[:,:,:,0]
    img_Mask = tf.expand_dims(img_Mask,3)
    return img_Mask


@tf.function
def train_G(x_real):

    with tf.GradientTape(persistent=True) as t:
        ground_truth = get_Mask(x_real)
        x1_fake, x2_fake = G(ground_truth, training=True)


        x1_fake_d_logit = D1(x1_fake, training=True)
        x2_fake_d_logit = D2(x2_fake, training=True)



        G1_loss = g_loss_fn(x1_fake_d_logit)
        G2_loss = g_loss_fn(x2_fake_d_logit)



    #   initialize the training_mode
    training_mode = args.training_mode

    #   Training Mode 0: update both ends with different length
    if training_mode == 0:
        for layer in G.layers:
            layer.trainable = True

        G2_grad = t.gradient(G2_loss, G.trainable_variables)
        G_optimizer.apply_gradients(zip(G2_grad, G.trainable_variables))

        G1_grad = t.gradient(G1_loss, G.trainable_variables)
        G_optimizer.apply_gradients(zip(G1_grad, G.variables))

    #   Training Mode 1: update the CT branch first, then update the PET branch
    if training_mode == 1:

        # initialize all layers
        for layer in G.layers:
            layer.trainable = True
        # fix the first branch
        for i in range(14,64,2):
            #print(G.layers[i].name)
            G.layers[i].trainable = False
        # Update G2
        G2_grad = t.gradient(G2_loss, G.trainable_variables)
        G_optimizer.apply_gradients(zip(G2_grad, G.trainable_variables))

        # initialize all layers
        #print('The following layers are fixed:')
        for layer in G.layers:
            layer.trainable = True
        # fix the second branch
        for i in range(15,65,2):
            #print(G.layers[i].name)
            G.layers[i].trainable = False
        # update G1
        G1_grad = t.gradient(G1_loss, G.trainable_variables)
        G_optimizer.apply_gradients(zip(G1_grad, G.trainable_variables))

        # reset all layers
        for layer in G.layers:
            layer.trainable = True


    #   Training Mode 2: training with D1 loss only. Both output should be similar to PET
    if training_mode == 2:

    # initialize all layers
        for layer in G.layers:
            layer.trainable = True

        # Update G2
        G2_grad = t.gradient(G1_loss, G.trainable_variables)
        G_optimizer.apply_gradients(zip(G2_grad, G.trainable_variables))
        G1_grad = t.gradient(G1_loss, G.trainable_variables)
        G_optimizer.apply_gradients(zip(G1_grad, G.trainable_variables))

    #   Training Mode 3: training with D2 loss only. Both output should be similar to PET
    if training_mode == 3:

    # initialize all layers
        for layer in G.layers:
            layer.trainable = False

        # Update G2
        G2_grad = t.gradient(G2_loss, G.trainable_variables)
        G_optimizer.apply_gradients(zip(G2_grad, G.trainable_variables))
        G1_grad = t.gradient(G2_loss, G.trainable_variables)
        G_optimizer.apply_gradients(zip(G1_grad, G.trainable_variables))
        

    return{'g1_loss': G1_loss, 'g2_loss': G2_loss}


@tf.function
def train_D(x_real):
    with tf.GradientTape(persistent=True) as t:
        # Changed by K.C, the input signal of the Generator from the original Z noise to the real PET dataset
        x1_real = get_PET(x_real)
        x2_real = get_CT(x_real)


        x1_fake, x2_fake = G(get_Mask(x_real), training=True)

        x1_real_d_logit = D1(x1_real, training=True)
        x1_fake_d_logit = D1(x1_fake, training=True)
        x2_real_d_logit = D2(x2_real, training=True)
        x2_fake_d_logit = D2(x2_fake, training=True)
        x1_real_d_loss, x1_fake_d_loss = d_loss_fn(x1_real_d_logit, x1_fake_d_logit)
        x2_real_d_loss, x2_fake_d_loss = d_loss_fn(x2_real_d_logit, x2_fake_d_logit)

        gp1 = gan.gradient_penalty(functools.partial(D1, training=True), x1_real, x1_fake, mode=args.gradient_penalty_mode)
        gp2 = gan.gradient_penalty(functools.partial(D2, training=True), x2_real, x2_fake, mode=args.gradient_penalty_mode)

        D1_loss = (x1_real_d_loss + x1_fake_d_loss) + gp1 * args.gradient_penalty_weight
        D2_loss = (x2_real_d_loss + x2_fake_d_loss) + gp2 * args.gradient_penalty_weight

    D1_optimizer = D_optimizer
    D2_optimizer = D_optimizer
    D1_grad = t.gradient(D1_loss, D1.trainable_variables)
    D1_optimizer.apply_gradients(zip(D1_grad, D1.trainable_variables))
    D2_grad = t.gradient(D2_loss, D2.trainable_variables)
    D2_optimizer.apply_gradients(zip(D2_grad, D2.trainable_variables))

    return {'d1_loss': x1_real_d_loss + x1_fake_d_loss, 'gp1': gp1, 'd2_loss': x2_real_d_loss + x2_fake_d_loss, 'gp2': gp2}


@tf.function
def sample(x_real):
    ground_truth = get_Mask(x_real)
    return G(ground_truth, training=False)



# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(G=G,
                                D1=D1,
                                D2=D2,
                                G_optimizer=G_optimizer,
                                D_optimizer=D_optimizer,
                                ep_cnt=ep_cnt),
                           py.join(output_dir, 'checkpoints'),
                           max_to_keep=5)
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)

# summary
train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'train'))

# sample
sample_dir = py.join(output_dir, 'samples_training')
py.mkdir(sample_dir)

# main loop
# z = tf.random.normal((100, 1, 1, args.z_dim))  # a fixed noise for sampling
with train_summary_writer.as_default():
    # Added by K.C: initialize the loss function list
    G1_loss_summary = []
    G1_loss_mean = []
    G2_loss_summary = []
    G2_loss_mean = []

    D1_GP_summary = []
    D1_GP_mean = []
    D1_loss_summary = []
    D1_loss_mean = []

    D2_GP_summary = []
    D2_GP_mean = []
    D2_loss_summary = []
    D2_loss_mean = []

    iteration_summary = []


    for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):
        if ep < ep_cnt:
            continue

        # update epoch counter
        ep_cnt.assign_add(1)

        # train for an epoch

        # Comment by K.C:
        # train the discriminator based on the real image
        # Here, x_real is the target domain, which is the CT dataset

        for x_real in tqdm.tqdm(dataset, desc='Inner Epoch Loop', total=len_dataset):
            # Comment by K.C:
            # run train_D means to update D once, D_loss can be printed here.
            # D_loss_dict = train_D(x_real)
            # To change the input stream D should take in not only x_real, but also x_fake, which is the fake CT_dataset
            D_loss_dict = train_D(x_real)
            tl.summary(D_loss_dict, step=D_optimizer.iterations, name='D_losses')
            # Comment by K.C:
            # Update the Discriminator for every n_d run of the Generator
            if D_optimizer.iterations.numpy() % args.n_d == 0:
                G_loss_dict = train_G(x_real)
                tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')

            # sample
            if G_optimizer.iterations.numpy() % 100 == 0:
                ground_truth = get_Mask(x_real)
                x1_real = get_PET(x_real)
                x2_real = get_CT(x_real)
                x1_fake, x2_fake = G(ground_truth, training=True)

                x1_fake = x1_fake[:,:,:,0]
                x2_fake = x2_fake[:,:,:,0]



                img1_real = im.immerge(x1_real.numpy(),n_rows=10)
                img2_real = im.immerge(x2_real.numpy(),n_rows=10)
                img3_real = im.immerge(ground_truth.numpy(),n_rows=10)
                img4_real = tf.concat([img1_real[:,:,0],img2_real[:,:,0],img3_real[:,:,0]],-1)
                im.imwrite(img4_real.numpy(), py.join(sample_dir, 'img4R-iter-%09d.jpg' % G_optimizer.iterations.numpy()))

                #print('\n Shape of the generated images:')
                #print('x1_fake.shape = ', x1_fake.shape)
                #print('x2_fake.shape = ', x2_fake.shape)

                img1 = im.immerge(x1_fake.numpy(), n_rows=10)
                img2 = im.immerge(x2_fake.numpy(), n_rows=10)
                img3 = im.immerge(ground_truth.numpy(), n_rows=10)
                img4 = tf.concat([img1,img2,img3[:,:,0]],-1)
                im.imwrite(img4.numpy(), py.join(sample_dir, 'img4-iter-%09d.jpg' % G_optimizer.iterations.numpy()))

                # Added by K.C: update the mean loss functions every 100 iterations, and plot them out
                D1_loss_summary.append(D_loss_dict.get('d1_loss','').numpy())
                D1_GP_summary.append(D_loss_dict.get('gp1', '').numpy())
                D2_loss_summary.append(D_loss_dict.get('d2_loss','').numpy())
                D2_GP_summary.append(D_loss_dict.get('gp2', '').numpy())

                iteration_summary.append(D_optimizer.iterations.numpy())
                G1_loss_summary.append(G_loss_dict.get('g1_loss', '').numpy())
                G2_loss_summary.append(G_loss_dict.get('g2_loss', '').numpy())

                D1_loss_mean = take_mean(D1_loss_summary)
                D1_GP_mean = take_mean(D1_GP_summary)
                D2_loss_mean = take_mean(D2_loss_summary)
                D2_GP_mean = take_mean(D2_GP_summary)

                G1_loss_mean = take_mean(G1_loss_summary)
                G2_loss_mean = take_mean(G2_loss_summary)

                # plot the G loss function
                G_figure, ax1 = plt.subplots()
                ax1.set_xlabel('iterations')
                ax1.set_ylabel('G1 loss', color='tab:red')
                ax1.plot(iteration_summary, G1_loss_summary, color='tab:red')
                ax1.tick_params(axis='y', labelcolor='tab:red')

                ax2 = ax1.twinx() # Added by K.C: instantiate a second axes that shares the same axis
                ax2.set_ylabel('G2 loss', color='tab:blue')
                ax2.plot(iteration_summary, G2_loss_summary)
                ax2.tick_params(axis='y', labelcolor='tab:blue')

                G_figure.savefig('%s/summaries/G_loss.png' % output_dir)
                plt.close(G_figure)




                # plot the comparison between D_loss
                D_figure, ax1 = plt.subplots()
                ax1.set_xlabel('iterations')
                ax1.set_ylabel('D1 Loss', color='tab:red')
                ax1.plot(iteration_summary, D1_loss_summary, color='tab:red')
                ax1.tick_params(axis='y', labelcolor='tab:red')

                ax2 = ax1.twinx() # Added by K.C: instantiate a second axes that shares the same axis
                ax2.set_ylabel('D2 Loss', color='tab:blue')
                ax2.plot(iteration_summary, D2_loss_summary)
                ax2.tick_params(axis='y', labelcolor='tab:blue')

                D_figure.savefig('%s/summaries/D_loss.png' % output_dir)
                plt.close(D_figure)

                # plot the comparison between GP
                D_figure, ax1 = plt.subplots()
                ax1.set_xlabel('iterations')
                ax1.set_ylabel('D1 GP', color='tab:red')
                ax1.plot(iteration_summary, D1_GP_summary, color='tab:red')
                ax1.tick_params(axis='y', labelcolor='tab:red')

                ax2 = ax1.twinx() # Added by K.C: instantiate a second axes that shares the same axis
                ax2.set_ylabel('D2 GP', color='tab:blue')
                ax2.plot(iteration_summary, D2_GP_summary)
                ax2.tick_params(axis='y', labelcolor='tab:blue')

                D_figure.savefig('%s/summaries/GP_loss.png' % output_dir)
                plt.close(D_figure)


                # plot the comparison between loss and GP of D1

                D_figure, ax1 = plt.subplots()
                ax1.set_xlabel('iterations')
                ax1.set_ylabel('D1 loss', color='tab:red')
                ax1.plot(iteration_summary, D1_loss_summary, color='tab:red')
                ax1.tick_params(axis='y', labelcolor='tab:red')

                ax2 = ax1.twinx() # Added by K.C: instantiate a second axes that shares the same axis
                ax2.set_ylabel('D1 GP', color='tab:blue')
                ax2.plot(iteration_summary, D1_GP_summary)
                ax2.tick_params(axis='y', labelcolor='tab:blue')

                D_figure.savefig('%s/summaries/D1_loss_GP.png' % output_dir)
                plt.close(D_figure)

                # plot the comparison between loss and GP of D1

                D1_figure, ax1 = plt.subplots()
                ax1.set_xlabel('iterations')
                ax1.set_ylabel('D2 Loss', color='tab:red')
                ax1.plot(iteration_summary, D2_loss_summary, color='tab:red')
                ax1.tick_params(axis='y', labelcolor='tab:red')

                ax2 = ax1.twinx() # Added by K.C: instantiate a second axes that shares the same axis
                ax2.set_ylabel('D2 GP', color='tab:blue')
                ax2.plot(iteration_summary, D2_GP_summary)
                ax2.tick_params(axis='y', labelcolor='tab:blue')

                D_figure.savefig('%s/summaries/D2_loss_GP.png' % output_dir)
                plt.close(D_figure)

        # save checkpoint
        checkpoint.save(ep)
