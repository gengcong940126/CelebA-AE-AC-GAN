"""

"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

from gan import AEACGAN
from tensorflow.keras.models import load_model
import generator
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as mp

################################################################################
# %% CONSTANTS
################################################################################

DATASET = 'celeba'
IMG_SHP = (64, 64, 3)
CLS_SHP = 40
LNV_SHP = 100
EPOCHS = 20
BATCH_SIZE = 128
DEPTH = 128
LEARN_RATE = 0.0001

################################################################################
# %% BUILD MODELS
################################################################################

##### INIT GAN
gan = AEACGAN(IMG_SHP, CLS_SHP, LNV_SHP, DEPTH, LEARN_RATE)

##### BUILD ENCODER AND GENERATOR
e_model = gan.build_encoder()
g_model = gan.build_generator()
d_model = gan.build_discriminator()

################################################################################
# %% LOAD MODEL WEIGHTS IF EXIST
################################################################################

#e_model.load_weights("/content/drive/My Drive/Data/enc_model.h5")
#g_model.load_weights("/content/drive/My Drive/Data/gen_model.h5")
#d_model.load_weights("/content/drive/My Drive/Data/dis_model.h5")

##### BUILD AUTOENCODER
ae_model = gan.build_autoencoder(e_model, g_model)

##### BUILD GAN
acgan_model = gan.build_acgan(g_model, d_model)

################################################################################
# %% INIT HISTORY
################################################################################

loss = []

################################################################################
# %% LOOP THROUGH EPOCHS
################################################################################

for epoch in range(EPOCHS):

    print(f'Epoch: {epoch+1}')

    ##### INIT GENERATOR
    real_gen = generator.image_generator(DATASET, BATCH_SIZE, CLS_SHP)

    ##### GET NUMBER OF BATCHES
    BATCHES = next(real_gen)

    ############################################################################
    # RUN THROUGH BATCHES
    ############################################################################

    for batch in range(BATCHES):

        ############################################################################
        # DISCRIMINATOR TRAINING
        ############################################################################

        ##### GET NEXT BATCH FROM REAL IMAGE GENERATOR
        X_real, y_real = next(real_gen)
        w_real = 0.9*np.ones((len(y_real),1))

        ##### GENERATE RANDOM DIGITS
        idx = np.random.randint(0, high=CLS_SHP, size=BATCH_SIZE, dtype='int')

        ##### ONE-HOT-ENCODE NUMBERS
        y_fake = -np.ones((BATCH_SIZE, CLS_SHP), dtype=float)
        y_fake[np.arange(BATCH_SIZE), idx] = 1

        ##### GENERATE LATENT NOISE VECTOR
        z_fake = np.random.randn(BATCH_SIZE, LNV_SHP)

        ##### PREDICT IMAGE FROM RANDOM INPUT
        X_fake = g_model.predict([y_fake, z_fake])

        ##### SET BINARY CLASS TO FAKE
        w_fake = 0.0*np.ones((BATCH_SIZE, 1), dtype=int)

        ##### CONCAT REAL AND FAKE DATA
        X_batch = np.concatenate((X_real, X_fake), axis=0)
        y_batch = np.concatenate((y_real, y_fake), axis=0)
        w_batch = np.concatenate((w_real, w_fake), axis=0)

        ##### TRAIN!
        d1_loss, d2_loss, d3_loss = d_model.train_on_batch(X_batch, [w_batch, y_batch])

        ############################################################################
        # GENERATOR TRAINING
        ############################################################################

        g1_loss, g2_loss, g3_loss = acgan_model.train_on_batch([y_fake, z_fake], [w_real, y_fake])

        ############################################################################
        # ENCODE TRAINING
        ############################################################################

        e1_loss = ae_model.train_on_batch(X_real, X_real)

        loss.append([d1_loss, d2_loss, d3_loss, g1_loss, g2_loss, g3_loss, e1_loss])

        if batch%5 == 0:
            print(loss[-1])


    ############################################################################
    # %% PLOT AUTOENCODER RESULTS
    ############################################################################

    idx = np.random.randint(low=0, high=BATCH_SIZE)
    img0 = ((X_real[idx, :, :, :]+1.0)*127.5).astype(np.uint8)
    y_pred, z_pred = e_model.predict(X_real)
    X_pred = g_model.predict([y_pred, z_pred])
    img1 = ((X_pred[idx, :, :, :]+1.0)*127.5).astype(np.uint8)
    out = np.concatenate((img0, img1), axis=1)
    mp.imsave(f'ae_{epoch:03d}.png', out)

    ############################################################################
    # %% SAVE MODELS
    ############################################################################

    g_model.save('/content/drive/My Drive/Data/gen_model.h5')
    d_model.save('/content/drive/My Drive/Data/dis_model.h5')
    acgan_model.save('/content/drive/My Drive/Data/gan_model.h5')
    e_model.save('/content/drive/My Drive/Data/enc_model.h5')
    ae_model.save('/content/drive/My Drive/Data/ae_model.h5')

    ############################################################################
    # %% PLOT LOSS CURVES
    ############################################################################

    fig = mp.figure(figsize=(10,8))
    mp.semilogy(np.array(loss)[:, [1, 2, 4, 5, 6]])
    mp.xlabel('batch')
    mp.ylabel('loss')
    mp.legend(['d_bin_loss', 'd_cat_loss', 'g_bin_loss',  'g_cat_loss', 'e_msq_loss'])
    mp.savefig('loss.png')
    mp.close()

    ############################################################################
    # %% TEST GENERATOR
    ############################################################################

    y = 2.0*np.random.random((40,40))-1.0
    z = np.random.random((40,100))
    img = g_model.predict([y, z])
    img = ((img[:, :, :, :]+1.0)*127.5).astype(np.uint8)
    img = img.transpose(1,2,0,3)
    out = img.reshape(64, 40*64, 3, order='F')
    out = np.concatenate((out[:,0:40*64//4], out[:,40*64//4:40*64//2], out[:,40*64//2:40*64*3//4], out[:,40*64*3//4:]))
    mp.imsave(f'gen_{epoch:03d}.png', out)
