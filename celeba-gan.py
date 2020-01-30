"""
Code for an Auto-Encoder/Auxillary-Classifier GAN (AE-AC-GAN) to generate face
based on CelebA dataset. The model consists of a encoder-generator (auto-encoder)
and a generator-discriminator (auxillary classifier GAN). Discriminator, GAN and
AE are trained per batch, with generator being trained in both auto-encoder and
GAN models. Mode-collapse can therefore be avoided.
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
from sklearn.metrics import accuracy_score, f1_score

################################################################################
# %% CONSTANTS
################################################################################

DATASET = 'celeba'
IMG_SHP = (64, 64, 3)
CLS_SHP = 40
LNV_SHP = 100
EPOCHS = 5
BATCH_SIZE = 256
DEPTH = 64
LEARN_RATE = 0.0002
RESTART = False

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

if RESTART:
    e_model.load_weights("/content/drive/My Drive/Data/enc_model.h5")
    g_model.load_weights("/content/drive/My Drive/Data/gen_model.h5")
    d_model.load_weights("/content/drive/My Drive/Data/dis_model.h5")

##### BUILD AUTOENCODER
ae_model = gan.build_autoencoder(e_model, g_model)

##### BUILD GAN
acgan_model = gan.build_acgan(g_model, d_model)

################################################################################
# %% INIT HISTORY
################################################################################

if RESTART:
    loss = np.load('/content/drive/My Drive/Data/loss.npy').tolist()
    acc = np.load('/content/drive/My Drive/Data/acc.npy').tolist()
else:
    loss = []
    acc = []

################################################################################
# %% LOOP THROUGH EPOCHS
################################################################################

for epoch in range(EPOCHS):

    print(f'Epoch: {epoch}')

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

        ##### GENERATE RANDOM DIGITS (10 RANDOM CLASSES)
        idx = np.random.randint(0, high=CLS_SHP, size=(10, BATCH_SIZE), dtype='int')

        ##### ONE-HOT-ENCODE NUMBERS (10 RANDOM CLASSES)
        y_fake = np.zeros((BATCH_SIZE, CLS_SHP), dtype=float)
        for cls in range(10):
            y_fake[np.arange(BATCH_SIZE), idx[cls]] = 1

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

        ##### SHUFFLE AND TAKE BATCH_SIZE SAMPLES (OTHERWISE DIS TRAINED ON 2X MORE)
        idx = np.random.permutation(len(X_batch))
        X_batch = X_batch[idx][:BATCH_SIZE]
        y_batch = y_batch[idx][:BATCH_SIZE]
        w_batch = w_batch[idx][:BATCH_SIZE]

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

        ############################################################################
        # OUTPUT
        ############################################################################

        loss.append([d1_loss, d2_loss, d3_loss, g1_loss, g2_loss, g3_loss, e1_loss])

        ##### TEST DISCRIMINATOR MODEL ACCURACY
        outputs = d_model.predict(X_batch)
        y_batch.astype(int)
        d2_acc = accuracy_score(w_batch.round(), outputs[0].round(), normalize=True)
        d3_acc = f1_score(y_batch.round(), outputs[1].round(), average='samples')

        ##### TEST GENERATOR MODEL ACCURACY
        outputs = acgan_model.predict([y_fake, z_fake])
        g2_acc = accuracy_score(w_real.round(), outputs[0].round(), normalize=True)
        g3_acc = f1_score(y_fake.round(), outputs[1].round(), average='samples')

        acc.append([d2_acc, d3_acc, g2_acc, g3_acc])

        if batch%8 == 0:
            ##### PRINT LOSS INFOS
            print(f"D(w) loss: {d2_loss:.2e}, D(y) loss: {d3_loss:.2e}, D(G(w)) loss: {g2_loss:.2e}, D(G(w)) loss: {d2_loss:.2e}")

            ##### PRINT ACCURACIES
            print(f"D(w) acc:  {d2_acc:.6f}, D(y) acc:  {d3_acc:.6f}, D(G(w)) acc:  {g2_acc:.6f}, D(G(y)) acc:  {g3_acc:.6f}")

    ############################################################################
    # %% PLOT AUTOENCODER RESULTS
    ############################################################################

    idx = np.random.randint(low=0, high=BATCH_SIZE)
    img0 = ((X_real[idx, :, :, :]+1.0)*127.5).astype(np.uint8)
    y_pred, z_pred = e_model.predict(X_real)
    X_pred = g_model.predict([y_pred, z_pred])
    img1 = ((X_pred[idx, :, :, :]+1.0)*127.5).astype(np.uint8)
    out = np.concatenate((img0, img1), axis=1)
    mp.imsave(f'/content/drive/My Drive/Data/ae_{epoch:03d}.png', out)

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
    mp.legend(['D(w) loss', 'D(y) loss', 'D(G(w)) loss',  'D(G(y)) loss', 'E(X) loss'])
    mp.savefig('/content/drive/My Drive/Data/loss.png')
    mp.close()
    np.save('/content/drive/My Drive/Data/loss.npy', np.array(loss))

    ############################################################################
    # %% PLOT ACCURACY CURVES
    ############################################################################

    fig = mp.figure(figsize=(10,8))
    mp.plot(np.array(acc))
    mp.xlabel('batch')
    mp.ylabel('accuracy')
    mp.legend(['D(w) acc', 'D(y) acc', 'D(G(w)) acc',  'D(G(y)) acc'])
    mp.savefig('/content/drive/My Drive/Data/acc.png')
    mp.close()
    np.save('/content/drive/My Drive/Data/acc.npy', np.array(acc))

    ############################################################################
    # %% TEST GENERATOR
    ############################################################################

    #y = 2.0*np.random.random((CLS_SHP,CLS_SHP))-1.0
    #y = 2.0*np.random.randint(low=0, high=2, size=(CLS_SHP, CLS_SHP))-1.0
    y = np.random.randint(low=0, high=2, size=(CLS_SHP, CLS_SHP))
    z = np.random.random((CLS_SHP,100))
    img = g_model.predict([y, z])
    img = ((img[:, :, :, :]+1.0)*127.5).astype(np.uint8)
    img = img.transpose(1,2,0,3)
    out = img.reshape(64, CLS_SHP*64, 3, order='F')
    out = np.concatenate((out[:,0:40*64//4], out[:,40*64//4:40*64//2], out[:,40*64//2:40*64*3//4], out[:,40*64*3//4:]))
    mp.imsave(f'/content/drive/My Drive/Data/gen_{epoch:03d}.png', out)


acgan_model.summary()
