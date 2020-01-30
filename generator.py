"""
Generator functions to supply training data batches for the different cases.
"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

import glob
import numpy as np

################################################################################
# %% GENERATE IMAGES IN BATCHES FROM FILES
################################################################################

def image_generator(DATASET='celeba', BATCH_SIZE=128, CAT_SHP=40):

    ##### GET IMAGE FILES
    #fimg = sorted(glob.glob('01-prep-data/X*.npy'))
    fimg = sorted(glob.glob('/content/drive/My Drive/Data/X*.npy'))

    ##### READ FIRST FILE TO BUFFER
    print(f'----> Reading {fimg[0]} <----')
    X = np.load(fimg[0])
    y = np.load(fimg[0].replace('X', 'y'))

    ##### CALCULATE NUMBER OF BATCHES
    BATCHES = int(len(X)*len(fimg)/BATCH_SIZE)

    ##### FIRST YIELD NUMBER OF BATCHES
    yield BATCHES

    ##### START COUNTING
    i = 0

    ##### START LOOP
    while True:

        ##### IF BUFFER NOT SUFFICIENT TO COVER NEXT BATCH
        if len(X) < BATCH_SIZE:

            ##### SET NEXT INDEX BASED ON LENGHT OF FOLDER CONTENTS
            if i<len(fimg)-1:
                i += 1
            else:
                i = 0

            ##### APPEND NEXT FILE
            print(f'----> Reading {fimg[i]} <----')
            X = np.concatenate((X, np.load(fimg[i])), axis=0)
            y = np.concatenate((y, np.load(fimg[i].replace('X', 'y'))), axis=0)

            ##### SHUFFLE
            idx = np.random.permutation(len(X))
            X = X[idx, :, :, :]
            y = y[idx]

        else:

            ##### YIELD SET
            yield X[:BATCH_SIZE]/127.5-1.0, (y[:BATCH_SIZE]+1.0)/2.0

            ##### REMOVE YIELDED RESULTS
            X = np.delete(X, range(BATCH_SIZE), axis=0)
            y = np.delete(y, range(BATCH_SIZE), axis=0)
