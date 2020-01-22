"""

"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

import glob
import imageio
import numpy as np
import matplotlib.pyplot as mp

################################################################################
# %% GET DIRECTORY CONTENTS
################################################################################

##### READ LABELS FROM FILE
with open('00-raw-data/list_attr_celeba.txt') as lbl_file:
    data = lbl_file.readlines()

batch = 0
idx = 0
BATCH_SIZE = 4096

##### INIT EMPTY IMAGE AND LABEL ARRAY
X = np.zeros((BATCH_SIZE, 64, 64, 3), dtype=int)
y = np.zeros((BATCH_SIZE, 40), dtype=int)

##### SET LABELS AND IMAGES IN ARRAY
for i, line in enumerate(data[2:]):

    ##### READ LABELS
    y[idx, :] = line.strip().replace('  ',' ').replace(' ',',').split(',')[1:]

    ##### READ IMAGE
    img = imageio.imread(f"00-raw-data/{line.split(' ')[0]}")

    ##### CROP IMAGE
    img = img[59:59+128, 24:24+128]

    ##### SCALE IMAGE
    img = img[::2, ::2]

    ##### SET IMAGE
    X[idx, :, :, :] = img

    ##### GET NEXT SAMPLE
    idx += 1

    ##### IF BATCH IS FULL
    if idx==BATCH_SIZE:

        ##### SAVE BATCH TO FILE
        np.save(f'X_{batch:03d}.npy', X)
        np.save(f'y_{batch:03d}.npy', y)

        ##### INIT EMPTY IMAGE AND LABEL ARRAY
        X = np.zeros((4096, 64, 64, 3), dtype=int)
        y = np.zeros((4096, 40), dtype=int)

        ##### SET NEXT BATCH
        idx = 0
        batch += 1

    ##### OUTPUT FOR STATUS
    if i % 1000 == 0:
        print(f"Processing {i}")
