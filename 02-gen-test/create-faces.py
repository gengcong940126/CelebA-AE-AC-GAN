"""

"""

############################################################################
# %% IMPORT PACKAGES
############################################################################

from tensorflow.keras.models import load_model
import matplotlib.pyplot as mp
import numpy as np

############################################################################
# %% CONSTANTS
############################################################################

CLS_SHP = 40
LNV_SHP = 100

############################################################################
# %% LOAD MODEL
############################################################################

g_model = load_model('02-gen-test/gen_model.h5')

############################################################################
# %% TEST ON INPUT STRING
############################################################################

fig = mp.figure(figsize=(40, 16))

y = 2.0*np.random.random((40,40))-1.0
z = np.random.random((40,100))



img = g_model.predict([y, z])
img = ((img[:, :, :, :]+1.0)*127.5).astype(int)
img = img.transpose(1,2,0,3)
out = img.reshape(64, len(input_string)*64, 3, order='F')

out = np.concatenate((out[:,0:len(input_string)*64//4], out[:,len(input_string)*64//4:len(input_string)*64//2], out[:,len(input_string)*64//2:len(input_string)*64*3//4], out[:,len(input_string)*64*3//4:]))
mp.imshow(out)
ax = mp.gca()
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
mp.savefig('tesssset.png')
