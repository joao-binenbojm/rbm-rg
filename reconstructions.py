# %% Import packages
import numpy as np
import pickle
import os
import zipfile
from PIL import Image
import utils

MODEL_PATH = '/Users/joaobinenbojm/Desktop/new_models/200epochs/model6.p'
# %% Testing whether code was successfully adapted
PATH = '/Volumes/MYTHINGS/data/ising_data_64/1st/merged.zip'
print('Processing data... \n')
#with zipfile.ZipFile('60k.zip', 'r') as my_zip:
#    n_samples = len(my_zip.namelist()) # number of available image files
n_samples = 20000
width, height = 64, 64
n_components = [32**2, 16**2, 8**2]
training,test = utils.extract_images(PATH, (n_samples,height,width))
training,test = utils.preprocess(training,test)
print('Completed data processing! \n')
# %% Loading in model
belief_net = pickle.load(open(MODEL_PATH, 'rb'))
# %% Obtaining reconstruction plots
import matplotlib.pyplot as plt

img_idx = np.random.randint(15000)
V = training[img_idx, :]
Vrec = belief_net.reconstruct(V)
V_img = V.reshape((64,64))
Vrec_img = np.around(Vrec.reshape((64,64))).astype('uint8')

figure, axes = plt.subplots(nrows=1, ncols=2)
# Original
axes[0].imshow(V_img, cmap='gray')
# axes[0].title.set_text('Original')
axes[0].axis('off')
# Reconstruction
axes[1].imshow(Vrec_img, cmap='gray')
# axes[1].title.set_text('Reconstruction')
axes[1].axis('off')
# %% Visualizing fields
utils.plot_erfs(belief_net, l=1, numhid=64, opt=False)
utils.plot_erfs(belief_net, l=2, numhid=64, opt=False)
utils.plot_erfs(belief_net, l=3, numhid=64, opt=False)

# %%
