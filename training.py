# %% Demo training script used when training (and saving a given DBN)
import numpy as np
from numpy.matlib import repmat
import pickle
import utils
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
import zipfile
from PIL import Image
from time import time

# %% Testing whether code was successfully adapted
PATH = '//icnas1.cc.ic.ac.uk/jp2717/Desktop/DBNs/64/60k/60k.zip'
print('Processing data... \n')
#with zipfile.ZipFile('60k.zip', 'r') as my_zip:
#    n_samples = len(my_zip.namelist()) # number of available image files
n_samples = 60000
width, height = 64, 64
n_components = [32**2, 16**2, 8**2]
training,test = utils.extract_images(PATH, (n_samples,height,width))
training,test = utils.preprocess(training,test)
print('Completed data processing! \n')

# %% Train new RBM
belief_net = utils.DBN2(n_components=n_components,lr=0.1,maxepoch=800,momentum=0.5,penalty=2*(10**-4))
belief_net.fit(training)

# %% Assessing learnt model
idx = np.random.randint(0, 40000)
V = training[idx,:]
Vrec = belief_net.reconstruct(V)
# Plotting
V = V.reshape((64,64))
Vrec = np.around(Vrec.reshape((64,64))).astype('int8')
fig = plt.figure()
plt.imshow(np.hstack( (V,np.ones((64,10)),Vrec) ))
# %%
pickle.dump(belief_net,open('c:/model_info/model1.p','wb'))
# %%
utils.plot_erfs(belief_net, l=1, numhid=64,plot=True)
utils.plot_erfs(belief_net, l=2, numhid=64,plot=True)
utils.plot_erfs(belief_net, l=3, numhid=64,plot=True)
# %
