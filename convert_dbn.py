import utils
import numpy as np
import os
import zipfile
import pickle
# import glob
from tqdm import tqdm

# Takes old version of belief net and converts to new version
def convert_dbn(belief_net_path):
    belief_net = pickle.load(open(belief_net_path, 'rb'))
    n_rbms = belief_net.n_rbms
    n_components = belief_net.n_components
    new_belief_net = utils.DBN(n_rbms, n_components, np.zeros_like(n_components))
    # Add learned components to new DBN object
    for idx in range(n_rbms):
        new_belief_net.rbms[idx] = belief_net.rbms[idx]
    os.remove(belief_net_path)
    pickle.dump(new_belief_net, open(belief_net_path, 'wb'))

# Defining filepath where models are located    
DIR = '/Volumes/MYTHINGS/models'

# Looping over all model files
# print(glob.glob(DIR, recursive=True))
# for filepath in tqdm(glob.glob(DIR, recursive=True)):
#     print(filepath, '\n')
#     if 'dbn' in filepath:
#         convert_dbn(filepath)
for subdir, dirs, files in os.walk(DIR):
    for file in files:
        # If a model file
        if 'dbn' in file:
            convert_dbn(os.path.join(subdir, file.replace('._','')))
