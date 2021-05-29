import numpy as np
import pickle
import os
from tqdm import tqdm

FILE_PATH = '/Volumes/MYTHINGS/ising_data_256/'

# Merging 10 files into 1 single pickle file
file_names = [str(i+1) + '.p' for i in range(10)]
for i in tqdm(range(10)):
  if i == 0:
    data = pickle.load( open( FILE_PATH + file_names[i], "rb" ) )
  else:
    data = np.concatenate((data, pickle.load( open( FILE_PATH + file_names[i], "rb" )))) # merging data into one single numpy array

# Randomly mixing and pickling data
data = np.random.shuffle(data)
pickle.dump(data, open( FILE_PATH + 'data.p', 'wb'))