import numpy as np
import shutil
from tqdm import tqdm

destination = '/Volumes/MYTHINGS/data/ising_data_64/80k/'
source1 = '/Volumes/MYTHINGS/data/ising_data_64/1st/merged/'
source2 = '/Volumes/MYTHINGS/data/ising_data_64/2nd/merged/'
source3 = '/Volumes/MYTHINGS/data/ising_data_64/3rd/merged/'
source4 = '/Volumes/MYTHINGS/data/ising_data_64/4th/merged/'

# Move stuff from 1st set of images
for idx in tqdm(range(20000)):
	shutil.copyfile(source1 + '{}.png'.format(idx), destination + '{}.png'.format(idx))
	
# Move stuff from 2nd set of images
for idx in tqdm(range(20000)):
	shutil.copyfile(source2 + '{}.png'.format(idx), destination + '{}.png'.format(idx + 20000))

# Move stuff from 3rd set of images
for idx in tqdm(range(20000)):
	shutil.copyfile(source3 + '{}.png'.format(idx), destination + '{}.png'.format(idx + 40000))

# Move stuff from 3rd set of images
for idx in tqdm(range(20000)):
	shutil.copyfile(source4 + '{}.png'.format(idx), destination + '{}.png'.format(idx + 60000))