# %% This script will be used for extracting circle locations and areas
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import utils
from numpy.matlib import repmat
# import skimage.measure

plt.rcParams.update({'font.size': 15})
MODEL_PATH = 'main_model.p'
# %% Extract list of effective receptive fields
l = 3
N = 64
belief_net = pickle.load(open(MODEL_PATH, 'rb'))
_,numhid = belief_net.rbms[l-1].W.shape
erfs = np.array([utils.eff_receptive_field(belief_net, n, l=l, plot=False, opt=False) for n in tqdm(range(numhid))])

# %% Obtained periodic pattern
for idx in np.random.randint(0,1024,100):
    fig = plt.figure()
    plt.imshow(erfs[idx,:].reshape((N,N)), cmap='hot')
    plt.axis('off')
    plt.colorbar()
    plt.show()
# %% Obtaining the ERF plots for each layer
utils.plot_erfs(belief_net, l=1, numhid=64,plot=True)
utils.plot_erfs(belief_net, l=2, numhid=64,plot=True)
utils.plot_erfs(belief_net, l=3, numhid=64,plot=True)
# %% Plotting intensity histograms
N = 64
erfs_max = repmat(erfs.max(axis=1), N**2 ,1).T
erfs_min = repmat(erfs.min(axis=1), N**2, 1).T
erfs_norm = np.divide(erfs - erfs_min, erfs_max-erfs_min)
fig = plt.figure()
hist = plt.hist(erfs_norm.reshape(-1), bins=100)
plt.title('Normalized Pixel Intensity Histogram')
plt.xlabel('Normalized Pixel Intensity')
plt.ylabel('Pixel Count')
plt.show()
# %% Plotting ERFs
for idx in np.random.randint(0,1024,100):
    fig = plt.figure()
    plt.imshow(erfs[idx,:].reshape((N,N)), cmap='hot')
    plt.colorbar()
    plt.axis('off')
    plt.show()
# %% Determine whether positive spot, negative spot or neither
# Plot entropy histogram
# erfs_entropy = []
# for n in range(numhid):
#     erf_norm = (255*(erfs[n,:] - erfs[n,:].min())/(erfs[n,:].max() - erfs[n,:].min())).astype(np.uint8)
#     erfs_entropy.append(skimage.measure.shannon_entropy(erf_norm))
# fig = plt.figure()
# ax = plt.axes()
# ax.patch.set_facecolor((229/255,236/255,246/255))
# plt.tick_params(
#     which='both',
#     bottom=False,
#     top=False)
# plt.grid(axis= 'y', color='white')
# plt.xlabel('Shannon Entropy')
# plt.ylabel('Frequency')
# plt.title('ERF Entropy Histogram (l=1)')
# hist = plt.hist(erfs_entropy, bins=100)
# plt.show()
# Plot spins and display erf entropy
# N = 64
# for n in range(0,numhid):
#     erf = (255*(erfs[n,:] - erfs[n,:].min())/(erfs[n,:].max() - erfs[n,:].min())).astype(np.uint8)
#     entropy = skimage.measure.shannon_entropy(erf)
#     if entropy < 7.2:
#         fig = plt.figure(n)
#         plt.imshow(erfs[n,:].reshape((N,N)), cmap='hot')
#         plt.show()
#         print('Entropy:', entropy)
# Shannon entropy for classifying images (keep the ones with dots)
# Bottom layer: 5.5, Middle layer: 6.2, top layer: 7.2
if l == 1:
  thrs = 5.5
if l == 2:
  thrs = 6.2
elif l == 3:
    thrs = 7.2
erfs_dummy = erfs.reshape((numhid,-1))
entropy = []
for n in range(numhid):
    erf = (255*(erfs_dummy[n,:] - erfs_dummy[n,:].min())/(erfs_dummy[n,:].max() - erfs_dummy[n,:].min())).astype(np.uint8)
    entropy.append(skimage.measure.shannon_entropy(erf))
entropy = np.array(entropy)
classes = np.zeros(numhid) # classes of erfs
classes[np.logical_and(entropy < thrs, erfs_dummy.mean(axis=1)>0)] = 1
classes[np.logical_and(entropy < thrs, erfs_dummy.mean(axis=1)<=0)] = 2
for u,c in zip(*np.unique(classes, return_counts=True)):
    print('Class: {}, Count: {}'.format(u,c))
# %% Normalization
erfs_norm = []
for n in range(numhid):
    if classes[n] == 1: # positive circle
        erfs_norm.append((erfs[n,:] - erfs[n,:].min())/(erfs[n,:].max() - erfs[n,:].min()) )
    elif classes[n] == 2: # negative circle
        erf_flip = np.abs(erfs[n,:]) # flip the signs
        erfs_norm.append((erf_flip - erf_flip.min())/(erf_flip.max() - erf_flip.min()) )
erfs_norm = np.array(erfs_norm)
# %% Normalized Intensity Histograms
print(erfs_norm.shape)
fig = plt.figure()
hist = plt.hist(erfs_norm.reshape(-1), bins=5)
plt.title('Normalized Pixel Intensity Histogram')
plt.xlabel('Normalized Pixel Intensity')
plt.ylabel('Pixel Count')
plt.show()
# %% Thresholding operation
# 0.73 for middle layer and 0.8 for top layer
if l == 1:
    thrs = 0.58
elif l == 2:
    thrs = 0.6
elif l == 3:
    thrs = 0.65
erfs_thrs = (erfs_norm > thrs).astype('uint8')
        # %% Plotting thresholded spins
n_spin = np.random.randint(0,erfs_thrs.shape[0])
N = 64
for n_spin in np.random.randint(0,erfs_thrs.shape[0],100):
    f, ax = plt.subplots(1, 2, sharey=True)
    ax[0].imshow(erfs_norm[n_spin,:].reshape((N,N)), cmap='hot')
    # ax[0].set_title('Original')
    ax[0].axis('off')
    ax[1].imshow(erfs_thrs[n_spin,:].reshape((N,N)), cmap='gray')
    # ax[1].set_title('Thresholded')
    ax[1].axis('off')
    plt.show()
# %% Compute average circle area (number of pixels in the circle)
avg_area = np.mean(erfs_thrs.sum(axis=1))
std_area = np.std(erfs_thrs.sum(axis=1))
print('Average Spot Area: {}, STD Spot Area: {}'.format(avg_area, std_area))
# %% Plotting bar graph for areas
means = [19.132450331125828, 71.58108108108108, 246.88]
stds = [7.034141429592279,12.405324441824943,39.93526762148965]
har_counts = [151, 74, 25]
sems = [std/np.sqrt(har_count) for std,har_count in zip(stds, har_counts)]
layer_names = ['Bottom Layer', 'Middle Layer', 'Top Layer']

fig = plt.figure()
ax = plt.axes()
ax.patch.set_facecolor((229/255,236/255,246/255))
plt.tick_params(
    which='both',
    bottom=False,
    top=False)
plt.grid(axis= 'y', color='white')
plt.ylabel('Area (pixels)')
plt.title('ERF High Intensity Region Areas')
ax.bar(layer_names, means,yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
plt.show()

##################### Centroid extraction #########################
# %% Extracting centroid coordinate from a given binary image
# https://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions
def centroid_periodic(img):
    height, width = img.shape
    mean_generalized = np.zeros((2,2))
    for y in range(height):
        for x in range(width):
            theta_x = 2*np.pi*x/(width-1)
            theta_y = 2*np.pi*y/(height-1)
            generalized_x = np.array([np.cos(theta_x), np.sin(theta_x)])
            generalized_y = np.array([np.cos(theta_y), np.sin(theta_y)])
            mean_generalized[0,:] += img[y,x] * generalized_x
            mean_generalized[1,:] += img[y,x] * generalized_y
    mean_generalized /= np.prod(img.shape)
    theta_x_mean = np.arctan2(-mean_generalized[0,1],-mean_generalized[0,0]) + np.pi
    theta_y_mean = np.arctan2(-mean_generalized[1,1],-mean_generalized[1,0]) + np.pi
    centroid = (1/(2*np.pi))*np.array([theta_y_mean*(height-1), theta_x_mean*(width-1)])
    return np.around(centroid).astype(np.uint8)
# %% Image showing erf centroid coordinates
cent_list = []
img = np.zeros((N,N))
for n in range(erfs_thrs.shape[0]):
    erf = erfs_thrs[n,:].reshape((N,N))
    cent = centroid_periodic(erf)
    cent_list.append(cent)
    img[tuple(cent)] = 1
    if cent[0] > 0:
        img[tuple(cent + np.array([-1,0]))] = 1
    if cent[1] > 0:
        img[tuple(cent + np.array([0,-1]))] = 1
    if cent[0] < N-1:
        img[tuple(cent + np.array([1,0]))] = 1
    if cent[1] < N-1:
        img[tuple(cent + np.array([0,1]))] = 1
# Plotting image
fig = plt.figure()
plt.imshow(img, cmap='gray', vmin=0, vmax=1)
plt.title('ERF Centroids ($l=1$)')
plt.axis('off')
plt.show()

# %% Finding average lattice spacing (4nn + itself)
def lattice_spacing(cent, cent_list):
    nearest_cent = [np.zeros(2) for i in range(5)]
    nearest_dist = [np.linalg.norm(cent) for i in range(5)]
    for cent2 in cent_list:
        dist = np.linalg.norm(cent2 - cent)
        for idx in range(5):
            if dist < nearest_dist[idx]:
                nearest_cent.insert(idx, cent2)
                nearest_dist.insert(idx, dist)
                nearest_cent.pop()
                nearest_dist.pop()
                break
    # After iterating, we have our four samples of lattice spacings
    return np.mean(nearest_dist[1:])
# Determining average lattice spacing
spacings = [lattice_spacing(cent, cent_list) for cent in cent_list]
mean_spacings = np.mean(spacings)
std_spacings = np.std(spacings)
print('Average Lattice Spacing: {}, Spacing STD: {}'.format(mean_spacings,std_spacings))
# %% Plotting bar for spacings
means = [14.176642718958309, 20.450399623361534, 33.29292865901385]
stds = [16.1746921796763,18.158318068958543,18.68971664094417]
har_counts = [151, 74, 25]
sems = [std/np.sqrt(har_count) for std,har_count in zip(stds, har_counts)]
layer_names = ['Bottom Layer', 'Middle Layer', 'Top Layer']

fig = plt.figure()
ax = plt.axes()
ax.patch.set_facecolor((229/255,236/255,246/255))
plt.tick_params(
    which='both',
    bottom=False,
    top=False)
plt.grid(axis= 'y', color='white')
plt.ylabel('Spacing (pixels)')
# plt.title('ERF High Intensity Region Areas')
ax.bar(layer_names, means,yerr=sems, align='center', alpha=0.5, ecolor='black', capsize=10)
plt.show()
# %%
