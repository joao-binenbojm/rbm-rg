import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt
from scipy.constants import mu_0
from sklearn.neural_network import BernoulliRBM
from tqdm import tqdm
import pickle
from PIL import Image
import zipfile
import os
from time import time

############## CLASSES ##############

# Classes that contain internal properties of the given Ising model
# Class for 1D Ising Model
class IsingModel1D:
    def __init__(self, J, B, N, beta=1):
        self.J = J
        self.mu = mu_0*np.sqrt(3) # given only two orientations (s=1/2)
        self.B = B
        self.beta = beta
        self.N = N
    def Hamiltonian(self, spins): # for a given 1D system with nn interactions
        field_comp = -self.mu*self.B*spins.sum()
        nn_comp = -self.J*self.beta*np.dot(spins,np.roll(spins,shift=1))
        return field_comp + nn_comp
    def H_change(self,spins,spin_idx): # Computes change in Hamiltonian by only looking at nn interactions
        H_local_old = -self.J*self.beta*spins[spin_idx]*(spins[spin_idx-1] + spins[(spin_idx+1) % self.N])
        dummy_spins = spins.copy()
        dummy_spins[spin_idx] = -dummy_spins[spin_idx] # flip the spin
        H_local_new = -self.J*self.beta*dummy_spins[spin_idx]*(dummy_spins[spin_idx-1]\
                                                               + dummy_spins[(spin_idx+1) % self.N] )
        return H_local_new - H_local_old
    def ising_metropolis(self, n_itrs=1000, init_cond=True, N_store=None, N_init=0):
        ''' Takes in a Hamiltonian function (H), lattice dimension (dim) and length(N), and number of iterations (n)
            and returns the spins at each iteration. If init_cond = True, start with constant initial lattice.
            If False, then initialize with random spins. N_store determines how many samples will be stored.
            N_init determines how many initial samples will be skipped
        '''
        if N_store: # if a specific number of samples
          self.spin_tracker = np.zeros((N_store,self.N)) # limited sample storage
          step = np.floor((n_itrs - N_init)/N_store).astype(int) # skip as many samples to prevent correlation
        else:
          self.spin_tracker = np.zeros((n_itrs,self.N)) # unlimited sample storage
          step = 1 # store all samples
        if init_cond: # initialize lattice
          spins = np.ones(self.N)
        else:
          spins = np.random.choice([1,-1],size=self.N)
        sample_counter = 0
        for n in tqdm(range(n_itrs)):
            s = np.random.randint(0,self.N)
            deltaH = self.H_change(spins, s)
            r = np.exp(-self.beta*(deltaH))
            z = np.random.uniform()
            if r > z: # determining whether to keep flipped spin
                spins[s] = -spins[s]
            # self.spin_tracker[n,:] = spins
            if (n >= N_init) and (n % step == 0):
              self.spin_tracker[sample_counter,:] = spins
              sample_counter += 1
            # If not enough separation for another sample, end simulation
            if (n_itrs - n) < step:
              break     
# Class for 2D Ising model
class IsingModel2D(IsingModel1D):
    def __init__(self, J, B, N, beta=1):
        super().__init__(J, B, N, beta)
    def Hamiltonian(self, spins): # for a given D system with nn interactions
        field_comp = -self.mu*self.B*spins.sum()
        nn_comp = 0
        # Only count neighbours right or below to avoid double counting
        for i in range(self.N):
            for j in range(self.N):
                if i + j == 2*(self.N-1): # accounting for periodic boundary conditions
                    nn_comp += spins[i,j]*spins[i,0] + spins[i,j]*spins[0,j]
                elif i == self.N-1:
                    nn_comp += spins[i,j]*spins[i,j+1] + spins[i,j]*spins[0,j]
                elif j == self.N-1:
                    nn_comp += spins[i,j]*spins[i,0] + spins[i,j]*spins[i+1,j]
                else:
                    nn_comp += spins[i,j]*spins[i,j+1] + spins[i,j]*spins[i+1,j]
        nn_comp *= -self.J*self.beta
        return field_comp + nn_comp
    def H_change(self,spins,s):
      s1,s2 = s
      H_local_old = -self.J*self.beta*spins[s]*(spins[(s1+1)%self.N,s2] + spins[s1-1,s2] \
                                                + spins[s1,(s2+1)%self.N] + spins[s1,s2-1])
      # flipping the spins to calculate energy change
      dummy_spins = spins.copy()
      dummy_spins[s] = -dummy_spins[s]
      H_local_new = -self.J*self.beta*dummy_spins[s]*(dummy_spins[(s1+1)%self.N,s2] + dummy_spins[s1-1,s2] \
                                    + dummy_spins[s1,(s2+1)%self.N] + dummy_spins[s1,s2-1])
      return (H_local_new - H_local_old)
    def ising_metropolis(self, n_itrs=1000, init_cond=True, N_store=None, N_init=0):
        ''' Takes in a Hamiltonian function (H), length(N), and number of iterations (n)
            and returns the spins at each iteration. If init_cond = True, start with constant initial lattice.
            If False, then initialize with random spins. N_store determines how many samples will be stored.
            N_init determines how many initial samples will be skipped.
        '''
        if N_store: # if a specific number of samples
          self.spin_tracker = np.zeros((N_store,self.N,self.N)) # limited sample storage
          step = np.floor((n_itrs - N_init)/N_store).astype(int) # skip as many samples to prevent correlation
        else:
          self.spin_tracker = np.zeros((n_itrs,self.N,self.N)) # unlimited sample storage
          step = 1 # store all samples
        if init_cond: # initialize lattice
          spins = np.ones((self.N,self.N))
        else:
          spins = np.random.choice([1,-1],size=(self.N, self.N))
        sample_counter = 0
        for n in tqdm(range(n_itrs)):
            s = tuple(np.random.randint(0,self.N,(2))) # two indices required for spin position
            deltaH = self.H_change(spins,s)
            r = np.exp(-self.beta*(deltaH))
            z = np.random.uniform()
            if r > z: # determining whether to keep flipped spin
                spins[s] = -spins[s]
            if (n >= N_init) and (n % step == 0):
              self.spin_tracker[sample_counter,:,:] = spins
              sample_counter += 1
            # If not enough separation for another sample, end simulation
            if (n_itrs - n) < step:
              break
# My own class representation of a deep belief network
# Got some ideas from https://github.com/2015xli/DBN
class DBN:
  # Initializing RBMs that constitute DBN
  def __init__(self, n_rbms, n_components, learning_rates, n_iter=100, batch_size=10):
    self.rbms = [BernoulliRBM(n_components=n,learning_rate=lr,n_iter=n_iter, batch_size=batch_size, verbose=1) for n,lr in zip(n_components,learning_rates)] # initializing RBMs
    self.n_rbms = n_rbms
    self.n_components = n_components
  # Single backward step for H of a given RBM (row vectors)
  def rbm_backward(self, H, rbm_idx):
    signal = self.rbms[rbm_idx].intercept_visible_.reshape((1, len(self.rbms[rbm_idx].intercept_visible_))) + np.matmul(H,self.rbms[rbm_idx].components_)
    Vp = (1 + np.exp(-signal))**-1
    return Vp # sampled components of the previous layer
  # Network forwarding through each RBM
  def forward(self, Hs):
    # Make visible unit the 'previous hidden unit' samples
    for i in range(self.n_rbms):
      Hs = self.rbms[i].transform(Hs) # get probability distribution over hidden units
      # Hs = np.random.binomial(n=1, p=Hs, size=Hs.shape)
    return Hs # return outputs and probabilities of all hidden layers
  # Network sending signal backwards through each RBM
  def backward(self, Hs):
    for i in reversed(range(self.n_rbms)): # iteratively sample backwards until visible units
      Hs = self.rbm_backward(Hs, i)
    return Hs
  # One step forward for a given input, then backward to obtain the corresponding reconstruction
  def reconstruct(self, V):
    if V.ndim == 1: # make 1dim arrays 2d
      V = V.reshape(1,len(V))
    V = self.forward(V)
    V = self.backward(V)
    return V
  # Training all constituents RBMs
  def fit(self, Hs):
    print('Training DBN Layers ... \n')
    # Make visible unit the 'previous hidden unit' samples
    for i in range(self.n_rbms):
      print('Training RBM {} ... \n'.format(i+1))
      self.rbms[i].fit(Hs)
      Hs = self.rbms[i].transform(Hs) # get probability distribution over hidden units
      # Hs = np.random.binomial(n=1, p=Hs, size=Hs.shape) # get sampled hidden units
  def get_weights(self): # return a list of weight matrices for each RBM
    Ws = []
    for i in range(self.n_rbms):
      Ws.append(self.rbms[i].components_)
    return Ws
# Converting MATLAB functions (by Andrej Karpathy) to Python Class 
class RBM:
    '''Learn RBM with Bernoulli hidden and visible units by Andrej Karpathy
    based on implementation of Kevin Swersky and Ruslan Salakhutdinov
    
    INPUTS: 
    X              ... data. should be binary, or in [0,1] to be interpreted 
                ... as probabilities
    numhid         ... number of hidden layers
    
    additional inputs (specified as name value pairs or in struct)
    method         ... CD or SML 
    eta            ... learning rate
    momentum       ... momentum for smoothness amd to prevent overfitting
                ... NOTE: momentum is not recommended with SML
    maxepoch       ... # of epochs: each is a full pass through train data
    avglast        ... how many epochs before maxepoch to start averaging
                ... before. Procedure suggested for faster convergence by
                ... Kevin Swersky in his MSc thesis
    penalty        ... weight decay factor
    batchsize      ... The number of training instances per batch
    verbose        ... For printing progress
    anneal         ... Flag. If set true, the penalty is annealed linearly
                ... through epochs to 10 of its original value
    
    OUTPUTS:
    model.type     ... Type of RBM (i.e. type of its visible and hidden units)
    model.W        ... The weights of the connections
    model.b        ... The biases of the hidden layer
    model.c        ... The biases of the visible layer
    model.top      ... The activity of the top layer, to be used when training
                ... DBN's
    errors         ... The errors in reconstruction at every epoch
    '''
    def __init__(self, numhid, method='CD', lr=0.01, \
        momentum=0,maxepoch=100,penalty=0, \
        batchsize=100, verbose=True,anneal=False):
        # Initializing rbm parameters
        self.numhid = numhid # number of hidden units
        self.method = method  # learning method
        self.lr = lr    # learning rate
        self.momentum = momentum  # momentum
        self.maxepoch = maxepoch  # number of training epochs
        self.penalty = penalty  # for L1 reg
        self.verbose = verbose  # verbosity
        self.anneal = anneal    # whether to add anneal
        self.batchsize = batchsize  # size of a batch
        self.avglast = 5
        self.avgstart = maxepoch - self.avglast
        self.oldpenalty = penalty
    # Logistic function
    def logistic(self,X):
        return (1+np.exp(-X))**-1
    # Training RBM based on data
    def fit(self,X):
        if self.verbose:
          print('Preprocessing data...\n')
        # Create batches
        N,d = X.shape
        numcases = N # number of samples
        numdims = d # dimensionality of the vector
        numbatches= np.ceil(N/self.batchsize).astype('int64')
        groups= repmat(list(range(numbatches)), 1, self.batchsize)
        groups = groups[:N]
        perm = np.random.permutation(N) # shuffling samples to put into batches
        groups = groups[0,perm]
        # Initializing batchdata container
        batchdata = []
        for idx in range(numbatches):
            batchdata.append(X[groups==idx,:])
        # Train RBM
        W = 0.1*np.random.randn(numdims,self.numhid)
        c = np.zeros((1,numdims))
        b = np.zeros((1,self.numhid))
        ph = np.zeros((numcases,self.numhid))
        nh = np.zeros((numcases,self.numhid))
        phstates = np.zeros((numcases,self.numhid))
        nhstates = np.zeros((numcases,self.numhid))
        negdata = np.zeros((numcases,numdims))
        negdatastates = np.zeros((numcases,numdims))
        Winc  = np.zeros((numdims,self.numhid))
        binc = np.zeros((1,self.numhid))
        cinc = np.zeros((1,numdims))
        Wavg = W
        bavg = b
        cavg = c
        t = 1
        errors=np.zeros(self.maxepoch)
        # Begin training
        for epoch in range(self.maxepoch):
            # Tracking time taken
            start_t = time()
            errsum=0
            if self.anneal:
                # Apply linear weight penalty decay
                penalty= self.oldpenalty - 0.9*epoch/self.maxepoch*self.oldpenalty
            
            for batch in range(numbatches):
                numcases,numdims= batchdata[batch].shape
                data = batchdata[batch]
                # Go up the rbm
                ph = self.logistic(np.matmul(data,W) + repmat(b,numcases,1))
                phstates = ph > np.random.rand(numcases,self.numhid)
                if self.method == 'SML':
                    if (epoch == 0) and (batch == 0):
                        nhstates = phstates
                elif self.method == 'CD':
                    nhstates = phstates
                # Go down
                negdata = self.logistic(np.matmul(nhstates,W.T) + repmat(c,numcases,1))
                negdatastates = negdata > np.random.rand(numcases,numdims)
                # Go up one more time
                nh = self.logistic(np.matmul(negdatastates,W) + repmat(b,numcases,1))
                nhstates = nh > np.random.rand(numcases,self.numhid)
                # Update weights and biases
                dW = (np.matmul(data.T,ph) - np.matmul(negdatastates.T,nh))
                dc = np.sum(data,axis=0) - np.sum(negdatastates,axis=0)
                db = np.sum(ph,axis=0) - np.sum(nh,axis=0)
                Winc = self.momentum*Winc + self.lr*(dW/numcases - self.penalty*W)
                binc = self.momentum*binc + self.lr*(db/numcases)
                cinc = self.momentum*cinc + self.lr*(dc/numcases)
                W = W + Winc
                b = b + binc
                c = c + cinc
                if epoch > self.avgstart:
                    # Apply averaging
                    Wavg = Wavg - (1/t)*(Wavg - W)
                    cavg = cavg - (1/t)*(cavg - c)
                    bavg = bavg - (1/t)*(bavg - b)
                    t = t+1
                else:
                    Wavg = W
                    bavg = b
                    cavg = c
                # accumulate reconstruction error
                err= np.sum((data-negdata)**2)
                errsum = err + errsum  
            errors[epoch]=errsum
            if self.verbose:
                print('Ended epoch {}/{}. Error: {}, Time: {}'.format(epoch+1, self.maxepoch, errsum,time()-start_t))
        # Save relevant properties of the model as attributes
        self.W = Wavg
        self.b = bavg
        self.c = cavg
        return errors
    # Going from V to H
    def V2H(self, V):
        n,_ = V.shape
        return self.logistic(np.matmul(V,self.W) + repmat(self.b,n,1))
    # Going from H to V
    def H2V(self,H):
        n,_ = H.shape
        return self.logistic(np.matmul(H,self.W.T) + repmat(self.c,n,1))
    # RBM reconstruction
    def reconstruct(self, V):
        return self.H2V(self.V2H(V))
# Second DBN class
class DBN2:
    # Initializing RBMs that constitute DBN
    def __init__(self, n_components, lr, maxepoch=100, batchsize=100, momentum=0,penalty=0,verbose=True,anneal=False,method='CD'):
      # Initializing RBMs
      self.n_rbms = len(n_components)
      self.rbms = []
      for numhid in n_components:
          self.rbms.append(RBM(numhid=numhid,lr=lr,momentum=momentum,maxepoch=maxepoch,\
          penalty=penalty, batchsize=batchsize,method=method))
      # self.n_components = n_components
    # Network forwarding through each RBM
    def forward(self, Hs):
        # Make visible unit the 'previous hidden unit' samples
        for i in range(self.n_rbms):
            Hs = self.rbms[i].V2H(Hs) # get probability distribution over hidden units
            # Hs = np.random.binomial(n=1, p=Hs, size=Hs.shape)
        return Hs # return outputs and probabilities of all hidden layers
    # Network sending signal backwards through each RBM
    def backward(self, Hs):
        for i in reversed(range(self.n_rbms)):
            Hs = self.rbms[i].H2V(Hs) # get probability distribution over hidden units
            # Hs = np.random.binomial(n=1, p=Hs, size=Hs.shape)
        return Hs # return outputs and probabilities of all hidden layers
    # One step forward, then backward to obtain the corresponding reconstruction
    def reconstruct(self, V, plot=False):
        if V.ndim == 1: # make 1dim arrays 2d
            V = V.reshape(1,len(V))
        H = self.forward(V)
        Vrec = self.backward(H)
        if plot:
          N = round(np.sqrt(len(V))).astype('uint64')
          fig = plt.figure()
          plt.imshow(V.reshape((N,N)),np.ones(N,10),Vrec.reshape((N,N)) )
          plt.show()
        return Vrec
    # Training all constituents RBMs
    def fit(self, Hs):
        print('Training DBN Layers ... \n')
        # Make visible unit the 'previous hidden unit' samples
        for i in range(self.n_rbms):
            print('Training RBM {} ... \n'.format(i+1))
            self.rbms[i].fit(Hs)
            Hs = self.rbms[i].V2H(Hs) # get probability distribution over hidden units
            # Hs = np.random.binomial(n=1, p=Hs, size=Hs.shape) # get sampled hidden units
    def get_weights(self): # return a list of weight matrices for each RBM
        Ws = []
        for i in range(self.n_rbms):
            Ws.append(self.rbms[i].W)
        return Ws

############# FUNCTIONS #############

# Extracting images from zip (given a number of images and image dims)
def extract_images(file_name, shape):
  n_imgs,height,width = shape
  n_train, n_test = round(0.8*n_imgs), round(0.2*n_imgs)
  training, test = np.zeros((n_train,height,width)), np.zeros((n_test,height,width))
  with zipfile.ZipFile(file_name, 'r') as my_zip:
    for idx in tqdm(range(n_imgs)): # 1st idx is number of images
      img_name = str(idx) + '.png'
      my_zip.extract(img_name) # extract image file
      if idx < n_train:
        training[idx,:,:] = np.array(Image.open(img_name)) # add image to data array
      else:
        test[idx-n_train,:,:] = np.array(Image.open(img_name)) # add image to data array
      os.remove(img_name)
    return training,test
# Preprocessing
def preprocess(training,test):
   n_train,height,width = training.shape
   n_test,_,_ = test.shape
   training = training.reshape((n_train,height*width)) # flattens image into 1D vector
   training[training > 0] = 1 # ensures image is in binary format
   test = test.reshape((n_test,height*width)) # flattens image into 1D vector
   test[test > 0] = 1 # ensures image is in binary format
   return training, test
# Reconstruction error on unseen data
def reconstruction_error(dbn,data):
  reconstruction = dbn.reconstruct(data)
  return ((data - reconstruction)**2).mean()
# Returns belief net without top layer
def remove_layers(belief_net, n_rbms_new):
    new_n_components = belief_net.n_components[0:n_rbms_new]
    # Make new DBN object
    new_belief_net = DBN(n_rbms_new, new_n_components, learning_rates=np.ones(n_rbms_new), n_iter=1000,batch_size=1000)
    for idx in range(n_rbms_new):
        new_belief_net.rbms[idx] = belief_net.rbms[idx]
    return new_belief_net
# Returns the reconstruction error from each of the DBN layers
def all_layer_error(belief_net, data):
    errors = []
    for idx in range(1, belief_net.n_rbms):
        errors.append(reconstruction_error(belief_net,data))
        belief_net = remove_layers(belief_net, belief_net.n_rbms - idx)
    return errors
# Computes the receptive field of a single spin at a given hidden layer
def eff_receptive_field(belief_net, n_spin, l=1, plot=True, opt=True):
    ''' Computes the effective receptive field of a single spin of the model
        and plots the results
        Inputs:
        belief_net: Trained DBN model
        n_spin: Spin index of the given hidden layer
        l: layer number, generally set to first hidden layer ( l > 0 )
        plot: Boolean that determines whether to suppress effective receptive field plots or not
        opt: Determines which DBN class which will be used
    '''
    # Extract weight matrices from belief net required for computation
    for idx in range(l):
        if idx == 0:
            if opt:
              r = belief_net.rbms[idx].components_.T
            else:
              r = belief_net.rbms[idx].W
        else:
            if opt:
              r = np.matmul(r, belief_net.rbms[idx].components_.T)
            else:
              r = np.matmul(r, belief_net.rbms[idx].W)
    ERF = r[:,n_spin] # effective receptive field is single column from weight matrix
    # If chosen to plot, make plot
    if plot:
        N = round(np.sqrt(len(ERF))).astype('int64')
        fig = plt.figure()
        plt.imshow(ERF.reshape((N,N)) , cmap='hot')
        plt.axis('off')
        plt.colorbar()
        plt.show()
    return ERF
# Plotting all receptive fields of a given layer in a grid
def plot_erfs(belief_net, l, N=64, numhid=None, plot=True):
  # numhid must be a perfect square
  # Get number of hidden units
  if not numhid:
    numhid = belief_net.rbms[l-1].numhid
  # Iterate over all hidden spins in the given layer
  side = round(np.sqrt(numhid)).astype('int64')
  perms = np.random.permutation(numhid)
  # Make image array to contain all erfs
  img = np.zeros((side*N, side*N))
  for idx in range(side):
    for jdx in range(side):
      img[N*idx:N*(idx+1), N*jdx:N*(jdx+1)] = \
        eff_receptive_field(belief_net, perms[idx*side+jdx], l=l, plot=False, opt=False).reshape((N,N))
  # Making grid
  img_max = np.amax(img)
  for idx in range(1,side):
    print(N*idx-1)
    img[N*idx-1:N*idx+4,:] = img_max.copy()
    img[:, N*idx-1:N*idx+4] = img_max.copy()
  # Plotting field
  if plot:
    fig = plt.figure()
    plt.imshow(img, cmap='hot')
    plt.axis('off')
    plt.colorbar()
    plt.show()
  else:
    return img
