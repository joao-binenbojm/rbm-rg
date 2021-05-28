import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.constants import mu_0
from tqdm import tqdm

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
    def make_gif(self):
        # image_list = [self.spin_tracker[i,:].reshape(1,self.N).astype('int8') for i in range(0,len(self.spin_tracker),round(len(self.spin_tracker)/100))]
        # imageio.mimwrite('animated_from_images.gif', image_list)
        image_list = []
        imgs = self.spin_tracker.copy()
        for i in range(0,len(self.spin_tracker)):
          imgs[i,-1] = -imgs[i,-1] # flipping rightmost pixel to make gif work
          imgs[i,:] = (imgs[i,:]>0).astype(np.uint8)
          image_list.append(imgs[i,:].reshape(1,self.N))
        imageio.mimwrite('animated_from_images.gif', image_list)
            
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
    def make_gif(self):
      image_list = []
      imgs = self.spin_tracker.copy()
      for i in range(0,len(self.spin_tracker)):
        imgs[i,-1,-1] = -imgs[i,-1,-1] # flipping bottom corner pixel to prevent error
        imgs[i,:,:] = (imgs[i,:,:]>0).astype(np.uint8)
        image_list.append(imgs[i,:,:])
      imageio.mimwrite('animated_from_images.gif', image_list)

# My own class representation of a deep belief network
class DBN:
  def __init__(self, n_rbms, n_components, learning_rates, n_iter=100, batch_size=50):
    self.rbms = [BernoulliRBM(n_components=n,learning_rate=lr,n_iter=n_iter, batch_size=batch_size) \
            for n,lr in zip(n_components,learning_rates)] # initializing RBMs
    self.n_rbms = n_rbms
    self.n_components = n_components
  def fit(self, X):
    self.data = X
    rbm_input = X
    for i in range(self.n_rbms):
      self.rbms[i].fit(rbm_input) # training the given RBM
      rbm_input = self.rbms[i].h_samples_ # making the input to the next RBM the samples from the given RBM