# SCRIPT FOR SOLVING THE MERTON PROBLEM

#%% import needed packages

import DGM
import tensorflow.compat.v1 as tf
import tensorflow as tf2
import numpy as np
from utilities import CartPoleModel
#import matplotlib.pyplot as plt

#%% Parameters
tf.compat.v1.disable_eager_execution()

# Cartpole problem parameters
T = 1            # terminal time (investment horizon)
X_d = 4          # the dimension of X

# Solution parameters (domain on which to solve PDE)
t_low = 0 + 1e-10    # time lower bound

# cannot be None
X_low = 0         # space lower bound
X_high = 100        # space upper bound

# cartpole parameters
m = 0.1
M = 1
L = 1
D = 4
Q = np.identity(D)
Qt = np.identity(D)
snoise = 0.01
R = 1

# neural network parameters
num_layers = 3
nodes_per_layer = 50
starting_learning_rate = 0.01

# Training parameters
sampling_stages  = 5000   # number of times to resample new time-space domain points
steps_per_sample = 100    # number of SGD steps to take before re-sampling

# Sampling parameters
nSim_interior = 1000
nSim_terminal = 10

# Plot options
n_plot = 41  # Points on plot grid for each dimension

# Save options
saveOutput = False
saveName   = 'Cartpole'
saveFigure = False
figureName = 'Cartpole'

#### HELPER METHODS
def getTFUtils():
    matmul = tf.linalg.matmul
    multiply = tf.math.multiply
    rowsum = lambda x : tf.math.reduce_sum(x, axis = 1);
    return (matmul, multiply, rowsum)
    
def BtXgradV(Bx, Vgrad): # M x D, M x D
    matmul, multiply, rowsum = getTFUtils()
    print('Bx=%s' % Bx)
    print('Vgrad=%s' % Vgrad)
    mul = tf.multiply(Bx, Vgrad)
    print('multiply(Bx, Vgrad)=%s' % mul)
    return rowsum(multiply(Bx, Vgrad))
    
def quadraticForm(X, PSD): # M x D, D x D
    matmul, multiply, rowsum = getTFUtils()
    return rowsum(multiply(matmul(X, PSD), X)) # M x

def u(x):
    ''' Compute u(x)
    
    Args:
        x: space points
    '''
    
    # V(x, T) = 1/2 * x^T * Q_T * x
    Norm_x = tf.norm(x, ord = 2, axis=None)
    return 0.5 * quadraticForm(x, Qt)


#%% Sampling function - randomly sample time-space pairs

def sampler(nSim_interior, nSim_terminal):
    ''' Sample time-space points from the function's domain; points are sampled
        uniformly on the interior of the domain, at the initial/terminal time points
        and along the spatial boundary at different time points. 
    
    Args:
        nSim_interior: number of space points in the interior of the function's domain to sample 
        nSim_terminal: number of space points at terminal time to sample (terminal condition)
    ''' 
    
    # Sampler #1: domain interior    
    t_interior = np.random.uniform(low=t_low, high=T, size=[nSim_interior, 1])
    X_interior = np.random.uniform(low=X_low, high=X_high, size=[nSim_interior, X_d])

    # Sampler #2: spatial boundary
    # no spatial boundary condition for this problem
    
    # Sampler #3: initial/terminal condition
    t_terminal = T * np.ones((nSim_terminal, 1))
    X_terminal = np.random.uniform(low=X_low, high=X_high, size = [nSim_terminal, X_d])
    
    return t_interior, X_interior, t_terminal, X_terminal

#%% Loss function for Merton Problem PDE

def loss(model, cartpoleUtil, t_interior, X_interior, t_terminal, X_terminal):
    ''' Compute total loss for training.
    
    Args:
        model:      DGM model object
        t_interior: sampled time points in the interior of the function's domain
        X_interior: sampled space points in the interior of the function's domain
        t_terminal: sampled time points at terminal point (vector of terminal times)
        X_terminal: sampled space points at terminal time
    ''' 
    
    # Loss term #1: PDE
    # compute function value and derivatives at current sampled points
    # \frac{\partial u}{\partial t}(t, x) + \Delta u(t, x) - \lambda \| \nabla u(t, x) \|^2 = 0
    # => V_t + V_xx - lambda * L2_norm(V_x)^2
    matmul, multiply, rowsum = getTFUtils()
    
    V = model(t_interior, X_interior)
    V_t = tf.gradients(V, t_interior)[0]
      
    print('V_t=%s' % V_t)
    
    # f = phi1 + phi2
    const = tf.constant
    print('X_interior=%s' % X_interior)
    phi1 = tf.constant(0.5) * quadraticForm(X_interior, Q)
    A, B = cartpoleUtil.f(X_interior)
    
    print('A=%s' % A)
    print('B=%s' % B)
    
    V_x = tf.gradients(V, X_interior)[0]
    
    print('V_x=%s' % V_x)
    Bt_gradV = BtXgradV(B, V_x)
    print('Bt_gradV=%s' % Bt_gradV)
    phi2 = const(0.5) * tf.square(Bt_gradV) / const(R * 1.0)
    print('phi1=%s' % phi1)
    print('phi2=%s' % phi2)
    f = phi1 + phi2
    
    # mu^T
    uinput = const(1.0 / R) * Bt_gradV
    inputs = tf.repeat(tf.reshape(uinput, (-1, 1)), repeats = D, axis = 1)
    print('inputs=%s' % inputs)
    mu_t = A - multiply(inputs, B)
    
    V_xx = tf.gradients(V_x, X_interior)[0]
    
    print('V_t=%s' % V_t)
    print('f=%s' % f)
    print('mu_t=%s' % mu_t)
    print('V_x=%s' % V_x)
    print('snoise=%s' % snoise)
    print('V_xx=%s' % V_xx)
    
    mul = rowsum(multiply(mu_t, V_x))
    diff_V = V_t + f + mul + 0.5 * (snoise**2) * tf.linalg.trace(V_xx)

    # compute average L2-norm of differential operator
    L1 = tf.reduce_mean(tf.square(diff_V)) 
    
    # Loss term #2: boundary condition
    # no boundary condition for this problem
    
    # Loss term #3: initial/terminal condition
    target_terminal = u(X_terminal)
    fitted_terminal = model(t_terminal, X_terminal)
    
    L3 = tf.reduce_mean( tf.square(fitted_terminal - target_terminal) )

    return L1, L3
    

#%% Set up network

# initialize DGM model (last input: space dimension = 1)
model = DGM.DGMNet(nodes_per_layer, num_layers, X_d)
cartpoleUtil = CartPoleModel(m, M, L)

# tensor placeholders (_tnsr suffix indicates tensors)
# inputs (time, space domain interior, space domain at initial time)
#t_interior_tnsr = tf.placeholder(tf.float32, [None,1])
#X_interior_tnsr = tf.placeholder(tf.float32, [nSim_interior,X_d])
#t_terminal_tnsr = tf.placeholder(tf.float32, [None,1])
#X_terminal_tnsr = tf.placeholder(tf.float32, [nSim_terminal,X_d])

# for tensorflow 2
t_interior_tnsr = tf2.keras.Input(dtype=tf.dtypes.float32, shape=[1])
X_interior_tnsr = tf2.keras.Input(dtype=tf.dtypes.float32, shape=[X_d], batch_size=nSim_interior)
t_terminal_tnsr = tf2.keras.Input(dtype=tf.dtypes.float32, shape=[1])
X_terminal_tnsr = tf2.keras.Input(dtype=tf.dtypes.float32, shape=[X_d], batch_size=nSim_terminal)

# loss 
L1_tnsr, L3_tnsr = loss(model, cartpoleUtil, t_interior_tnsr, X_interior_tnsr, t_terminal_tnsr, X_terminal_tnsr)
loss_tnsr = L1_tnsr + L3_tnsr

# set optimizer - NOTE THIS IS DIFFERENT FROM OTHER APPLICATIONS!
global_step = tf.Variable(0, trainable=False)
#learning_rate = tf.train.exponential_decay(starting_learning_rate, global_step, 100000, 0.96, staircase=True)
learning_rate = tf.train.exponential_decay(starting_learning_rate, global_step, 100000, 0.96, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_tnsr)

# initialize variables
init_op = tf.global_variables_initializer()

# open session
sess = tf.Session()
sess.run(init_op)

#%% Train network
# initialize loss per training
loss_list = []

# for each sampling stage
for i in range(sampling_stages):
    
    # sample uniformly from the required regions
    t_interior, X_interior, t_terminal, X_terminal = sampler(nSim_interior, nSim_terminal)
    
    # for a given sample, take the required number of SGD steps
    for _ in range(steps_per_sample):
        loss,L1,L3,_ = sess.run([loss_tnsr, L1_tnsr, L3_tnsr, optimizer],
                                feed_dict = {t_interior_tnsr:t_interior, X_interior_tnsr:X_interior, t_terminal_tnsr:t_terminal, X_terminal_tnsr:X_terminal})
        loss_list.append(loss)
    
    print(loss, L1, L3, i)

# save outout
if saveOutput:
    saver = tf.train.Saver()
    saver.save(sess, './SavedNets/' + saveName)
