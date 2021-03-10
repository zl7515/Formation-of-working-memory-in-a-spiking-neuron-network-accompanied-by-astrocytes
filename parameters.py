# -*- coding: utf-8 -*-
"""
parameters for the model
"""

####### parameters #######

""" structure parameters"""
#### height and width must be equal ####
Mastro = 26
Nastro = 26
Mneuro = 79
Nneuro = 79
N_out = 40  # out-degree of each neuron 
lambda_exp = 5  # exponential distribution parameter

""" neuron astrocyte interaction parameters"""
Na = 4 # square root of number of neurons connected to each astrocyte
Na_overlap = 1 
alpha_glu = 10
alpha_Glu = 5
k_glu = 600
A_glu = 5.0
G_thr = 0.1
F_act = 0.5
F_astro = 0.375
v_Ca_star = 0.5
Ca_thr = 0.15

alpha_G = 10 # 32
beta_G = 600 # 295
V_G = 0.5

""" neuronal network parameters"""
spiking_thres=30
a = 0.1
b = 0.2
c = -65
d = 2
eta = 0.025
E_syn = 0
k_syn = 0.2
I_app_learn = 80
I_app_test = 8
A_stim = 0.01
f_bg = 1.5

""" astrocytic network model parameters"""
c0 = 2.0  
c1 = 0.185  
v1 = 6  
v2 = 0.11
v3 = 2.2 #0.9 
v4 = 0.3
v6 = 0.2
k1 = 0.5
k2 = 1.0
k3 = 0.1
k4 = 1.1
d1 = 0.13
d2 = 1.049
d3 = 0.9434
d5 = 0.082  # 0.2
a2 = 0.2
dCa = 0.05
dIP3 = 0.1
IP3_0 = 0.16
tau_IP3 = 7.143
Ca_init = 0.2
IP3_init = 3 #0.820204
h_init = 0.886314 #0.886314
alpha = 0.8

""" simulation parameters"""
dt = 0.0005
dtn = 1000*dt
T = 0.2
num_steps = int(T/dt)+1
##########################