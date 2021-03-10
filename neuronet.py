# -*- coding: utf-8 -*-
"""
Neuronal network using Izhikevich model
"""
from parameters import *
from neuroastro_interaction import *
import gv
import numpy as np
import random
import matplotlib.pyplot as plt
import numba

""" neuron network random connections """
def neuro_connection_uniform():
    gv.adj_mat = np.zeros((Mneuro*Nneuro, Mneuro*Nneuro))
    for i in range(Mneuro*Nneuro):
        indices = [j for j in range(Mneuro*Nneuro)]
        indices.remove(i)
        indices_sampled = random.sample(indices, N_out)
        gv.adj_mat[i, indices_sampled] = 1
    #return adj_mat

""" neuron network connections with exponential distribution """
def neuro_connection():
    #gv.adj_mat = np.zeros((Mneuro*Nneuro, Mneuro*Nneuro), 'int8')
    ties_stock = 10*N_out
    for i in range(Mneuro):
        for j in range(Nneuro):
            XY = np.zeros((2, ties_stock), 'int8')
            R = np.random.exponential(lambda_exp, size = ties_stock)
            fi = 2 * np.pi * np.random.rand(ties_stock, )
            pos = R * np.cos(fi)
            XY[0,:] = pos.astype(int)
            pos = R * np.sin(fi)
            XY[1,:] = pos.astype(int)
            XY1 = np.unique(XY.T, axis=0)
            np.random.shuffle(XY1)
            n = 0
        
            for k in range(XY1.shape[0]):
                pp = 0
                x = i + XY1[k, 0]
                y = j + XY1[k, 1]
                if i==x and j==y:
                    pp = 1
                if x>=0 and y>=0 and x<Mneuro and y<Nneuro and pp==0:
                    x2 = np.ravel_multi_index((x,y), dims=(Mneuro,Nneuro), order='C')
                    gv.adj_mat[j+i*Nneuro, x2] = 1
                    n = n + 1
                if n>=N_out:
                    break
    gv.adj_mat = gv.adj_mat.T


""" make specific input"""
def make_I_app(img_gray, stage=("learn", "test")):
    if stage == "learn":
        return I_app_learn * img_gray
    elif stage == "test":
        return I_app_test * img_gray
    else:
        print("Two options available for stage argument: learn or test")

""" make Poisson noise input"""
def make_Poisson_noise():
    I_noise = np.random.uniform(0, 1, (Mneuro, Nneuro))
    I_noise = np.where(I_noise < f_bg * dt, 1, 0)
    amplitude = np.random.uniform(-A_stim, A_stim, (Mneuro, Nneuro))
    return I_noise * amplitude

""" calculate g_syn determined by calcium level, return value must be a vector"""
"""
def calcul_g_syn(astro_Ca):
    neuro_Ca = calcul_neuro_Ca(astro_Ca, gv.adj_ts_an)
    return np.where(neuro_Ca > Ca_thr, eta+v_Ca_star, eta).reshape((Mneuro*Nneuro))
"""
"""
def calcul_g_syn(astro_Ca):
    g_syn_mat = np.zeros((Mneuro*Nneuro, Mneuro*Nneuro))
    for j in range(Mneuro*Nneuro):
        for k in range(Mneuro*Nneuro):
            if gv.adj_mat[j, k] == 1:
                xj = j//Mneuro
                yj = j%Mneuro
                xk = k//Nneuro
                yk = k%Nneuro
                astro_j = gv.adj_ts_an[xj, yj]
                astro_k = gv.adj_ts_an[xk, yk]
                astro_sum = astro_j + astro_k
                astro_sum = np.where(astro_sum>1, 1, 0)
                syn_Ca = (astro_Ca * astro_sum).sum() 
                if syn_Ca > Ca_thr:
                    g_syn_mat[j, k] = eta+v_Ca_star
                else:
                    g_syn_mat[j, k] = eta
    return g_syn_mat      
"""

@numba.jit(nopython=True)
def calcul_g_syn(astro_Ca, adj1, adj2):
    g_syn_mat = np.zeros((Mneuro*Nneuro, Mneuro*Nneuro))
    for j in range(Mneuro*Nneuro):
        for k in range(Mneuro*Nneuro):
            if adj1[j, k] == 1:
                xj = j//Mneuro
                yj = j%Mneuro
                xk = k//Nneuro
                yk = k%Nneuro
                astro_j = adj2[xj, yj]
                astro_k = adj2[xk, yk]
                astro_sum = astro_j + astro_k
                astro_sum = np.where(astro_sum>1, 1, 0)
                syn_Ca = (astro_Ca * astro_sum).sum() 
                if syn_Ca > Ca_thr:
                    g_syn_mat[j, k] = eta+v_Ca_star
                else:
                    g_syn_mat[j, k] = eta
    return g_syn_mat      
           
    
""" calculate I_syn"""
"""
def calcul_I_syn(g_syn, v):
    v_vec = v.reshape((Mneuro*Nneuro))
    denom = 1 + np.exp(-v_vec/k_syn)  
    prod = np.dot(gv.adj_mat, 1/denom)
    v_mat = np.multiply(g_syn*(E_syn - v_vec), prod) 
    return v_mat.reshape((Mneuro, Nneuro))
"""
def calcul_I_syn(g_syn, v):
    v_vec = v.reshape((Mneuro*Nneuro))
    denom = 1 + np.exp(-v_vec/k_syn)
    sum_mat = np.dot(g_syn, 1/denom)
    v_mat = np.multiply(E_syn - v_vec, sum_mat)
    return v_mat.reshape((Mneuro, Nneuro))



""" update V and U in each step"""
def step_V_U(step, I_app):
    old_v = np.copy(gv.V_all[step-1])
    old_u = np.copy(gv.U_all[step-1])
    temp_v = np.where(old_v >= spiking_thres, c, old_v)
    temp_u = np.where(old_v >= spiking_thres, old_u + d, old_u)
    du = a * (b* temp_v - temp_u)
    g_syn_t = calcul_g_syn(gv.Ca_all[step-1], gv.adj_mat, gv.adj_ts_an)
    #g_syn_t = 0.025
    dv = (0.04*temp_v*temp_v + 5*temp_v - temp_u + 140 
               + calcul_I_syn(g_syn_t, temp_v) + I_app)
    new_v = temp_v + dv * dtn
    new_u = temp_u + du * dtn
    new_v = np.where(new_v > spiking_thres, spiking_thres, new_v)
    gv.V_all[step] = new_v
    gv.U_all[step] = new_u
    
""" update glutamate in each step"""
def step_neuro_G(step):
    """
    dG = -alpha_G * gv.G_neuro_all[step-1] + beta_G / (1 + np.exp(-gv.V_all[step-1] / V_G))
    gv.G_neuro_all[step] = gv.G_neuro_all[step-1] + dG * dt
    """
    dG = -alpha_glu * gv.G_neuro_all[step-1] + k_glu * np.where(gv.V_all[step-1] >= 30, 1, 0)
    gv.G_neuro_all[step] = gv.G_neuro_all[step-1] + dG * dt

""" initialize U, V concentration"""
"""
def initialize_global_variables():
    V_all = np.zeros((num_steps_n, Mneuro, Nneuro))
    U_all = np.zeros((num_steps_n, Mneuro, Nneuro))
    G_neuro_all = np.zeros((num_steps_n, Mneuro, Nneuro))
    V_all[0] = np.full((Mneuro, Nneuro), c)
    U_all[0] = np.full((Mneuro, Nneuro), b * c)
"""
    
""" simulate the dynamics of the neuronal network for time T"""
"""
def simulate_neuro_network():
    for step in range(1, num_steps_n):
        step_V_U(step)
        step_neuro_G(step)
"""

"""
if __name__ == "__main__":

    g_syn = 0.025
    neuro_connection()
    initialize_global_variables()
    simulate_neuro_network()
"""