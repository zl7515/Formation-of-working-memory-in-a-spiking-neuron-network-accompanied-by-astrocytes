# -*- coding: utf-8 -*-
"""
Astrocyte network using Li-Rinzel model
"""

from parameters import *
from neuroastro_interaction import *
import gv
import numpy as np
import matplotlib.pyplot as plt
import numba
import time

"""compute J_glu for astrocytes according to glutamates released by neurons"""
def calcul_J_glu(neuro_G):
    astro_G = calcul_astro_G(neuro_G)
    return np.where(astro_G > 4, A_glu, 0)


""" compute flux of Ca and IP3 into each astrocyte in each time step"""
@numba.jit(nopython=True) # use parallel computing when Mastro and Nastro is large
def step_diff_Ca_IP3(Ca, IP3):
    # astrocyte network is of size (M,N)
    diff_Ca = np.zeros((Mastro, Nastro))
    diff_IP3 = np.zeros((Mastro, Nastro))
    ###### compute flux of Ca and IP3 into each astrocyte ######
    for j in range(Mastro):
        for k in range(Nastro):
            if j==0 and k==0: # top left
                diff_Ca[j,k] = Ca[j + 1,k] + Ca[j,k + 1] - 2 * Ca[j,k]
                diff_IP3[j,k] = IP3[j + 1,k] + IP3[j,k + 1] - 2 * IP3[j,k]
            elif j==Mastro-1 and k==Nastro-1: # bottom right
                diff_Ca[j,k] = Ca[j - 1,k] + Ca[j,k - 1] - 2 * Ca[j,k]
                diff_IP3[j,k] = IP3[j - 1,k] + IP3[j,k - 1] - 2 * IP3[j,k]
            elif j==0 and k==Nastro-1: # top right
                diff_Ca[j,k] = Ca[j + 1,k] + Ca[j,k - 1] - 2 * Ca[j,k]
                diff_IP3[j,k] = IP3[j + 1,k] + IP3[j,k - 1] - 2 * IP3[j,k]
            elif j==Mastro-1 and k==0: #bottom left
                diff_Ca[j,k] = Ca[j - 1,k] + Ca[j,k + 1] - 2 * Ca[j,k]
                diff_IP3[j,k] = IP3[j - 1,k] + IP3[j,k + 1] - 2 * IP3[j,k]
            elif j==0: # top edge
                diff_Ca[j,k] = Ca[j + 1, k] + Ca[j, k - 1] + Ca[j,k + 1] - 3 * Ca[j,k]
                diff_IP3[j,k] = IP3[j + 1, k] + IP3[j, k - 1] + IP3[j,k + 1] - 3 * IP3[j,k]
            elif j==Mastro-1: # bottom edge 
                diff_Ca[j,k] = Ca[j - 1, k] + Ca[j, k - 1] + Ca[j,k + 1] - 3 * Ca[j,k]
                diff_IP3[j,k] = IP3[j - 1, k] + IP3[j, k - 1] + IP3[j,k + 1] - 3 * IP3[j,k]
            elif k==0: # left edge
                diff_Ca[j,k] = Ca[j - 1, k] + Ca[j + 1, k] + Ca[j,k + 1] - 3 * Ca[j,k]
                diff_IP3[j,k] = IP3[j - 1, k] + IP3[j + 1, k] + IP3[j,k + 1] - 3 * IP3[j,k]
            elif k==Nastro-1: # right edge
                diff_Ca[j,k] = Ca[j - 1, k] + Ca[j + 1, k] + Ca[j,k - 1] - 3 * Ca[j,k]
                diff_IP3[j,k] = IP3[j - 1, k] + IP3[j + 1, k] + IP3[j,k - 1] - 3 * IP3[j,k]
            else: # middle
                diff_Ca[j,k] = (Ca[j - 1, k] + Ca[j + 1, k] + Ca[j, k - 1] + Ca[j,k + 1] 
                - 4 * Ca[j,k])
                diff_IP3[j,k] = (IP3[j - 1, k] + IP3[j + 1, k] + IP3[j, k - 1] 
                + IP3[j,k + 1] - 4 * IP3[j,k])
    
    return diff_Ca, diff_IP3

""" compute J_ER in each time step"""
@numba.vectorize(nopython=True)
def step_J_ER(Ca, h, IP3):
    factor = c1 * v1 * (Ca**3) * (h**3) * (IP3**3)
    num = c0/c1 - (1.0+1.0/c1) * Ca
    den = ((IP3 + d1) * (Ca + d5))**3
    return factor * num / den

""" compute J_pump in each time step"""
@numba.vectorize(nopython=True)
def step_J_pump(Ca):
    return v3 * Ca**2 / (k3**2 + Ca**2)

""" compute J_leak in each time step"""
@numba.vectorize(nopython=True)
def step_J_leak(Ca):
    return c1 * v2 * (c0/c1 - (1 + 1/c1) * Ca)

""" compute J_in in each time step"""
@numba.vectorize(nopython=True)
def step_J_in(IP3):
    return v6 * IP3**2 / (k2**2 + IP3**2)

""" compute J_out in each time step"""
@numba.vectorize(nopython=True)
def step_J_out(Ca):
    return k1 * Ca


""" update Ca, h, IP3 in each step"""
def step_Ca_h_IP3(step):
    diff_Ca_t, diff_IP3_t = step_diff_Ca_IP3(gv.Ca_all[step-1], gv.IP3_all[step-1])
    #d_IP3 = dIP3 * diff_IP3_t
    d_IP3 = (IP3_0 - gv.IP3_all[step-1]) / tau_IP3 + A_glu + dIP3 * diff_IP3_t
    gv.IP3_all[step] = gv.IP3_all[step-1] + d_IP3 * dt
    gv.IP3_all[step] = np.where(gv.IP3_all[step] < 0, 0, gv.IP3_all[step])
    
    d_Ca = (step_J_ER(gv.Ca_all[step-1], gv.h_all[step-1], gv.IP3_all[step-1]) 
    - step_J_pump(gv.Ca_all[step-1])
    + step_J_leak(gv.Ca_all[step-1]) + dCa * diff_Ca_t)
    gv.Ca_all[step] = gv.Ca_all[step-1] + d_Ca * dt
    gv.Ca_all[step] = np.where(gv.Ca_all[step] < 0, 0, gv.Ca_all[step])
    
    d_h = (a2 * (d2 * (gv.IP3_all[step-1] + d1) 
                / (gv.IP3_all[step-1] + d3) * (1 - gv.h_all[step-1]) 
                - gv.Ca_all[step-1] * gv.h_all[step-1]) )
    gv.h_all[step] = gv.h_all[step-1] + d_h * dt
    gv.h_all[step] = np.where(gv.h_all[step] < 0, 0, gv.h_all[step])

""" update IP3 in each step"""
def step_IP3(step):
    J_PLC_delta = v4 * (gv.Ca_all[step-1] + (1-alpha) * k4) / (gv.Ca_all[step-1] + k4)
    diff_Ca_t, diff_IP3_t = step_diff_Ca_IP3(gv.Ca_all[step-1], gv.IP3_all[step-1])
    d_IP3 =  ( (IP3_0 - gv.IP3_all[step-1]) / tau_IP3 
              + J_PLC_delta + dIP3 * diff_IP3_t + calcul_J_glu(gv.G_neuro_all[step-1]) )
    gv.IP3_all[step] = gv.IP3_all[step-1] + d_IP3 * dt
    gv.IP3_all[step] = np.where(gv.IP3_all[step] < 0, 0, gv.IP3_all[step])

""" update Ca in each step"""
def step_Ca(step):
    diff_Ca_t, diff_IP3_t = step_diff_Ca_IP3(gv.Ca_all[step-1], gv.IP3_all[step-1])
    d_Ca = (step_J_ER(gv.Ca_all[step-1], gv.h_all[step-1], gv.IP3_all[step-1]) 
    - step_J_pump(gv.Ca_all[step-1])
    + step_J_leak(gv.Ca_all[step-1]) 
    + step_J_in(gv.IP3_all[step-1])
    - step_J_out(gv.Ca_all[step-1])
    + dCa * diff_Ca_t)
    gv.Ca_all[step] = gv.Ca_all[step-1] + d_Ca * dt
    gv.Ca_all[step] = np.where(gv.Ca_all[step] < 0, 0, gv.Ca_all[step])
    
"""update h in each step"""
def step_h(step):
    d_h = (a2 * (d2 * (gv.IP3_all[step-1] + d1) 
                / (gv.IP3_all[step-1] + d3) * (1 - gv.h_all[step-1]) 
                - gv.Ca_all[step-1] * gv.h_all[step-1]) )
    gv.h_all[step] = gv.h_all[step-1] + d_h * dt
    gv.h_all[step] = np.where(gv.h_all[step] < 0, 0, gv.h_all[step])
    

""" initialize Ca, h, IP3 concentration"""
"""
def initialize_global_variables():
    Ca_all = np.zeros((num_steps, Mastro, Nastro))
    h_all = np.zeros((num_steps, Mastro, Nastro))
    IP3_all = np.zeros((num_steps, Mastro, Nastro))
    Ca_all[0] = np.random.uniform(0.1, 0.2, size=(Mastro, Nastro))
    h_all[0] = np.zeros((Mastro, Nastro)) + h_init
    IP3_all[0] = np.zeros((Mastro, Nastro)) + IP3_init
"""
    
    
""" simulate the dynamics of the astrocyte network for time T"""
"""
def simulate_astro_network(specify_IP3 = True):
    for step in range(1, num_steps):
        if specify_IP3:
            if int(num_steps/4) < step < int(num_steps/4*3):
                gv.IP3_all[step] = gv.IP3_all[0] 
            else:
                gv.IP3_all[step] = gv.IP3_all[0]
        else:
            step_IP3(step)
        step_Ca(step)
        step_h(step)
        #step_Ca_h_IP3(step)
"""

""" plot strength heatmap"""
def plot_strength_heatmap(array):
    fig = plt.figure(figsize=(12,12))
    plt.imshow(array, cmap='Blues')
    plt.colorbar()
    plt.show()

""" line plot """
def plot_strength_line(array, xl, yl, title):
    fig = plt.figure(figsize=(12,12))
    plt.plot(dt * np.arange(num_steps), array)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(title)
    plt.show()


"""
if __name__ == "__main__":

    initialize_global_variables()
    t1 = time.time()
    simulate_astro_network(False)
    t2 = time.time()
    print(t2 - t1)
    my_Ca = gv.Ca_all[:,2,2]
    plot_strength_line(my_Ca, xl="time(s)", yl="$Ca^{2+}$", title="Calcium dynamics")
"""