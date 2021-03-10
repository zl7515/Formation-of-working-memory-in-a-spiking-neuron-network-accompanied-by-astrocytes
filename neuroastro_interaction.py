# -*- coding: utf-8 -*-
"""
build neuron-astrocyte interactions

"""

from parameters import *
import numpy as np
import numba
import gv

""" 
adjacency tensor to pass info from neuro to astro
shape: (Mastro, Nastro, Mneuro, Nneuro)
"""
def neuroastro_connection():
    m_start = 0
    n_start = 0
    for i in range(Mastro):
        for j in range(Nastro):
            gv.adj_ts_na[i,j][m_start:m_start+Na, n_start:n_start+Na] = np.full((Na, Na), 1)
            n_start = n_start + Na - Na_overlap
        n_start = 0
        m_start = m_start + Na - Na_overlap


""" 
adjacency tensor to pass info from astro to neuro
shape: (Mneuro, Nneuro, Mastro, Nastro)
"""
def astroneuro_connection():
    gv.adj_ts_an = np.transpose(gv.adj_ts_na, (2,3,0,1))

""" calculate astrocyte glutamate according to neuron glutamate"""  
    
def calcul_astro_G(neuro_G):
    astro_G = np.zeros((Mastro, Nastro))
    for j in range(Mastro):
        for k in range(Nastro):
            G_mat = gv.adj_ts_na[j, k] * neuro_G
            astro_G[j, k] = G_mat.sum()
    return astro_G

"""
def calcul_astro_G(neuro_G):
    G_list = [(gv.adj_ts_na[j, k] * neuro_G).sum() for j in range(Mastro) for k in range(Nastro)]
    return np.array(G_list).reshape((Mastro, Nastro))
"""

"""
@numba.jit(nopython=True)
def calcul_astro_G(neuro_G, adj):
    astro_G = np.zeros((Mastro, Nastro))
    for j in range(Mastro):
        for k in range(Nastro):
            G_mat = adj[j, k] * neuro_G
            astro_G[j, k] = G_mat.sum()
    return astro_G
"""

""" calculate neuron calcium according to astrocyte calcium"""
"""
def calcul_neuro_Ca(astro_Ca):
    neuro_Ca = np.zeros((Mneuro, Nneuro))
    for j in range(Mneuro):
        for k in range(Nneuro):
            Ca_mat = gv.adj_ts_an[j, k] * astro_Ca
            neuro_Ca[j, k] = Ca_mat.sum()
    return neuro_Ca
"""

@numba.jit(nopython=True)
def calcul_neuro_Ca(astro_Ca, adj):
    neuro_Ca = np.zeros((Mneuro, Nneuro))
    for j in range(Mneuro):
        for k in range(Nneuro):
            Ca_mat = adj[j, k] * astro_Ca
            neuro_Ca[j, k] = Ca_mat.sum()
    return neuro_Ca

"""
neuroastro_connection()
astroneuro_connection()
a0=calcul_astro_G(np.eye(Mneuro))
b0=calcul_neuro_Ca(np.eye(Mastro))
"""