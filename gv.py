# -*- coding: utf-8 -*-
"""
global variables settings
"""
from parameters import *
import numpy as np

def init():
    global images, adj_mat, adj_ts_na, adj_ts_an, V_all, U_all, G_neuro_all
    global Ca_all, h_all, IP3_all
    images = []  # image list
    
    adj_mat = np.zeros((Mneuro*Nneuro, Mneuro*Nneuro))  # adjacency matrix of neurons
    adj_ts_na = np.zeros((Mastro, Nastro, Mneuro, Nneuro))  # adjacency tensor astro-neuro
    adj_ts_an = np.zeros((Mneuro, Nneuro, Mastro, Nastro))  # adjacency tensor neuro-astro
    
    V_all = np.zeros((num_steps, Mneuro, Nneuro))  # 3d array to store V 
    U_all = np.zeros((num_steps, Mneuro, Nneuro))  # 3d array to store U
    G_neuro_all = np.zeros((num_steps, Mneuro, Nneuro))  # 3d array to store glutamate
    V_all[0] = np.full((Mneuro, Nneuro), c)
    U_all[0] = np.full((Mneuro, Nneuro), b * c)
    
    Ca_all = np.zeros((num_steps, Mastro, Nastro))  # 3d array to store Ca
    h_all = np.zeros((num_steps, Mastro, Nastro))  # 3d array to store h
    IP3_all = np.zeros((num_steps, Mastro, Nastro))  # 3d array to store IP3
    Ca_all[0] = np.full((Mastro, Nastro), 0.07)
    h_all[0] = np.full((Mastro, Nastro), h_init) 
    IP3_all[0] = np.full((Mastro, Nastro), 0.85)