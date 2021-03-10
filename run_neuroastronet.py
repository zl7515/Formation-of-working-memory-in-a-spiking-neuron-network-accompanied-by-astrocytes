# -*- coding: utf-8 -*-
"""
simulate neuro-astro network
"""

"""load source codes"""
from parameters import *
import gv
from load_images import *
from neuroastro_interaction import *
from astronet import *
from neuronet import *

"""load packages"""
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numba
import os
from skimage import color
import time
from celluloid import Camera

t1 = time.time()
"""initialize global variables"""
gv.init()

"""load images and process"""
load_images("../images")
convert_im_to_I()


"""set up adjacency matrix and tensors for neuro and astro networks"""

neuro_connection()  # set up neuronal connections
neuroastro_connection()  
astroneuro_connection()


"""
fig = plt.figure()
camera = Camera(fig)
for step in range(1, num_steps):
    step_V_U(step, gv.images[0].T)
    step_neuro_G(step)
    step_Ca(step)
    step_h(step)
    step_IP3(step)
    plt.imshow(gv.V_all[step].T)
    camera.snap()
animation = camera.animate()
animation.save('./anim.gif', writer='PillowWriter')
t2 = time.time()
print(t2 - t1)
"""



for step in range(1, num_steps):
    step_V_U(step, gv.images[0].T)
    step_neuro_G(step)
    step_Ca(step)
    step_h(step)
    step_IP3(step)
t2 = time.time()
print(t2 - t1)


V_1d = gv.V_all.reshape((num_steps, Mneuro*Nneuro))
spikes = np.argwhere(V_1d == 30)
steps, neurons = spikes.T
plt.figure(figsize=(12,12))
plt.scatter(steps*dt, neurons, s=0.3)
#plt.savefig("./fig_cut.png")

#plt.show()


"""
plt.plot(dt * np.arange(num_steps), gv.Ca_all[:,2,2])
plt.show()

im_vec = gv.images[0].reshape((Mneuro*Nneuro))
im_fired = np.argwhere(im_vec>10)
im_fired = list(im_fired)
im_fired = [i for [i] in im_fired]
im_fired_ext = im_fired * 200
t_im = np.arange(200)
t_im_ext = np.repeat(t_im, len(im_fired))
plt.figure(figsize=(12,12))
plt.scatter(t_im_ext, im_fired_ext, s=0.1)
"""