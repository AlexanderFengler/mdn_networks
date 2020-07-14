#IMPORTS 
import numpy as np
import os
import pandas as pd
import time
import psutil
import argparse
from datetime import datetime
import pickle
import yaml
from copy import deepcopy


# Toy problem 1
def noisy_rising_oscillation(n = 1000,
                             freq_base = 4,
                             oscillation_scale = 0.3,
                             noise_sd = 0.1,
                             x_range = (0, 1), 
                             invert = False):

    x = np.random.uniform(x_range[0], x_range[1], size = (n, 1)).astype(np.float32)
    y = x + (oscillation_scale * np.sin(freq_base * np.pi * x)) + (noise_sd * np.random.normal(size = (n, 1)).astype(np.float32))
    
    # test data
    x_test = np.linspace(0, 1, n).reshape(-1, 1).astype(np.float32)
    
    if invert:
        return (y, x, x_test)
    else:
        return (x, y, x_test)

# Toy problem 2
def create_spiral(n = 1000, 
                  a_base = 2, 
                  noise_sd = 0.1, 
                  z_range = (0, 1), 
                  n_spirals = 4,
                  freq_base = 1):

    x = []
    y = []
    z = []
    z_test = []

    for i in range(n_spirals):
        freq_tmp = freq_base * np.random.normal(loc = 0, scale = 1)
        a_tmp = a_base * np.random.normal(loc = 3, scale = 0.5)
        z_tmp = np.random.uniform(z_range[0], z_range[1], size = (n, 1)).astype(np.float32)
        x_tmp = a_tmp * np.cos(freq_tmp * np.pi * z_tmp) + (noise_sd) * np.random.normal(size = (n, 1)).astype(np.float32)
        y_tmp = a_tmp * np.sin(freq_tmp * np.pi * z_tmp) + (noise_sd) * np.random.normal(size = (n, 1)).astype(np.float32)

        z.append(z_tmp)
        x.append(x_tmp)
        y.append(y_tmp)

    x = np.concatenate(x)
    y = np.concatenate(y)
    z = np.concatenate(z)

    z_test = np.linspace(z_range[0], z_range[1], z.shape[0]).reshape(-1, 1).astype(np.float32)
    return (x, y, z, z_test) 



def create_gauss_pillars(n = 10000, 
                         n_c = 10, 
                         noise_sd_base = 0.4,
                         xy_range = (-10, 10), 
                         z_range = (0 , 1), 
                         slope_range = (-5, 5)):
    
    unif_x = np.random.uniform(xy_range[0], xy_range[1], size = n_c).astype(np.float32)
    unif_y = np.random.uniform(xy_range[0], xy_range[1], size = n_c).astype(np.float32)
    centers = np.stack([unif_x, unif_y]).T
    
    z = np.random.uniform(z_range[0], z_range[1], size = (n, 1)).astype(np.float32)
    xy = np.zeros((n, 2)).astype(np.float32)

    slopes = np.random.uniform(slope_range[0],slope_range[1], size = (n_c, 2)).astype(np.float32)

    for i in range(n):
        c_tmp = np.random.choice(n_c)
        xy[i,:] = np.random.normal(loc = centers[c_tmp, :] + (z[i] * slopes[c_tmp, :]), 
                                   scale = noise_sd_base * np.abs(z[i] - 0.5) + 0.25)
        #print(np.random.normal(loc = centers[np.random.choice(5), :], scale = z[i] + 0.1))

    x = np.expand_dims(xy[:, 0], -1)
    y = np.expand_dims(xy[:, 1], -1)
    z_test = np.linspace(z_range[0], z_range[1], n).reshape(-1, 1).astype(np.float32)
    return (x, y, z, z_test)


