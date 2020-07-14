import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow_probability as tfp
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

# Plotting
import matplotlib.pyplot as plt
plt.style.use('default')
#%matplotlib inline
from mpl_toolkits import mplot3d

import mdn_network as mdnn
import toy_data as td
# ------------------------------------------------

batch_size = 1024
epochs = 700
print_every = 1
losses = []

# MAKE TRAINING DATA
(x, y, z, z_test) = td.create_gauss_pillars(n = 20000, n_c = 5)


plt.figure()
ax = plt.axes(projection = '3d')
ax.scatter3D(z, x, y, c = z[:, 0], alpha = 0.2, cmap = 'Greens');
#plt.show(block = False)
plt.savefig('training_data.png')
plt.close()


# TURN INTO TF DATASET CLASS
train_labels = np.concatenate([x,y], axis = 1)
N = train_labels.shape[0]

dataset = tf.data.Dataset \
          .from_tensor_slices((z, train_labels)) \
          .shuffle(N).batch(batch_size)

# GET NETWORK AND LOSS
mdn_network, mdn_optimizer = mdnn.get_mdn_iso()
loss_fun = mdnn.mdn_loss_iso

# TRAINING LOOP
print('Print every {} epochs'.format(print_every))
for i in range(epochs):
    for train_features, train_labels in dataset:
        loss = mdnn.train_step(mdn_network, mdn_optimizer, loss_fun, train_features, train_labels)
        losses.append(loss)
    if i % print_every == 0:
        print('Epoch {}/{}: loss {}'.format(i, epochs, losses[- 1]))  

# LOSS PLOT
plt.figure()
plt.plot(range(len(losses)), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training')
plt.savefig('loss_history.png')
plt.close()

# GET MODEL PREDICTIONS ( MIXTURE PARAMETERS) ON SOME TEST INPUTS
pi_vals, mu_vals, var_vals = mdn_network.predict(z_test)
#print(pi_vals.shape, mu_vals.shape, var_vals.shape)

# SAMPLE FROM MIXTURE PARAMETERS
print('Sampling from model')
sampled_predictions = mdnn.sample_predictions_mdn(pi_vals, mu_vals, var_vals, samples = 10)

plt.figure()
ax = plt.axes(projection = '3d')
#x_det, y_det, z_det, z_test_det = create_spiral(n = 10000, noise_range = 0.01, freq_base = 2)

# PLOT SAMPLES FROM MODEL
print('Plotting model samples')
for i in range(10):
    ax.scatter3D(z_test, sampled_predictions[:, i, 0], sampled_predictions[:, i, 1], c = z_test[:, 0], alpha = 0.05, cmap = 'Greens') #cmap = 'Greens', alpha = 0.2);
    #ax.scatter3D(z, xy[:, 0], xy[:, 1], c = 'black', alpha = 1)
    #ax.scatter3D(z_det, x_det, y_det, c = 'black') #c = z[:, 0], alpha = 1) # , cmap = 'Greens') #, alpha = 0.2);
plt.savefig('model_samples.png')
plt.close()

# NOT USED FOR NOW

# Train step Full cov
# @tf.function
# def train_step_jensen(model, optimizer, train_x, train_y):
#     # GradientTape: Trace operations to compute gradients
#     with tf.GradientTape() as tape:
#         pi_, mu_, var_ = model(train_x, training = True)
#         # calculate loss
#         loss = mdn_loss_full_cov_jensen(train_y, pi_, mu_, var_)
#     # compute and apply gradients
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     return loss