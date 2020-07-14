# IMPORTS
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

# 1D Output model

# Network
def get_mdn_1d(n_input_features = 1,
               n_mixture_components = 5, 
               layer_sizes_fc = [100]):  # this is the one we need for ddm anyways..
    
    input = tf.keras.Input(shape = (n_input_features, ))
    cnt = 0
    
    for layer_size in layer_sizes_fc:
        if cnt == 0:
            layer = tf.keras.layers.Dense(layer_size, activation = 'tanh', name = 'baselayer_0')(input)
        else:
            layer = tf.keras.layers.Dense(layer_size, activation = 'tanh', name = 'baselayer_' + str(cnt))(layer)
        cnt + 1

    # mu nodes
    mu = tf.keras.layers.Dense(n_mixture_components, activation = None, name = 'mean_layer', bias_initializer = tf.keras.initializers.RandomUniform(minval = - 1, maxval = 1, seed = None))(layer)
    
    # variance nodes (should be greater than 0 so we exponentiate it)
    var = tf.keras.layers.Dense(n_mixture_components, activation = 'softplus', name = 'dense_var_layer', bias_initializer = tf.keras.initializers.RandomUniform(minval = - 10, maxval = 10, seed = None))(layer)
    
    # mixing coefficient should sum to 1.0
    pi = tf.keras.layers.Dense(k, activation = 'softmax', name = 'pi_layer', bias_initializer = tf.keras.initializers.RandomUniform(minval = 0, maxval = 1, seed = None))(layer)
    
    # Make model
    model = tf.keras.models.Model(input, [pi, mu, var])
    # Make optimizer
    optimizer = tf.keras.optimizers.Adam()
    
    return (model, optimizer)


# In our toy example, we have single input feature
l = 1

# Number of dimensions in output
dim_out = 2
utri_dim = int((dim_out * (dim_out + 1) / 2) - dim_out)
covar_dim = int((dim_out * (dim_out + 1) / 2))

# Number of gaussians to represent the multimodal distribution
k = 5

def get_mdn_iso(n_input_features = 1, 
                dim_out = 2,
                n_mixture_components = 10, 
                layer_sizes_fc = [100]):
    
    input = tf.keras.Input(shape = (n_input_features, ))
    
    cnt = 0
    for layer_size in layer_sizes_fc:
        if cnt == 0:
            layer = tf.keras.layers.Dense(50, activation = 'tanh', name = 'baselayer_0')(input)
        else:    
            layer = tf.keras.layers.Dense(100, activation = 'tanh', name = 'baselayer_' + str(cnt))(layer)
        cnt += 1
    
    # mu nodes
    mu = tf.keras.layers.Dense((dim_out * n_mixture_components), activation = None, name = 'mean_layer', bias_initializer = tf.keras.initializers.RandomUniform(minval = -1, maxval = 1, seed = None))(layer)
    mu = tf.keras.layers.Reshape((n_mixture_components, dim_out), name = 'mean_layer_reshaped')(mu)

    # variance nodes
    var = tf.keras.layers.Dense((dim_out * n_mixture_components), activation = 'softplus', name = 'dense_var_layer', bias_initializer = tf.keras.initializers.RandomUniform(minval = 5, maxval = 10, seed = None))(layer)
    var = tf.keras.layers.Reshape((n_mixture_components, dim_out), name = 'var_layer_reshaped')(var)

    # mixture probability nodes
    pi = tf.keras.layers.Dense(n_mixture_components, activation = 'softmax', name = 'pi_layer', bias_initializer = tf.keras.initializers.RandomUniform(minval = -1, maxval = 1, seed = None))(layer)
    
    # Make model
    model = tf.keras.models.Model(input, [pi, mu, var])

    # Make optimizer
    optimizer = tf.keras.optimizers.Adam()

    return (model, optimizer)



def get_mdn_full_cov(n_input_features = 1,
                     dim_out = 2,
                     n_mixture_components = 10,
                     layer_sizes_fc = [100],
                     batch_norm =  False):

    covar_dim = int((dim_out * (dim_out + 1) / 2))

    input = tf.keras.Input(shape = (n_input_features, ))

    cnt = 0
    for layer_size in layer_sizes_fc:
        if cnt == 0:
            layer = tf.keras.layers.Dense(layer_size, activation = None, name = 'baselayer_' + str(cnt))(input)
        else:
            layer = tf.keras.layers.Dense(layer_size, activation = None, name = 'baselayer_' + str(cnt))(layer)
        
        # Batch norm layer
        if batch_norm:
            batch_norm_layer = tf.keras.layers.BatchNormalization(name = 'baselayer_batchnorm_' + str(cnt))(layer)
        
        # Activation
        layer_act = tf.keras.layers.ReLU(name = 'baselayer_relu_act_' + str(cnt))(layer)

        cnt += 1

    # Mu nodes
    mu = tf.keras.layers.Dense((dim_out * n_mixture_components), activation = None, name = 'mean_layer', bias_initializer = tf.keras.initializers.RandomUniform(minval = -1, maxval = 1, seed = None))(layer_act2)
    mu = tf.keras.layers.Reshape((n_mixture_components, dim_out), name = 'mean_layer_reshaped')(mu)
    
    # Mixture probability nodes
    pi = tf.keras.layers.Dense(k, activation = 'softmax', name = 'pi_layer', bias_initializer = tf.keras.initializers.RandomUniform(minval = 5, maxval = 10, seed = None))(layer_act2)

    # Covariance nodes
    covar = tf.keras.layers.Dense((covar_dim * n_mixture_components), name = 'covar_layer', activation = None, bias_initializer = tf.keras.initializers.RandomUniform(minval = -1, maxval = 1, seed = None))(layer_act2)
    covar = tf.keras.layers.Reshape((n_mixture_components, covar_dim), name = 'covar_layer_reshaped')(covar)
    
    # Make model
    model = tf.keras.models.Model(input, [pi, mu, covar])

    # Make optimizer
    optimizer = tf.keras.optimizers.Adam()
    return (model, optimizer)



# Losses

# 1D
def calc_pdf_1d(y, mu, var):
    """Calculate Component Density"""
    value = tf.square(tf.subtract(y, mu)) 
    value = (1 / tf.sqrt(2 * np.pi * var)) * tf.exp( (- value) / (2 * var))
    return value

def mdn_loss_1d(y_true, pi, mu, var):
    """MDN Loss Function"""
    out = calc_pdf(y_true, mu, var)
    out = tf.multiply(out, pi)
    out = tf.reduce_sum(out, 1, keepdims = True)
    out = - tf.math.log(out + 1e-10)
    return tf.reduce_mean(out)

# ISO
def calc_pdf_iso(y, mu, var):
    """Calculate Component Density"""
    var = var
    inv_var = tf.linalg.diag(1 / var)
    #print(inv_var.shape)
    inv_var_det = tf.expand_dims(tf.expand_dims(tf.sqrt(tf.linalg.det(inv_var)), -1), -1)
    #print(inv_var_det.shape)
    y_proper = tf.expand_dims(y, 1)
    diff = tf.expand_dims(tf.subtract(y_proper, mu), - 1)
    #print(diff.shape)
    diff_transp = tf.transpose(diff, perm = [0, 1, 3, 2])
    print('diff_transp:', diff_transp.shape)
    #print(diff_transp.shape)
    exponent = (- 1 / 2) * tf.matmul(tf.matmul(diff_transp, inv_var), diff)
    #print(exponent.shape)
    denominator = 1 / tf.sqrt(tf.pow(np.pi, dim_out))
    #determinantf = tf.sqrt(tf.linalg.det(inv_var))
    #value = tf.multiply(tf.exp(exponent), inv_var_det)
    
    print(denominator.shape)
    print(inv_var_det.shape)
    print(exponent.shape)
    value = denominator * tf.multiply(tf.exp(exponent), inv_var_det)
    #value = tf.exp(tf.divide(tf.square(tf.subtract(y, mu)), 2 * var)) # Gaussian exp
    #value = (1 / 2 * np.pi * var)) * tf.exp( (- value) / (2 * var))
    return tf.squeeze(value)

def mdn_loss_iso(y_true, pi, mu, var, omega = 0.0001):
    """MDN Loss Function"""
    out = calc_pdf_iso(y_true, mu, var)
    print('out shape: ',out.shape)
    print('pi shape:', pi.shape)
    out = tf.multiply(out, pi)
    print('out after multiply with pi:', out.shape)
    # print(out)
    out = tf.reduce_sum(out, 1, keepdims = True)
    inv_var_red = tf.reduce_sum(1 / var, (1, 2), keepdims = True)
    out = - tf.math.log(out + 1e-10) #+ (omega * inv_var_red)
    return tf.reduce_mean(out)

# FULL
def calc_pdf_full_cov(y, mu, covar):
    """Calculate Component Density"""
    # var = var
    U = tfp.math.fill_triangular(covar, upper = True)
    #tf.print('U_before_transform', U[0, 0, :, :])
    U_diag = tf.linalg.diag(tf.linalg.diag_part(U))
    #tf.print('U_diag', U_diag[0, 0, :, :])
    #print('U_diag: ', U_diag.shape)
    U = U - U_diag + tf.linalg.diag(tf.math.softplus(tf.linalg.diag_part(U_diag))) # + (tf.eye(U_diag.shape) * 1e-20)
    #tf.print('U', U[0,0,:,:])
    S_inv = tf.matmul(tf.transpose(U, perm = [0, 1, 3, 2]), U)
    #print('S_inv')
    log_sqrt_S_inv_det = tf.minimum(tf.expand_dims((tf.reduce_sum(tf.linalg.diag_part(U_diag), (2), keepdims = True)), -1), 66.76)
    #tf.print(tf.reduce_min(log_sqrt_S_inv_det))
    #tf.print(tf.reduce_max(log_sqrt_S_inv_det))
    #print('log_sqrt_S_inv_det:', log_sqrt_S_inv_det.shape)
    y_proper = tf.expand_dims(y, 1)
    #print('y_proper: ', y_proper.shape)
    #print('mu: ', mu.shape)
    diff = tf.expand_dims(tf.subtract(y_proper, mu), - 1)
    diff_transp = tf.transpose(diff, perm = [0, 1, 3, 2])
    exponent = tf.maximum((- (1 / 2) * tf.matmul(tf.matmul(diff_transp, S_inv), diff)), -66.76)
    #tf.print(exponent)
    #tf.print(tf.reduce_min(exponent))
    #tf.print(tf.reduce_max(exponent))
    value = tf.exp((log_sqrt_S_inv_det + exponent))
    #tf.print(value.shape)
    print('exponent: ', exponent.shape)
    #value = tf.multiply(tf.exp(exponent), S_inv_det_div_2)
    return tf.squeeze(value)

def mdn_loss_full_cov(y_true, pi, mu, var, omega = 0.0001):
    """MDN Loss Function"""
    out = calc_pdf_full_cov(y_true, mu, var)
    #tf.print(tf.reduce_min(pi))
    #tf.print(tf.reduce_max(pi))
    #print('out shape: ', out.shape)
    #print('pi shape:', pi.shape)
    out = tf.multiply(out, pi)
    #print('out after multiply with pi:', out.shape)
    #tf.print(out.shape)
    # print(out)
    out = tf.reduce_sum(out, 1, keepdims = True)
    #print('out after reduce sum: ', out.shape)
    #inv_var_red = tf.reduce_sum(1 / var, (1, 2), keepdims = True)
    out = - tf.math.log(out + 1e-10) #+ (omega * inv_var_red)
    return tf.reduce_mean(out)

# Jensen Loss
def calc_pdf_full_cov_jensen(y, mu, covar):
    """Calculate Component Density"""
    # var = var
    U = tfp.math.fill_triangular(covar, upper = True)
    U_diag = tf.linalg.diag(tf.linalg.diag_part(U))
    #print('U_diag: ', U_diag.shape)
    #U_diag_pos = tf.math.softplus(U_diag)
    U = U - U_diag + tf.linalg.diag(tf.math.softplus(tf.linalg.diag_part(U_diag))) # + (tf.eye(U_diag.shape) * 1e-20)
    
    S_inv = tf.matmul(tf.transpose(U, perm = [0, 1, 3, 2]), U)
    #print('S_inv')
    #tf.print(S_inv)
    #tf.print(S_inv)
    log_sqrt_S_inv_det = tf.expand_dims((tf.reduce_sum(tf.linalg.diag_part(U_diag), (2),keepdims = True)), -1)
    #print('log_sqrt_S_inv_det:', log_sqrt_S_inv_det.shape)
    #tf.print(S_inv_det_div_2)
    y_proper = tf.expand_dims(y, 1)
    #print('y_proper: ', y_proper.shape)
    #print('mu: ', mu.shape)
    diff = tf.expand_dims(tf.subtract(y_proper, mu), - 1)
    diff_transp = tf.transpose(diff, perm = [0, 1, 3, 2])
    exponent = - (1 / 2) * tf.matmul(tf.matmul(diff_transp, S_inv), diff)
    tf.print(tf.reduce_min(exponent))
    tf.print(tf.reduce_max(exponent))
    #tf.print(exponent)
    value = (log_sqrt_S_inv_det + exponent)
    
    print('exponent: ', exponent.shape)
    #value = tf.multiply(tf.exp(exponent), S_inv_det_div_2)
    return tf.squeeze(value)

def mdn_loss_full_cov_jensen(y_true, pi, mu, var, omega = 0.0001):
    """MDN Loss Function"""
    out = calc_pdf_full_cov_jensen(y_true, mu, var)
    #print('out shape: ', out.shape)
    #print('pi shape:', pi.shape)
    out = out + tf.math.log(pi)
    #print('out after multiply with pi:', out.shape)
    # print(out)
    out = tf.reduce_sum(out, 1, keepdims = True)
    #print('out after reduce sum: ', out.shape)
    #inv_var_red = tf.reduce_sum(1 / var, (1, 2), keepdims = True)
    #out = - tf.math.log(out + 1e-10) #+ (omega * inv_var_red)
    return  - tf.reduce_mean(out)


def sample_predictions_mdn_full(pi_vals, mu_vals, var_vals, input_dim = 1, samples = 1):
    n, k = pi_vals.shape
    #print(var_vals.shape)
    U = tfp.math.fill_triangular(var_vals, upper = True) 
    U_diag = tf.linalg.diag(tf.linalg.diag_part(U))
    U = U - U_diag + tf.math.softplus(U_diag)
    #print(U.shape)
    cov_inv = tf.matmul(tf.transpose(U, perm = [0, 1, 3, 2]), U)
    cov = tf.linalg.inv(cov_inv).numpy()
    #print(cov.shape)
    #var_vals = var_vals.numpy()
    
    # print('shape: ', n, k, l)
    # place holder to store the y value for each sample of each row
    out = np.zeros((n, samples, dim_out))
    for i in range(n):
        for j in range(samples):
            # for each sample, use pi/probs to sample the index
            # that will be used to pick up the mu and var values
            idx = np.random.choice(range(k), p = pi_vals[i])
            #for li in range(input_dim):
                # Draw random sample from gaussian distribution
                #print(cov[i, idx, :, :])
                #print(li)
                #print(mu_vals[i, idx, :])
                #print(i)
                #print(j)
                #print(cov.shape)
            out[i, j, :] = np.random.multivariate_normal(mean = mu_vals[i, idx, :], cov = (cov[i, idx, :, :]))
    return out  


def sample_predictions_mdn(pi_vals, mu_vals, var_vals, samples = 1):
    n, k = pi_vals.shape
    # print('shape: ', n, k, l)
    # place holder to store the y value for each sample of each row
    out = np.zeros((n, samples, dim_out))
    for i in range(n):
        for j in range(samples):
            # for each sample, use pi/probs to sample the index
            # that will be used to pick up the mu and var values
            idx = np.random.choice(range(k), p = pi_vals[i])
            #for li in range(input_dim):
                # Draw random sample from gaussian distribution
            out[i, j, :] = np.random.normal(mu_vals[i, idx, :], np.sqrt(var_vals[i, idx, :]))
    return out  


# TRAINING 

# Train step:
@tf.function
def train_step(model, optimizer, loss_fun, x, y):
    # GradientTape: Trace operations to compute gradients
    with tf.GradientTape() as tape:
        # get model predictions
        pi_, mu_, var_ = model(x, training = True)
        
        # calculate loss
        loss = loss_fun(y, pi_, mu_, var_)
    
    # compute and apply gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    #tf.print(tf.reduce_min(gradients))
    #tf.print(tf.reduce_max(gradients))
    #capped_gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
    #capped_gradients = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gradients]
    #optimizer.apply_gradients(zip(capped_gradients, model.trainable_variables))
    # Apply gradients through optimizer
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


