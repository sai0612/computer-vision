import random

import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main


def get_mini_batch(im_train, label_train, batch_size):
    # TO DO
    mini_batch_x = []
    mini_batch_y = []
    samples = im_train.shape[1]
    im_train = np.transpose(im_train)
    label_train = np.transpose(label_train)
    indices = list(range(samples))
    random.shuffle(indices)
    batch_x = []
    batch_y = []
    for ind in indices:
        batch_x.append(im_train[ind])
        one_hot_encoding = np.zeros(10)
        one_hot_encoding[int(label_train[ind])] = 1
        batch_y.append(one_hot_encoding)
        if len(batch_x) == batch_size:
            mini_batch_x.append(np.transpose(np.asarray(batch_x)))
            mini_batch_y.append(np.transpose(np.asarray(batch_y)))
            batch_x = []
            batch_y = []
    if len(batch_x) > 0:
        mini_batch_x.append(np.transpose(np.asarray(batch_x)))
        mini_batch_y.append(np.transpose(np.asarray(batch_y)))
    return mini_batch_x, mini_batch_y


def fc(x, w, b):
    # TO DO
    y = np.matmul(w, x)+b
    return y


def fc_backward(dl_dy, x, w, b, y):
    # TO DO
    dl_db = np.transpose(dl_dy)
    dl_dx = np.matmul(dl_dy, w)
    dl_dw = np.zeros(w.shape)
    for i in range(len(dl_dw)):
        dl_dw[i] = dl_dy[0][i] * x[:, 0]
    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    # TO DO
    l = 0
    for i in range(len(y)):
        t = y_tilde[i][0]-y[i][0]
        t = t*t
        l = l+t
    dl_dy = (y_tilde - y)*2
    return l, dl_dy


def loss_cross_entropy_softmax(x, y):
    # TO DO
    sigma_exp_x = 0
    for i in range(len(x)):
        sigma_exp_x = sigma_exp_x + np.exp(x[i][0])
    y_tilde = []
    for i in range(len(x)):
        y_tilde.append(np.exp(x[i][0])/sigma_exp_x)
    l = 0
    for i in range(len(x)):
        if y_tilde[i] > 0:
            l = l+y[i][0]*np.log(y_tilde[i])
    y_tilde = np.asarray(y_tilde).reshape(y.shape)
    dl_dx = (y_tilde - y)
    return l, dl_dx


def relu(x):
    # TO DO
    x_flat = x.reshape(-1)
    y = np.zeros(x_flat.shape)
    epsilon = 0.01
    for i in range(len(x_flat)):
        y[i] = max(x_flat[i], x_flat[i] * epsilon)
    y = y.reshape(x.shape)
    return y


def relu_backward(dl_dy, x, y):
    # TO DO
    dl_dy_flat = dl_dy.reshape(-1)
    dl_dx = np.zeros(dl_dy_flat.shape)
    epsilon = 0.01
    for i in range(len(dl_dy_flat)):
        dl_dx[i] = max(dl_dy_flat[i], dl_dy_flat[i]*epsilon)
    dl_dx = dl_dx.reshape(dl_dy.shape)
    return dl_dx


def conv(x, w_conv, b_conv):
    # TO DO
    H, W, c1 = x.shape
    h, w, c1, c2 = w_conv.shape
    y = np.zeros([H, W, c2])
    h_pad = int(h/2)
    w_pad = int(w/2)
    x_pad = np.pad(x, ((h_pad, h_pad), (w_pad, w_pad), (0, 0)))
    H_pad, W_pad, c1 = x_pad.shape
    for k in range(c2):
        for i in range(0, H_pad-h+1):
            for j in range(0, W_pad-w+1):
                sum = 0
                for fk in range(c1):
                    for fi in range(h):
                        for fj in range(w):
                            sum = sum + w_conv[fi][fj][fk][k]*x_pad[i+fi][j+fj][fk]
                y[i][j][k] = sum+b_conv[k]
    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    # TO DO
    H, W, c1 = x.shape
    h, w, c1, c2 = w_conv.shape
    dl_dw = np.zeros([h, w, c1, c2])
    dl_db = np.zeros(b_conv.shape)
    h_pad = int(h / 2)
    w_pad = int(w / 2)
    x_pad = np.pad(x, ((h_pad, h_pad), (w_pad, w_pad), (0, 0)))
    H_pad, W_pad, c1 = x_pad.shape
    for k in range(c2):
        sum_b = 0
        for fk in range(c1):
            for fi in range(h):
                for fj in range(w):
                    sum = 0
                    sum_b = 0
                    for i in range(fi, fi+H):
                        for j in range(fj, fj+W):
                            sum = sum + x_pad[i][j][fk]*dl_dy[i-fi][j-fj][k]
                            sum_b = sum_b + dl_dy[i-fi][j-fj][k]
                    dl_dw[fi][fj][fk][k] = sum
        dl_db[k] = sum_b
    return dl_dw, dl_db


def pool2x2(x):
    # TO DO
    H, W, c1 = x.shape
    h = int(H/2)
    w = int(W/2)
    i = 0
    j = 0
    y = np.zeros([h, w, c1])
    for pi in range(h):
        for pj in range(w):
            for c in range(c1):
                y[pi][pj][c] = np.max(x[i:i+2, j:j+2, c])
            j = j + 2
        j=0
        i = i + 2

    return y


def pool2x2_backward(dl_dy, x, y):
    # TO DO
    H, W, c1 = x.shape
    h, w, c1 = dl_dy.shape
    i = 0
    j = 0
    dl_dx = np.zeros([H, W, c1])
    for pi in range(h):
        for pj in range(w):
            for c in range(c1):
                max_v = dl_dy[pi][pj][c]
                if x[i][j][c] == max_v:
                    dl_dx[i][j][c] = max_v
                elif x[i + 1][j][c] == max_v:
                    dl_dx[i + 1][j][c] = max_v
                elif x[i][j + 1][c] == max_v:
                    dl_dx[i][j + 1][c] = max_v
                else:
                    dl_dx[i + 1][j + 1][c] = max_v
            j = j + 2
        j=0
        i = i + 2
    return dl_dx


def flattening(x):
    # TO DO
    y = x.reshape(-1, order='F')
    y = y.reshape((y.shape[0], 1))
    return y


def flattening_backward(dl_dy, x, y):
    # TO DO
    dl_dx = dl_dy.reshape(x.shape, order='F')
    return dl_dx


def train_slp_linear(mini_batch_x, mini_batch_y):
    # TO DO
    learning_rate = 0.001
    decay_rate = 0.9
    iteration = 2000
    w = np.random.normal(0, 1, size=(10, 196))
    b = np.random.normal(0, 1, size=(10, 1))
    k = 0
    for i in range(1, iteration+1):
        if i%1000 == 0:
            learning_rate = decay_rate*learning_rate
        dL_dw = np.zeros(w.shape)
        dL_db = np.zeros(b.shape)
        mini_batch_x_images = np.transpose(mini_batch_x[k])
        mini_batch_y_images = np.transpose(mini_batch_y[k])
        for j in range(len(mini_batch_x_images)):
            x = mini_batch_x_images[j]
            x = x.reshape(len(x), 1)
            y = mini_batch_y_images[j]
            y = y.reshape(len(y), 1)
            y_tilde = fc(x, w, b)
            l, dl_dy = loss_euclidean(y_tilde, y)
            dl_dy = np.transpose(dl_dy)
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y_tilde)
            dL_dw = dL_dw + dl_dw
            dL_db = dL_db + dl_db
        k = k+1
        if k == len(mini_batch_x):
            k = 0
        w = w - learning_rate * dL_dw
        b = b - learning_rate * dL_db

    return w, b


def train_slp(mini_batch_x, mini_batch_y):
    # TO DO
    # TO DO
    learning_rate = 0.01
    decay_rate = 0.9
    iteration = 2000
    w = np.random.normal(0, 1, size=(10, 196))
    b = np.random.normal(0, 1, size=(10, 1))
    k = 0
    for i in range(1, iteration + 1):
        if i % 1000 == 0:
            learning_rate = decay_rate * learning_rate
        dL_dw = np.zeros(w.shape)
        dL_db = np.zeros(b.shape)
        mini_batch_x_images = np.transpose(mini_batch_x[k])
        mini_batch_y_images = np.transpose(mini_batch_y[k])
        for j in range(len(mini_batch_x_images)):
            x = mini_batch_x_images[j]
            x = x.reshape(len(x), 1)
            y = mini_batch_y_images[j]
            y = y.reshape(len(y), 1)
            y_tilde = fc(x, w, b)
            l, dl_dy = loss_cross_entropy_softmax(y_tilde, y)
            dl_dy = np.transpose(dl_dy)
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y_tilde)
            dL_dw = dL_dw + dl_dw
            dL_db = dL_db + dl_db
        k = k + 1
        if k == len(mini_batch_x):
            k = 0
        w = w - learning_rate * dL_dw
        b = b - learning_rate * dL_db
    return w, b


def train_mlp(mini_batch_x, mini_batch_y):
    # TO DO
    learning_rate = 0.001
    decay_rate = 0.9
    iteration = 20000
    w1 = np.random.normal(0, 1, size=(30, 196))
    b1 = np.random.normal(0, 1, size=(30, 1))
    w2 = np.random.normal(0, 1, size=(10, 30))
    b2 = np.random.normal(0, 1, size=(10, 1))
    k = 0
    for i in range(1, iteration + 1):
        if i % 500 == 0:
            learning_rate = decay_rate * learning_rate
        dL_dw1 = np.zeros(w1.shape)
        dL_db1 = np.zeros(b1.shape)
        dL_dw2 = np.zeros(w2.shape)
        dL_db2 = np.zeros(b2.shape)
        mini_batch_x_images = np.transpose(mini_batch_x[k])
        mini_batch_y_images = np.transpose(mini_batch_y[k])
        for j in range(len(mini_batch_x_images)):
            x = mini_batch_x_images[j]
            x = x.reshape(len(x), 1)
            y = mini_batch_y_images[j]
            y = y.reshape(len(y), 1)
            y_tilde = fc(x, w1, b1)
            y_relu = relu(y_tilde)
            y_tilde_relu = fc(y_relu, w2, b2)
            l, dl_dy = loss_cross_entropy_softmax(y_tilde_relu, y)
            dl_dy = np.transpose(dl_dy)
            dl_dy, dl_dw2, dl_db2 = fc_backward(dl_dy, y_relu, w2, b2, y_tilde_relu)

            dl_dy = relu_backward(dl_dy, y_tilde, y_relu)
            dl_dy, dl_dw1, dl_db1 = fc_backward(dl_dy, x, w1, b1, y_tilde)
            dL_dw1 = dL_dw1 + dl_dw1
            dL_db1 = dL_db1 + dl_db1
            dL_dw2 = dL_dw2 + dl_dw2
            dL_db2 = dL_db2 + dl_db2
        k = k + 1
        if k == len(mini_batch_x):
            k = 0
        w1 = w1 - learning_rate * dL_dw1
        b1 = b1 - learning_rate * dL_db1
        w2 = w2 - learning_rate * dL_dw2
        b2 = b2 - learning_rate * dL_db2
    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    # TO DO
    learning_rate = 0.0022
    decay_rate = 0.6
    iteration = 10000
    w_conv = np.random.normal(0, 1, size=(3, 3, 1, 3))
    b_conv = np.random.normal(0, 1, size=3)
    w_fc = np.random.normal(0, 1, size=(10, 147))
    b_fc = np.random.normal(0, 1, size=(10, 1))
    k = 0
    for i in range(1, iteration + 1):
        if i % 500 == 0:
            learning_rate = decay_rate * learning_rate
        dL_dw_conv = np.zeros(w_conv.shape)
        dL_db_conv = np.zeros(b_conv.shape)
        dL_dw_fc = np.zeros(w_fc.shape)
        dL_db_fc = np.zeros(b_fc.shape)
        mini_batch_x_images = np.transpose(mini_batch_x[k])
        mini_batch_y_images = np.transpose(mini_batch_y[k])
        for j in range(len(mini_batch_x_images)):
            x = mini_batch_x_images[j]
            x = x.reshape((14, 14, 1), order='F')
            y = mini_batch_y_images[j]
            y = y.reshape(len(y), 1)
            y_conv = conv(x, w_conv, b_conv)
            y_relu = relu(y_conv)
            y_pool = pool2x2(y_relu)
            y_flatten = flattening(y_pool)

            y_fc = fc(y_flatten, w_fc, b_fc)
            l, dl_dy = loss_cross_entropy_softmax(y_fc, y)
            dl_dy = np.transpose(dl_dy)
            dl_dy, dl_dw_fc, dl_db_fc = fc_backward(dl_dy, y_flatten, w_fc, b_fc, y_fc)
            dl_dy = flattening_backward(dl_dy, y_pool, y_flatten)
            dl_dy = pool2x2_backward(dl_dy, y_relu, y_pool)
            dl_dy = relu_backward(dl_dy, y_conv, y_relu)
            dl_dw_conv, dl_db_conv = conv_backward(dl_dy, x, w_conv, b_conv, y_conv)

            dL_dw_conv = dL_dw_conv + dl_dw_conv
            dL_db_conv = dL_db_conv + dl_db_conv
            dL_dw_fc = dL_dw_fc + dl_dw_fc
            dL_db_fc = dL_db_fc + dl_db_fc
        k = k + 1
        if k == len(mini_batch_x):
            k = 0
        w_conv = w_conv - learning_rate * dL_dw_conv
        b_conv = b_conv - learning_rate * dL_db_conv
        w_fc = w_fc - learning_rate * dL_dw_fc
        b_fc = b_fc - learning_rate * dL_db_fc
    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    main.main_slp_linear()
    main.main_slp()
    main.main_mlp()
    main.main_cnn()



