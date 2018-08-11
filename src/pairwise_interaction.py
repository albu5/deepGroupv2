# pairwise interactions and group formation matrix

import numpy as np
from utils import read_anno, pairwise_features_interaction_batch, pairwise_features_at_time_t, \
    pairwise_interactions_at_time_t, draw_pairwise_features_at_time_t, draw_groups_at_time_t
from networks import pairwise_interaction_mat, my_categorical_crossentropy_mask, my_categorical_accuracy_mask
from matplotlib import pyplot as plt
from skimage.io import imread
from scipy.io import loadmat
from keras.optimizers import adam
from random import shuffle
import pickle
from os.path import isfile
from sklearn.cluster import DBSCAN
from keras.models import load_model, model_from_json
from keras.losses import categorical_crossentropy
import keras.losses
import keras.metrics
import os
from keras import backend as kb
from keras.metrics import categorical_accuracy
import warnings

feature_len = 38
n_people = 10
n_epoch = 100
pose_info = ['right', 'front-right', 'front', 'front-left', 'left', 'left-back', 'back', 'right-back']
action_info = ['standing', 'walking', 'running']
interaction_info = ['no-interaction', 'int1', 'int2', 'int3', 'int4', 'int5', 'int6', 'int7', 'int8', 'int9', 'int10']
n_interaction_info = len(interaction_info)
group_act_info = ['act1', 'act2', 'act3', 'act4', 'act5', 'act6', 'act7', 'act8', 'act9', 'act10']
scene_act_info = ['act1', 'act2', 'act3', 'act4', 'act5', 'act6', 'act7', 'act8', 'act9', 'act10']

keras.losses.my_categorical_crossentropy = my_categorical_crossentropy_mask
keras.metrics.my_categorical_accuracy = my_categorical_accuracy_mask
warnings.filterwarnings("ignore", ".*Using default event loop until function specific to this GUI is implemented.*")

anno_fmt = "../data/final_csv_anno/anno%2.2d.mat"
img_fmt = "../data/images/seq%2.2d/frame%6.6d.jpg"
title_fmt = "Seq: %2.2d | Frame: %6.6d"
group_fmt = "../data/group_int/group_int%2.2d.mat"
mask_fmt = "../data/mask/mask%2.2d.mat"
model_weights = "../models/pairwise_interaction/pi.h5"
data_file = "./tmp/pi_data.bin"
ax = plt.gca()

train_data = []
test_data = []
data = []

test_seq = [1, 2, 7, 12, 13, 19, 20, 21, 26, 27, 30]

# create if to be written directories do not exist
if not os.path.isdir(os.path.dirname(data_file)):
    print('creating ' + os.path.dirname(data_file) + 'directory...')
    os.makedirs(os.path.dirname(data_file))

if not os.path.isdir(os.path.dirname(model_weights)):
    print('creating ' + os.path.dirname(model_weights) + 'directory...')
    os.makedirs(os.path.dirname(model_weights))

if not isfile(data_file):
    print('preparing data...')
    for _seq in range(0, 33):
        seq = _seq + 1
        anno_data = read_anno(anno_path=anno_fmt % seq)
        group_data = loadmat(group_fmt % seq)['group_int']
        mask_data = loadmat(mask_fmt % seq)['mask_mat']

        print('seq: ', seq)
        for t in range(0, anno_data.shape[0], 10):
            pf = pairwise_features_at_time_t(anno_data, t)
            pi = pairwise_interactions_at_time_t(anno_data, t)
            px, py = pairwise_features_interaction_batch(pf, pi)
            n_people = anno_data.shape[2]

            px = np.reshape(px, newshape=(n_people, n_people, feature_len))
            gd = group_data[:, :, t]
            mask = mask_data[t, :, :]

            # get GA and add it to data, use GA as pairwise interaction
            ga = anno_data[t, -2, :]
            interaction = np.repeat(np.reshape(ga, newshape=(ga.shape[0], 1)), ga.shape[0], axis=1)
            interaction *= gd

            # convert interaction to ont hot
            interaction = np.expand_dims(interaction, axis=2)
            interaction = np.repeat(interaction, repeats=len(group_act_info), axis=2)
            _interaction = np.reshape(np.arange(len(group_act_info)), (1, 1, len(group_act_info)))
            _interaction = np.repeat(_interaction, repeats=interaction.shape[0], axis=0)
            _interaction = np.repeat(_interaction, repeats=interaction.shape[0], axis=1)
            interaction = interaction == _interaction

            # reshape to n_people*n_people and remove masked entries
            px = np.reshape(px, newshape=(n_people*n_people, feature_len))
            interaction = np.reshape(interaction, newshape=(n_people * n_people, len(group_act_info)))
            mask = np.reshape(mask, newshape=(n_people * n_people,)) > 0
            px = px[mask, :]
            interaction = interaction[mask, :]
            n_valid_people = np.round(np.sqrt(px.shape[0])).astype(np.int)
            px = np.reshape(px, newshape=(n_valid_people, n_valid_people, feature_len))
            interaction = np.reshape(interaction, newshape=(n_valid_people, n_valid_people, len(group_act_info)))

            if seq in test_seq:
                test_data.append((px, interaction))
            else:
                train_data.append((px, interaction))
    with open(data_file, mode='wb') as f:
        print('caching prepared data to hard disk...')
        pickle.dump({'train_data': train_data, 'test_data': test_data}, f)
else:
    print('loading from ' + data_file + ' ...')
    f = open(data_file, mode='rb')
    data = pickle.load(f)
    train_data = data['train_data']
    test_data = data['test_data']
    f.close()

if os.path.isfile(model_weights):
    print('loading ' + model_weights + ' ...')
    pi_net = pairwise_interaction_mat()
    pi_net.load_weights(model_weights)
else:
    print(model_weights + ' not found. Creating and training new model ...')
    pi_net = pairwise_interaction_mat()
    pi_net.compile(optimizer=adam(1e-4), loss=my_categorical_crossentropy_mask,
                   metrics=[my_categorical_accuracy_mask])

    mae_epoch_train = []
    mae_epoch_test = []
    for n_epoch in range(n_epoch):
        shuffle(train_data)
        mae_train = []
        for train_datum in train_data:
            if train_datum[0].size == 0:
                continue
            pi_pred = pi_net.predict(np.expand_dims(train_datum[0], axis=0),
                                     verbose=0)
            pi_pred = pi_pred[0, :, :, :]
            mae = np.mean((np.argmax(pi_pred, axis=2) == np.argmax(train_datum[1], axis=2)).astype(np.float))
            mae_train.append(mae)
            pi_net.train_on_batch(np.expand_dims(train_datum[0], axis=0), np.expand_dims(train_datum[1], axis=0))
        mae_epoch_train.append(np.mean(np.array(mae_train)))

        mae_test = []
        for test_datum in test_data:
            if test_datum[0].size == 0:
                continue
            pi_pred = pi_net.predict(np.expand_dims(test_datum[0], axis=0),
                                     verbose=0)
            pi_pred = pi_pred[0, :, :, :]
            mae = np.mean((np.argmax(pi_pred, axis=2) == np.argmax(test_datum[1], axis=2)).astype(np.float))
            mae_test.append(mae)
            if np.isnan(mae):
                print(np.argmax(test_datum[1], axis=2))
                print(np.argmax(pi_pred, axis=2))
            else:
                mae_test.append(mae)
        mae_epoch_test.append(np.mean(np.array(mae_test)))

        plt.clf()
        plt.plot(mae_epoch_train, 'r')
        plt.plot(mae_epoch_test, 'g')
        plt.pause(1)

    pi_net.save_weights(model_weights)

mae_test = []
for test_datum in test_data:
    if test_datum[0].size == 0:
        continue
    pi_pred = pi_net.predict(np.expand_dims(test_datum[0], axis=0),
                             verbose=0)
    pi_pred = pi_pred[0, :, :, :]
    mae = np.mean((np.argmax(pi_pred, axis=2) == np.argmax(test_datum[1], axis=2)).astype(np.float))
    mae_test.append(mae)
    if np.isnan(mae):
        print(np.argmax(test_datum[1], axis=2))
        print(np.argmax(pi_pred, axis=2))
    else:
        mae_test.append(mae)

print(np.mean(np.array(mae_test)))

