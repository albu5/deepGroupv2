# pairwise interactions and group formation matrix

import numpy as np
from utils import read_anno, pairwise_features_interaction_batch, pairwise_features_at_time_t, \
    pairwise_interactions_at_time_t, draw_pairwise_features_at_time_t, draw_groups_at_time_t
from networks import group_activity, my_categorical_crossentropy, my_categorical_accuracy
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
import sys
sys.setrecursionlimit(10000)

feature_len = 38
n_people = 10
n_epoch = 20
pose_info = ['right', 'front-right', 'front', 'front-left', 'left', 'left-back', 'back', 'right-back']
action_info = ['standing', 'walking', 'running']
interaction_info = ['no-interaction', 'int1', 'int2', 'int3', 'int4', 'int5', 'int6', 'int7', 'int8', 'int9', 'int10']
n_interaction_info  = len(interaction_info)
group_act_info = ['act1', 'act2', 'act3', 'act4', 'act5', 'act6', 'act7', 'act8', 'act9', 'act10']
scene_act_info = ['act1', 'act2', 'act3', 'act4', 'act5', 'act6', 'act7', 'act8', 'act9', 'act10']

keras.losses.my_categorical_crossentropy = my_categorical_crossentropy
keras.metrics.my_categorical_accuracy = my_categorical_accuracy
warnings.filterwarnings("ignore", ".*Using default event loop until function specific to this GUI is implemented.*")

anno_fmt = "../data/final_csv_anno/anno%2.2d.mat"
img_fmt = "../data/images/seq%2.2d/frame%6.6d.jpg"
title_fmt = "Seq: %2.2d | Frame: %6.6d"
group_fmt = "../data/group_int/group_int%2.2d.mat"
mask_fmt = "../data/mask/mask%2.2d.mat"
model_json = "../models/group_activity/ga.json"
model_weights = "../models/group_activity/ga.h5"
data_file = "./tmp/ga_data.bin"
interaction_fmt = "../data/pairwise_int/int%2.2d.mat"
ax = plt.gca()

train_data = []
test_data = []
data = []

test_seq = [1, 2, 7, 12, 13, 19, 20, 21, 26, 27, 30]

# create if to be written directories do not exist
if not os.path.isdir(os.path.dirname(data_file)):
    print('creating ' + os.path.dirname(data_file) + 'directory...')
    os.makedirs(os.path.dirname(data_file))

if not os.path.isdir(os.path.dirname(model_json)):
    print('creating ' + os.path.dirname(model_json) + 'directory...')
    os.makedirs(os.path.dirname(model_json))

if not isfile(data_file):
    for _seq in range(0, 33):
        seq = _seq + 1
        if seq == 6:
            continue
        anno_data = read_anno(anno_path=anno_fmt % seq)
        group_data = loadmat(group_fmt % seq)['group_int']
        mask_data = loadmat(mask_fmt % seq)['mask_mat']
        interaction_data = loadmat(interaction_fmt % seq)['interaction']
        valid_persons_int_data = np.sum(np.sum(interaction_data, axis=2) > 0, axis=1) > 0

        mae_seq = []
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

            # convert ga to on hot
            ga = np.repeat(np.reshape(ga, (ga.size, 1)), repeats=len(group_act_info), axis=1)
            _ga = np.reshape(np.arange(len(group_act_info)), (1, len(group_act_info)))
            _ga = np.repeat(_ga, repeats=ga.shape[0], axis=0)
            ga = (ga == _ga).astype(np.float)

            # convert interaction to ont hot (toggle comments to use original interaction in place of from ga)
            interaction = np.expand_dims(interaction, axis=2)
            # interaction = np.expand_dims(interaction_data[:, :, t], axis=2)
            # interaction = interaction[:, valid_persons_int_data, :]
            # interaction = interaction[valid_persons_int_data, :, :]
            # change repeats argument when using original interaction
            interaction = np.repeat(interaction, repeats=len(group_act_info), axis=2)
            _interaction = np.reshape(np.arange(len(group_act_info)), (1, 1, len(group_act_info)))
            _interaction = np.repeat(_interaction, repeats=interaction.shape[0], axis=0)
            _interaction = np.repeat(_interaction, repeats=interaction.shape[0], axis=1)
            interaction = interaction == _interaction

            if interaction.shape[0] != gd.shape[0]:
                print(interaction.shape, gd.shape)
            if seq in test_seq:
                test_data.append((px, interaction, gd, ga, mask))
            else:
                train_data.append((px, interaction, gd, ga, mask))
    with open(data_file, mode='wb') as f:
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
    group_activity_net = group_activity()
    group_activity_net.load_weights(model_weights)
else:
    group_activity_net = group_activity()
    group_activity_net.compile(optimizer=adam(0.0001), loss=my_categorical_crossentropy,
                               metrics=[my_categorical_accuracy])

    mae_epoch = []
    for n_epoch in range(n_epoch):
        shuffle(train_data)
        mae_train = []
        for train_datum in train_data:
            if np.sum(train_datum[2].astype(np.float), axis=None) > 0:
                ga_pred = group_activity_net.predict([np.expand_dims(train_datum[1], axis=0),
                                                      np.expand_dims(train_datum[2], axis=0)],
                                                     verbose=0)
                ga_pred = np.squeeze(ga_pred)
                mae = np.mean((np.argmax(ga_pred, axis=1) == np.argmax(train_datum[3], axis=1)).astype(np.float))
                mae_train.append(mae)
                group_activity_net.train_on_batch([np.expand_dims(train_datum[1], axis=0),
                                                   np.expand_dims(train_datum[2], axis=0)],
                                                  np.expand_dims(train_datum[3], axis=0))
        mae_epoch.append(np.mean(np.array(mae_train)))
        plt.clf()
        plt.plot(mae_epoch)
        plt.pause(0.03)

    group_activity_net.save_weights(model_weights)

mae_train = []
for train_datum in test_data:
    if np.sum(train_datum[2].astype(np.float), axis=None) > 0:
        ga_pred = group_activity_net.predict([np.expand_dims(train_datum[1], axis=0),
                                              np.expand_dims(train_datum[2], axis=0)],
                                             verbose=0)
        ga_pred = np.squeeze(ga_pred)
        mae = np.mean((np.argmax(ga_pred, axis=1) == np.argmax(train_datum[3], axis=1)).astype(np.float))
        if np.isnan(mae):
            print(np.argmax(train_datum[3], axis=1))
            print(np.argmax(ga_pred, axis=1))
        else:
            mae_train.append(mae)

print(np.mean(np.array(mae_train)))

