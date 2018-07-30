import numpy as np
from utils import read_anno, pairwise_features_interaction_batch, pairwise_features_at_time_t, \
    pairwise_interactions_at_time_t, draw_pairwise_features_at_time_t, draw_groups_at_time_t
from networks import pairwise_distance_mat, min_max_group
from matplotlib import pyplot as plt
from skimage.io import imread
from scipy.io import loadmat
from keras.optimizers import adam
from random import shuffle
import pickle
from os.path import isfile
from sklearn.cluster import DBSCAN
from keras.models import load_model
import os
from keras import backend as kb
import keras.losses
import warnings

warnings.filterwarnings("ignore", ".*Using default event loop until function specific to this GUI is implemented.*")
warnings.filterwarnings("ignore", ".*for unknown op.*")


keras.losses.min_max_group = min_max_group

feature_len = 38

anno_fmt = "../data/final_csv_anno/anno%2.2d.mat"
img_fmt = "../data/images/seq%2.2d/frame%6.6d.jpg"
title_fmt = "Seq: %2.2d | Frame: %6.6d"
group_fmt = "../data/group_int/group_int%2.2d.mat"
mask_fmt = "../data/mask/mask%2.2d.mat"
model_file = "../models/pairwise_distance/pd30.h5"
data_file = "./tmp/pair2grp.bin"
ax = plt.gca()

test_seq = [i for i in range(10)]

# create if to be written directories do not exist
if not os.path.isdir(os.path.dirname(data_file)):
    print('creating ' + os.path.dirname(data_file) + 'directory...')
    os.makedirs(os.path.dirname(data_file))

if not os.path.isdir(os.path.dirname(model_file)):
    print('creating ' + os.path.dirname(model_file) + 'directory...')
    os.makedirs(os.path.dirname(model_file))

if os.path.isfile(model_file):
    print('loading ' + model_file + ' ...')
    pairwise_distance_net = load_model(model_file, custom_objects={'kb': kb})
else:
    pairwise_distance_net = pairwise_distance_mat()
    pairwise_distance_net.compile(optimizer=adam(0.0005), loss=min_max_group, metrics=['mae'])

    train_data = []
    test_data = []

    if not isfile(data_file):
        for _seq in range(0, 33):
            seq = _seq + 1
            anno_data = read_anno(anno_path=anno_fmt % seq)
            group_data = loadmat(group_fmt % seq)['group_int']
            mask_data = loadmat(mask_fmt % seq)['mask_mat']
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

                if seq in test_seq:
                    test_data.append((px, gd, mask))
                else:
                    train_data.append((px, gd, mask))
        with open(data_file, mode='wb') as f:
            pickle.dump({'train_data': train_data, 'test_data': test_data}, f)
    else:
        with open(data_file, mode='rb') as f:
            data = pickle.load(f)
        train_data = data['train_data']
        test_data = data['test_data']


    mae_epoch = []
    for n_epoch in range(30):
        shuffle(train_data)
        mae_train = []
        for train_datum in train_data:
            if np.sum(train_datum[2].astype(np.float), axis=None) > 0:
                gd_pred = pairwise_distance_net.predict(np.expand_dims(train_datum[0], axis=0), verbose=0)
                mae = np.sum(np.abs(gd_pred - train_datum[1]), axis=None) / np.sum(train_datum[2].astype(np.float), axis=None)
                mae_train.append(mae)
                pairwise_distance_net.train_on_batch(np.expand_dims(train_datum[0], axis=0), np.expand_dims(train_datum[1], axis=0))
        mae_epoch.append(np.mean(np.array(mae_train)))
        plt.clf()
        plt.plot(mae_epoch)
        plt.pause(0.03)

    pairwise_distance_net.save(model_file)

for _seq in range(0, 33):
    seq = _seq + 1
    if seq not in test_seq:
        continue

    anno_data = read_anno(anno_path=anno_fmt % seq)
    group_data = loadmat(group_fmt % seq)['group_int']
    mask_data = loadmat(mask_fmt % seq)['mask_mat']
    print('seq: ', seq)
    for t in range(0, anno_data.shape[0], 10):
        pf = pairwise_features_at_time_t(anno_data, t)
        pi = pairwise_interactions_at_time_t(anno_data, t)
        px, _ = pairwise_features_interaction_batch(pf, pi)
        mask = mask_data[t, :, :]

        n_people = anno_data.shape[2]

        img = imread(img_fmt % (seq, t))
        plt.clf()
        plt.imshow(img)

        px = np.reshape(px, newshape=(n_people, n_people, feature_len))
        gd_pred = np.squeeze(pairwise_distance_net.predict(np.expand_dims(px, axis=0), verbose=0))
        gd_pred *= mask
        not_in_frame = np.sum(mask, axis=1) == 0
        if np.sum(not_in_frame) > 0:
            gd_pred += np.matmul(np.reshape(not_in_frame, newshape=(n_people, 1)),
                                 np.reshape(not_in_frame, newshape=(1, n_people))) / np.sum(not_in_frame)

        af = DBSCAN(eps=0.55, metric='precomputed', min_samples=0, algorithm='auto', n_jobs=1)
        af.fit(1 - gd_pred)
        af_labels = np.array(af.labels_)
        adj_mat = np.repeat(np.expand_dims(af_labels, axis=1), n_people, axis=1) == \
                  np.transpose(np.repeat(np.expand_dims(af_labels, axis=1), n_people, axis=1))
        draw_groups_at_time_t(pf, adj_mat, plt.gca())
        plt.pause(0.003)

