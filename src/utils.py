from scipy.io import loadmat
from scipy.misc import imread
import numpy as np
from scipy.stats import linregress
from keras.utils import to_categorical
from matplotlib import patches
from random import random as rand
from scipy.sparse import csgraph


# COLORS = [(rand(), rand(), rand()) for i in range(20)]
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

# min and max labels for pose and action (excluding pose and action labels where a person is not in scene)
pose_min = 1
pose_max = 8
# 0: invalid pose, 1 to 8 directions?
action_min = 1
action_max = 3
# 0: standing, 1:walking, 2:running ?
n_interaction_classes = 10
# 0: no interaction, 1:?, 2:

# pls check these orders
pose_info = ['right', 'front-right', 'front', 'front-left', 'left', 'left-back', 'back', 'right-back']
action_info = ['standing', 'walking', 'running']
interaction_info = ['no-interaction', 'int1', 'int2', 'int3', 'int4', 'int5', 'int6', 'int7', 'int8', 'int9', 'int10']
group_act_info = ['act1', 'act2', 'act3', 'act4', 'act5', 'act6', 'act7', 'act8', 'act9', 'act10']
scene_act_info = ['act1', 'act2', 'act3', 'act4', 'act5', 'act6', 'act7', 'act8', 'act9', 'act10']


def read_anno(anno_path):
    # reads annotations from csv_anno folder in a 3D array anno
    # anno[t, :, i] is a descriptor for i_th person at time t
    # anno[t, :, i] has 46 values, first 4 are bounding box coordinates
    # 5th and 6th are pose and action respectively
    # 7th to 26 are group member information appropriately padded by zeros
    # 27th to 46 are pairwise interaction information appropriately padded by zeros
    anno = loadmat(anno_path)
    return anno['peds_attrs']


def one_hot(x, x_min, x_max, default=0):
    # our own one hot function
    # x is numpy 1D numpy array
    # x_min is minimum label value
    # x_max is maximum label value
    # in case of values in x being out of range, default value is used
    y = np.zeros(shape=[x.shape[0], x_max - x_min + 1])
    for i in range(x.shape[0]):
        # print(x[i], x_min)
        if x[i] < x_min or x[i] > x_max:
            y[i, default] = 1
        else:
            y[i, int(x[i] - x_min)] = 1

    return y


def pairwise_features_at_time_t(anno, t, del_t=10):
    # returns 2D list of pairwise features at time t in anno
    # anno is matrix annotation saved in csv_anno folder
    # del_t is block length on which analysis is to be done,should be generally fixed to 10 frames
    n_people = anno.shape[2]

    # initialize by empty features
    feature = np.zeros(shape=[2 * (8 + pose_max - pose_min + 1 + action_max - action_min + 1)])
    pairs = [[feature for i in range(n_people)] for j in range(n_people)]

    for i in range(n_people):
        for j in range(n_people):
            ped_i = np.squeeze(anno[t:t + del_t, :, i])
            ped_j = np.squeeze(anno[t:t + del_t, :, j])
            # print(ped_i.shape, ped_j.shape)

            if len(ped_i.shape) == 1:
                ped_i = np.expand_dims(ped_i, axis=0)
                ped_j = np.expand_dims(ped_j, axis=0)

            bbs_i = ped_i[:, 0:4]
            bbs_j = ped_j[:, 0:4]

            # if pose is 0 then it's an invalid or inactive person in scene
            valid_i = not(np.all(ped_i[:, 4] == 0))
            valid_j = not(np.all(ped_j[:, 4] == 0))

            # compute velocity and mean position from 10 frame data
            # and median poses and action to filter noisy observations
            if valid_i and valid_j:
                _del_t = bbs_j.shape[0]
                vx_i, mx_i, _, _, _ = linregress(np.arange(_del_t), np.squeeze(bbs_i[:, 0]))
                vy_i, my_i, _, _, _ = linregress(np.arange(_del_t), np.squeeze(bbs_i[:, 1]))
                vw_i, mw_i, _, _, _ = linregress(np.arange(_del_t), np.squeeze(bbs_i[:, 2]))
                vh_i, mh_i, _, _, _ = linregress(np.arange(_del_t), np.squeeze(bbs_i[:, 3]))
                vx_j, mx_j, _, _, _ = linregress(np.arange(_del_t), np.squeeze(bbs_j[:, 0]))
                vy_j, my_j, _, _, _ = linregress(np.arange(_del_t), np.squeeze(bbs_j[:, 1]))
                vw_j, mw_j, _, _, _ = linregress(np.arange(_del_t), np.squeeze(bbs_j[:, 2]))
                vh_j, mh_j, _, _, _ = linregress(np.arange(_del_t), np.squeeze(bbs_j[:, 3]))
                pose_i = np.median(a=one_hot(ped_i[:, 4], x_min=pose_min, x_max=pose_max), axis=0)
                pose_j = np.median(a=one_hot(ped_j[:, 4], x_min=pose_min, x_max=pose_max), axis=0)
                action_i = np.median(a=one_hot(ped_i[:, 5], x_min=action_min, x_max=action_max), axis=0)
                action_j = np.median(a=one_hot(ped_j[:, 5], x_min=action_min, x_max=action_max), axis=0)
                feature = np.hstack([np.array([vx_i, vy_i, vw_i, vh_i, mx_i, my_i, mw_i, mh_i]), pose_i, action_i,
                                     np.array([vx_j, vy_j, vw_j, vh_j, mx_j, my_j, mw_j, mh_j]), pose_j, action_j])
            else:
                # fill with empty features if any person is not in scene at time t
                feature = np.zeros(shape=[2 * (8 + pose_max - pose_min + 1 + action_max - action_min + 1)])
            pairs[i][j] = feature
    return pairs


def pairwise_interactions_at_time_t(anno, t, del_t=10):
    # returns 2D list of pairwise features at time t in anno
    # anno is matrix annotation saved in csv_anno folder
    # del_t is block length on which analysis is to be done,should be generally fixed to 10 frames
    n_people = anno.shape[2]

    # initialize by empty interactions... but does 0 denote invalid interaction or no interaction?
    pairs = [[0 for i in range(n_people)] for j in range(n_people)]

    for i in range(n_people):
        for j in range(n_people):
            ped_i = np.squeeze(anno[t:t + del_t, :, i])
            ped_j = np.squeeze(anno[t:t + del_t, :, j])

            if len(ped_i.shape) == 1:
                ped_i = np.expand_dims(ped_i, axis=0)
                ped_j = np.expand_dims(ped_j, axis=0)

            # if pose is 0 then it's an invalid or inactive person in scene
            valid_i = not(np.all(ped_i[:, 4] == 0))
            valid_j = not(np.all(ped_j[:, 4] == 0))
            if valid_i and valid_j:
                feature = ped_i[0, 26 + j]
            else:
                feature = 0
            pairs[i][j] = feature
    return pairs


def pairwise_features_interaction_batch(features, interactions):
    # create a batch of features and interactions from 2D list
    n_people = len(features)
    feature_len = features[0][0].shape[0]
    x = np.zeros(shape=(n_people*n_people, feature_len))
    y = np.zeros(shape=(n_people * n_people,))

    counter = 0
    for i in range(n_people):
        for j in range(n_people):
            x[counter, :] = features[i][j]
            y[counter] = interactions[i][j]
            counter += 1
    return x, to_categorical(y-1, num_classes=n_interaction_classes)


def draw_pairwise_features_at_time_t(features, interaction, ax, anno, t):
    n_people = len(features)
    scene_activity = -1
    for i in range(n_people):
        ped_i = features[i][0]

        # don't draw invalid ones
        if np.sum(ped_i[4:8]) == 0:
            continue

        # top left are x, y
        x = ped_i[4]
        y = ped_i[5]
        w = ped_i[6]
        h = ped_i[7]

        # add rectangles, color coding according to group activity
        color_code = int(anno[t, 46, i])
        group_activity = int(anno[t, 47, i])
        scene_activity = int(anno[t, 48, i])
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, color=COLORS[color_code]))

        # add velocity arrow
        arrow_len = 50
        ax.arrow(x+w/2,
                 y+h,
                 arrow_len*ped_i[0], arrow_len*ped_i[1],
                 head_width=5.0, head_length=5.0,
                 linewidth=2.0)

        # add pose
        pose_i = np.argmax(ped_i[8:16])
        ax.text(x, y, pose_info[pose_i],
                color=COLORS[color_code], fontsize=8,
                bbox={'facecolor': COLORS[color_code], 'alpha': 0.2, 'pad': 1})

        # add action
        action_i = np.argmax(ped_i[16:19])
        ax.text(x, y + h + 8, action_info[action_i],
                color=COLORS[color_code], fontsize=8,
                bbox={'facecolor': COLORS[color_code], 'alpha': 0.2, 'pad': 1})

        # add group activity
        ax.text(x, y + h/2 + 8, group_act_info[group_activity],
                color=COLORS[color_code], fontsize=8,
                bbox={'facecolor': COLORS[color_code], 'alpha': 0.2, 'pad': 1})

        # add pairwise interactions
        for j in range(n_people):
            if j is not i:
                ped_j = features[j][0]

                # skip invalid persons
                if np.sum(ped_j[4:8]) == 0:
                    continue

                ax.arrow(x + w / 2,
                         y,
                         ped_j[4] + ped_j[6] / 2 - x,
                         ped_j[5] + ped_j[7] / 2 - y,
                         head_width=4.0, head_length=4.0,
                         linewidth=1.0,
                         color=COLORS[color_code])
                ax.text(0.7*x + 0.3*ped_j[4], 0.7*y + 0.3*(ped_j[5] + ped_j[7]), interaction_info[int(interaction[i][j])],
                        color=COLORS[color_code], fontsize=8,
                        bbox={'facecolor': COLORS[color_code], 'alpha': 0.2, 'pad': 1})

    # add scene activity
    if scene_activity > 0:
        ax.text(5, 20, scene_act_info[scene_activity],
                color='w', fontsize=12,
                bbox={'facecolor': 'w', 'alpha': 0.2, 'pad': 1})


def draw_groups_at_time_t(features, gd, ax):
    n_people = len(features)
    for i in range(n_people):
        ped_i = features[i][0]

        # don't draw invalid ones
        if np.sum(ped_i[4:8]) == 0:
            continue

        (_, gl) = csgraph.connected_components(gd, directed=False)

        # top left are x, y
        x = int(ped_i[4])
        y = int(ped_i[5])
        w = int(ped_i[6])
        h = int(ped_i[7])

        # add rectangles, color coding according to group activity
        color_code = gl[i]
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, color=COLORS[color_code]))

        # add velocity arrow
        arrow_len = 50
        ax.arrow(x+w/2,
                 y+h,
                 arrow_len*ped_i[0], arrow_len*ped_i[1],
                 head_width=5.0, head_length=5.0,
                 linewidth=2.0)

        # add pose
        pose_i = np.argmax(ped_i[8:16])
        ax.text(x, y, pose_info[pose_i],
                color=COLORS[color_code], fontsize=8,
                bbox={'facecolor': COLORS[color_code], 'alpha': 0.2, 'pad': 1})

        # add action
        action_i = np.argmax(ped_i[16:19])
        ax.text(x, y + h + 8, action_info[action_i],
                color=COLORS[color_code], fontsize=8,
                bbox={'facecolor': COLORS[color_code], 'alpha': 0.2, 'pad': 1})


if __name__ == "__main__":
    # x = np.arange(7) + 1
    # y = one_hot(x[1], 1, 8)
    # print(y)
    anno_str = "../data/csv_anno/anno01.mat"
    x = read_anno(anno_path=anno_str)
    # print(x.shape)
    pf = pairwise_features_at_time_t(x, 0)
    pi = pairwise_interactions_at_time_t(x, 0)
    print(pi)
    px, py = pairwise_features_interaction_batch(pf, pi)
    print(px.shape, py.shape)


