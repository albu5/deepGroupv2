from keras import backend as kb
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Lambda
from keras.models import Model
import numpy as np
feature_len = 38


def pairwise_activity():
    pairwise_activity_input = Input(shape=[feature_len, 2], name='pairwise_input')
    dense1 = Dense(units=32, activation='sigmoid')(pairwise_activity_input)
    dense2 = Dense(units=32, activation='sigmoid')(dense1)
    pairwise_activity_output = Dense(units=10, activation='softmax')(dense2)
    return Model(inputs=[pairwise_activity_input], outputs=[pairwise_activity_output])


def min_max_group(y_true, y_pred):
    diag = tf.eye(num_rows=tf.shape(y_true)[2], batch_shape=kb.expand_dims(tf.shape(y_true)[0], axis=0))

    in_frame_row = kb.max(y_true, axis=1, keepdims=True)

    in_frame_col = kb.max(y_true, axis=2, keepdims=True)

    mask = kb.batch_dot(in_frame_col, in_frame_row, axes=(2, 1))

    intra_max = kb.max(y_pred + y_true + mask - diag - 2, axis=2)

    intra_min = kb.min(y_pred - y_true - mask + diag + 2, axis=2)

    inter_max = kb.max(y_pred - y_true + mask - 1, axis=2)

    return (kb.sum(inter_max - intra_max, axis=-1) + kb.epsilon()) / (kb.sum(in_frame_row, axis=-1) + kb.epsilon())


def pairwise_distance():
    pairwise_activity_input = Input(shape=[feature_len], name='pairwise_input')
    dense1 = Dense(units=32, activation='sigmoid')(pairwise_activity_input)
    dense2 = Dense(units=32, activation='sigmoid')(dense1)
    pairwise_activity_output = Dense(units=1, activation='sigmoid')(dense2)
    return Model(inputs=[pairwise_activity_input], outputs=[pairwise_activity_output])


def pairwise_distance_mat():
    pairwise_feature_mat = Input(batch_shape=(None, None, None, feature_len), name='pairwise_feature_mat')
    pairwise_distances = Lambda(lambda x: kb.squeeze(x, axis=3))(pairwise_distance()(pairwise_feature_mat))
    return Model(inputs=[pairwise_feature_mat], outputs=[pairwise_distances])


if __name__ == "__main__":
    a = [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [0, 0, 1, 1],
         [0, 0, 1, 1]]
    b = [[0.9, 0.1, 0.2, 0.2],
         [0.1, 0.9, 0.2, 0.2],
         [0.2, 0.2, 0.9, 0.8],
         [0.2, 0.2, 0.8, 0.9]]
    a = np.array(a).astype(np.float32)
    b = np.array(b).astype(np.float32)
    a = np.expand_dims(a, axis=0)
    b = np.expand_dims(b, axis=0)

    y_true = kb.variable(a)
    y_pred = kb.variable(b)
    print("y_true shape is: ", kb.eval(tf.shape(y_true)))
    print(kb.expand_dims(tf.shape(y_true)[0], axis=0))
    diag = tf.eye(num_rows=tf.shape(y_true)[2], batch_shape=kb.expand_dims(tf.shape(y_true)[0], axis=0))

    print('diag shape is: ', kb.eval(tf.shape(diag)))

    in_frame_row = kb.max(y_true, axis=1, keepdims=True)
    print(np.squeeze(kb.eval(in_frame_row)))

    in_frame_col = kb.max(y_true, axis=2, keepdims=True)
    print(np.squeeze(kb.eval(in_frame_col)))

    mask = kb.batch_dot(in_frame_col, in_frame_row, axes=(2, 1))
    print(np.squeeze(kb.eval(mask)))

    intra_max = kb.max(y_pred + y_true + mask - diag - 2, axis=2)
    print(np.squeeze(kb.eval(intra_max)))

    intra_min = kb.min(y_pred - y_true - mask + diag + 2, axis=2)
    print(np.squeeze(kb.eval(intra_min)))

    inter_max = kb.max(y_pred - y_true + mask - 1, axis=2)
    print(np.squeeze(kb.eval(inter_max)))

