from keras import backend as kb
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Lambda
from keras.models import Model
import numpy as np
feature_len = 38
n_people = 10
pose_info = ['right', 'front-right', 'front', 'front-left', 'left', 'left-back', 'back', 'right-back']
action_info = ['standing', 'walking', 'running']
interaction_info = ['no-interaction', 'int1', 'int2', 'int3', 'int4', 'int5', 'int6', 'int7', 'int8', 'int9', 'int10']
n_interaction_info = len(interaction_info)
group_act_info = ['act1', 'act2', 'act3', 'act4', 'act5', 'act6', 'act7', 'act8', 'act9', 'act10']
scene_act_info = ['act1', 'act2', 'act3', 'act4', 'act5', 'act6', 'act7', 'act8', 'act9', 'act10']


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
    pairwise_feature_input = Input(shape=(feature_len,))
    dense1 = Dense(units=32, activation='sigmoid')(pairwise_feature_input)
    dense2 = Dense(units=32, activation='sigmoid')(dense1)
    pairwise_activity_output = Dense(units=1, activation='sigmoid')(dense2)
    return Model(inputs=[pairwise_feature_input], outputs=[pairwise_activity_output])


def group_activity():
    pairwise_activity_input = Input(shape=[None, None, len(group_act_info)], name='pairwise_activity_input')
    pairwise_distance_input = Input(shape=[None, None], name='pairwise_distance_input')
    pairwise_distance_input_repeat = Lambda(lambda x: kb.repeat_elements(kb.expand_dims(pairwise_distance_input, 3),
                                                                         len(group_act_info),
                                                                         axis=3))(pairwise_distance_input)
    pairwise_activity_histogram = Lambda(lambda x: kb.mean(x[1]*x[0],
                                                           axis=2),
                                         name='weighted_mean_histogram')([pairwise_activity_input,
                                                                          pairwise_distance_input_repeat])
    dense1 = Dense(units=32, activation='sigmoid')(pairwise_activity_histogram)
    # dense2 = Dense(units=32, activation='sigmoid')(dense1)
    group_activity_output = Dense(units=len(group_act_info), activation='softmax', name='group_activity_output')(dense1)
    return Model(inputs=[pairwise_activity_input, pairwise_distance_input], outputs=[group_activity_output])


def my_categorical_crossentropy(target, output, from_logits=False):
    """Categorical crossentropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.

    # Returns
        Output tensor.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output,
                                len(output.get_shape()) - 1,
                                True)
        # manual computation of crossentropy
        _epsilon = tf.convert_to_tensor(kb.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        return - tf.reduce_sum(target * tf.log(output), axis=None)
    else:
        return tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                       logits=output)


def my_categorical_accuracy(y_true, y_pred):
    y_true = kb.argmax(y_true, axis=2)
    y_pred = kb.argmax(y_pred, axis=2)
    return tf.reduce_mean(tf.cast(y_true == y_pred, kb.floatx()), axis=None)


def pairwise_distance_mat():
    pairwise_feature_mat = Input(batch_shape=(None, None, None, feature_len), name='pairwise_feature_mat')
    pairwise_distances = Lambda(lambda x: kb.squeeze(x, axis=3))(pairwise_distance()(pairwise_feature_mat))
    return Model(inputs=[pairwise_feature_mat], outputs=[pairwise_distances])


def pairwise_interaction():
    pairwise_feature_input = Input(shape=(feature_len,))
    dense1 = Dense(units=32, activation='sigmoid')(pairwise_feature_input)
    pairwise_activity_output = Dense(units=len(group_act_info), activation='softmax')(dense1)
    return Model(inputs=[pairwise_feature_input], outputs=[pairwise_activity_output])


def pairwise_interaction_mat():
    pairwise_feature_mat = Input(batch_shape=(None, None, None, feature_len), name='pairwise_feature_mat')
    pairwise_interactions = Lambda(lambda x: kb.squeeze(x, axis=3))(pairwise_interaction()(pairwise_feature_mat))
    return Model(inputs=[pairwise_feature_mat], outputs=[pairwise_interactions])


if __name__ == "__main__":
    print(pairwise_interaction_mat().summary())

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

