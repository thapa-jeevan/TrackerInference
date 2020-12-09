import tensorflow as tf
from tensorflow.keras import losses

from tracker.detector.configs.dataset import NO_CLASSES
from tracker.detector.configs.loss import neg_ratio


def hard_negative_mining(loss, gt_confs, neg_ratio):
    pos_idx = gt_confs > 0
    num_pos = tf.reduce_sum(tf.dtypes.cast(pos_idx, tf.int32), axis=1)
    num_neg = num_pos * neg_ratio

    descending_loss_indices = tf.argsort(loss, axis=1, direction='DESCENDING')
    ascending_rank = tf.argsort(descending_loss_indices, axis=1)
    neg_idx = ascending_rank < tf.expand_dims(num_neg, 1)

    return pos_idx, neg_idx


class SSDLoss(object):
    def __init__(self):
        self.neg_ratio = neg_ratio
        self.num_classes = NO_CLASSES

    def __call__(self, confs, locs, gt_confs, gt_locs):
        cross_entropy = losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        temp_loss = cross_entropy(gt_confs, confs)
        pos_idx, neg_idx = hard_negative_mining(temp_loss, gt_confs, self.neg_ratio)

        cross_entropy = losses.SparseCategoricalCrossentropy(from_logits=True, reduction='sum')

        smooth_l1_loss = losses.Huber(reduction='sum')

        conf_idx = tf.math.logical_or(pos_idx, neg_idx)
        conf_loss = cross_entropy(gt_confs[conf_idx], confs[conf_idx])

        loc_loss = smooth_l1_loss(gt_locs[pos_idx], locs[pos_idx])

        num_pos = tf.reduce_sum(tf.dtypes.cast(pos_idx, tf.float32))
        num_confs = tf.reduce_sum(tf.dtypes.cast(conf_idx, tf.float32))

        conf_loss = conf_loss / num_confs
        loc_loss = loc_loss / num_pos

        return conf_loss, loc_loss
