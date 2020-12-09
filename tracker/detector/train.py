import argparse
import os
import time

import tensorflow as tf
from tensorflow.keras import optimizers
from tracker.detector.utils.config_logger import log_training_configs

from tracker.configs.settings import DETECTOR_ARCHITECTURE, DETECTOR_CHECKPOINTS_DIR
from tracker.detector.configs.dataset import NO_CLASSES
from tracker.detector.configs.loss import LOCALIZATION_WEIGHT, CLASSIFICATION_WEIGHT, WEIGHT_DECAY
from tracker.detector.data import create_batch_generator
from tracker.detector.models import get_network
from tracker.detector.utils.loss import SSDLoss

parser = argparse.ArgumentParser()

parser.add_argument('--commit', type=str)
parser.add_argument('--batch-size', default=32, type=int)

parser.add_argument('--num-epochs', default=200, type=int)
parser.add_argument('--num-batches', default=-1, type=int)

parser.add_argument('--initial-lr', default=1e-4, type=float)

args = parser.parse_args()


@tf.function
def train_step(detector, imgs, gt_confs, gt_locs, criterion, optimizer):
    with tf.GradientTape() as tape:
        pred_confs, pred_locs = detector(imgs)
        conf_loss, loc_loss = criterion(pred_confs, pred_locs, gt_confs, gt_locs)
        l2_loss = tf.math.reduce_sum([tf.nn.l2_loss(t) for t in detector.trainable_variables])

        loss = CLASSIFICATION_WEIGHT * conf_loss + LOCALIZATION_WEIGHT * loc_loss + WEIGHT_DECAY * l2_loss

        gradients = tape.gradient(loss, detector.trainable_variables)
        optimizer.apply_gradients(zip(gradients, detector.trainable_variables))

    return loss, conf_loss, loc_loss, l2_loss


if __name__ == '__main__':
    dir_train_outcome, checkpoints_dir, logs_dir = log_training_configs(args.commit)

    batch_generator, val_generator, info = create_batch_generator(args.batch_size, args.num_batches, 'train')

    detector = get_network(NO_CLASSES, DETECTOR_ARCHITECTURE)
    criterion = SSDLoss()

    steps_per_epoch = info['length'] // args.batch_size

    lr_fn = optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[
            int(steps_per_epoch * args.num_epochs * 1 / 5),
            int(steps_per_epoch * args.num_epochs * 2 / 5),
            int(steps_per_epoch * args.num_epochs * 3 / 5),
            int(steps_per_epoch * args.num_epochs * 4 / 5),
        ],
        values=[args.initial_lr, args.initial_lr * 0.1, args.initial_lr * 0.01, args.initial_lr * 0.005,
                args.initial_lr * 0.001]
    )

    optimizer = optimizers.Adam(learning_rate=lr_fn)

    train_log_dir = os.path.join(logs_dir, 'train')
    valid_log_dir = os.path.join(logs_dir, 'valid')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

    print("Training has started !!!")

    for epoch in range(args.num_epochs):
        start = time.time()

        avg_loss = 0.0
        avg_conf_loss = 0.0
        avg_loc_loss = 0.0

        avg_val_loss = 0.0
        avg_val_conf_loss = 0.0
        avg_val_loc_loss = 0.0

        for i, (_, imgs, gt_confs, gt_locs) in enumerate(batch_generator):
            loss, conf_loss, loc_loss, l2_loss = train_step(detector, imgs, gt_confs, gt_locs, criterion, optimizer)
            print('\r Training Step : {} / {}   Loss = {}'.format(i, steps_per_epoch, loss), end='')

            avg_loss = (avg_loss * i + loss.numpy()) / (i + 1)
            avg_conf_loss = (avg_conf_loss * i + conf_loss.numpy()) / (i + 1)
            avg_loc_loss = (avg_loc_loss * i + loc_loss.numpy()) / (i + 1)

        save_path = os.path.join(DETECTOR_CHECKPOINTS_DIR, 'ssd_epoch_{}.h5'.format(epoch + 1))
        detector.save_weights(save_path)

        for i, (_, imgs, gt_confs, gt_locs) in enumerate(val_generator):
            val_confs, val_locs = detector(imgs)
            val_conf_loss, val_bbox_loss = criterion(val_confs, val_locs, gt_confs, gt_locs)
            val_loss = val_conf_loss + val_bbox_loss
            avg_val_loss = (avg_val_loss * i + val_loss.numpy()) / (i + 1)
            avg_val_conf_loss = (avg_val_conf_loss * i + val_conf_loss.numpy()) / (i + 1)
            avg_val_bbox_loss = (avg_val_bbox_loss * i + val_bbox_loss.numpy()) / (i + 1)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', avg_loss, step=epoch)
            tf.summary.scalar('conf_loss', avg_conf_loss, step=epoch)
            tf.summary.scalar('loc_loss', avg_loc_loss, step=epoch)

        with valid_summary_writer.as_default():
            tf.summary.scalar('loss', avg_val_loss, step=epoch)
            tf.summary.scalar('conf_loss', avg_val_conf_loss, step=epoch)
            tf.summary.scalar('loc_loss', avg_val_loc_loss, step=epoch)

        print(
            '\r Epoch: {}   Time: {:.2}s \n'
            '\t Training   | Loss: {:.4f}  -  Classification: {:.4f}  -  Localization: {:.4f} \n'
            '\t Validation | Loss: {:.4f}  -  Classification: {:.4f}  -  Localization: {:.4f}'
                .format(
                epoch + 1, time.time() - start,
                avg_loss, avg_conf_loss, avg_loc_loss,
                avg_val_loss, avg_val_conf_loss, avg_val_loc_loss)
        )
