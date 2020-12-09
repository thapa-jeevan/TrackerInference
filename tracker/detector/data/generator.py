import ast
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from tracker.detector.configs.dataset import IDX_TO_NAME, NAME_TO_IDX, CSVS_DIR, IMG_DIR, TARGET_SHAPE, TRAIN_CSV, \
    VALIDATION_CSV
from tracker.detector.utils.anchor import generate_default_boxes
from tracker.detector.utils.box_utils import compute_target


class DataGenerator(object):
    def __init__(self, phase='train', img_dir=None, _csvs_dir=None, target_shape=None):
        assert phase in ['train', 'valid']
        self.phase = phase
        self.default_boxes = generate_default_boxes()

        self.idx_to_name = IDX_TO_NAME
        self.name_to_idx = NAME_TO_IDX

        self.target_shape = TARGET_SHAPE if target_shape is None else TARGET_SHAPE
        self.img_dir = IMG_DIR if img_dir is None else img_dir

        _csvs_dir = CSVS_DIR if _csvs_dir is None else _csvs_dir
        csv_path = os.path.join(_csvs_dir, TRAIN_CSV if phase == 'train' else VALIDATION_CSV)
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def generate(self):
        for row_data in self.df.values:
            rel_img_path, img, bboxes, categories = self.preprocess_data(row_data)
            img, gt_confs, gt_locs = self.process_row(img, bboxes, categories)
            yield rel_img_path, img, gt_confs, gt_locs

    def preprocess_data(self, row_data):
        """
        It processes the row provided from the dataframe.
        It deals with the structure of the dataframe and  creates an abstraction for process_row function
        Args:
            row_data: a row of self.df

        Returns:
            rel_img_path : str - relative path of image
            img: PIL image
            bboxes: np array
            categories: list

        """
        rel_img_path, _, bboxes, _, categories = row_data
        img_path = os.path.join(self.img_dir, rel_img_path)
        img = Image.open(img_path)

        return rel_img_path, img, np.array(ast.literal_eval(bboxes)), ast.literal_eval(categories)

    def process_row(self, img, bboxes, categories):
        """
        Takes the image, the bounding boxes and does necessary augmentations
        and returns tf tensors.
        Args:
            img: PIL image
            bboxes : np array
            categories : list

        Returns:
            img: tf tensor
            gt_confs: tf tensor
            gt_locs: tf tensor
        """

        # if self.phase == 'train':
        #     img, bboxes = self.augment(img, bboxes)

        bboxes = bboxes / np.array(img.size * 2)

        img = np.array(img.resize(self.target_shape, 2), dtype=np.float32)

        img = (img / 127) - 1
        img = tf.constant(img, dtype=tf.float32)

        # if np.random.rand() < 0.75:
        #     img = color_aug(img)

        bboxes = tf.constant(bboxes, dtype=tf.float32)
        categories = tf.constant(categories, dtype=tf.int64)

        gt_confs, gt_locs = compute_target(self.default_boxes, bboxes, categories)

        return img, gt_confs, gt_locs

    # @staticmethod
    # def augment(img, bboxes):
    #     augmenters = []
    #
    #     if np.random.rand() < 0.5:
    #         augmenters.append(hflip)
    #
    #     augmenters.append(crop_with_aspect_ratio)
    #     augmenters.append(random_crop_with_ratio)
    #
    #     if len(augmenters) != 0:
    #         for _augmenter in augmenters:
    #             img, bboxes = _augmenter(img, bboxes)
    #
    #     return img, bboxes
