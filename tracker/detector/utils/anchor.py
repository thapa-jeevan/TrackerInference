import itertools
import math

import tensorflow as tf
from tracker.detector.configs.anchor import RATIOS, SCALES, FM_SIZES


def generate_default_boxes():
    default_boxes = []

    for m, fm_size in enumerate(FM_SIZES):
        for i, j in itertools.product(range(fm_size), repeat=2):
            cx = (j + 0.5) / fm_size
            cy = (i + 0.5) / fm_size
            default_boxes.append([cx, cy, SCALES[m], SCALES[m]])

            default_boxes.append([cx, cy, math.sqrt(SCALES[m] * SCALES[m + 1]), math.sqrt(SCALES[m] * SCALES[m + 1])])

            for ratio in RATIOS[m]:
                r = math.sqrt(ratio)
                default_boxes.append([cx, cy, SCALES[m] * r, SCALES[m] / r])
                default_boxes.append([cx, cy, SCALES[m] / r, SCALES[m] * r])

    default_boxes = tf.constant(default_boxes)
    default_boxes = tf.clip_by_value(default_boxes, 0.0, 1.0)

    return default_boxes
