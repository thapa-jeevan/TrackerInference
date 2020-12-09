import os

from tracker.configs.settings import DATA_DIR, DETECTOR_ARCHITECTURE

IDX_TO_NAME = ['Vehicle']

NAME_TO_IDX = {v: k for k, v in enumerate(IDX_TO_NAME)}

CSVS_DIR = os.path.join(DATA_DIR, 'processed')

IMG_DIR = os.path.join(DATA_DIR, 'raw', 'images')

TARGET_SHAPE = (300, 300) if DETECTOR_ARCHITECTURE == 'ssd300' else (512, 512)

TRAIN_CSV = 'train.csv'
VALIDATION_CSV = 'valid.csv'

NO_CLASSES = len(IDX_TO_NAME) + 1