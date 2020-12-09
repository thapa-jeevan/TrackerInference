import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(os.path.dirname(ROOT_DIR), 'data')

WEIGHTS_DIR = os.path.join(DATA_DIR, 'weights')

CHECKPOINTS_DIR = os.path.join(DATA_DIR, 'checkpoints')

TRAIN_OUTCOMES_DIR = os.path.join(os.path.dirname(ROOT_DIR), 'train_outcomes')

DETECTOR_CHECKPOINTS_DIR = os.path.join(CHECKPOINTS_DIR, 'detector')

# Training
DETECTOR_ARCHITECTURE = 'ssd512'

# Test
DETECTOR_CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR, 'test', 'detector')
