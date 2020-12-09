import datetime
import os

from tracker.configs.settings import TRAIN_OUTCOMES_DIR


# from tracker.detector.configs import IDX_TO_NAME, CSVS_DIR, IMG_DIR, TARGET_SHAPE, TRAIN_CSV, VALIDATION_CSV


def create_train_dir(commit):
    now = datetime.datetime.now()
    train_time = now.strftime("%Y-%h-%d-%H:%M-")

    commit = '_'.join(commit.split(' '))

    folder_train = train_time + commit
    dir_train_outcome = os.path.join(TRAIN_OUTCOMES_DIR, folder_train)

    os.makedirs(dir_train_outcome)

    return dir_train_outcome


def log_training_configs(commit):
    dir_train_outcome = create_train_dir(commit)
    checkpoints_dir = os.path.join(dir_train_outcome, 'checkpoints')
    logs_dir = os.path.join(dir_train_outcome, 'logs')

    os.makedirs(checkpoints_dir)
    os.makedirs(logs_dir)
    # train_config_file = os.path.join(dir_train_outcome, 'train.config')
    #
    # with open(train_config_file, 'w') as _file:
    #     _file.write(IDX_TO_NAME)
    #     _file.write(CSVS_DIR)
    #     _file.write(IMG_DIR)
    #     _file.write(TARGET_SHAPE)
    #     _file.write(TRAIN_CSV)
    #     _file.write(VALIDATION_CSV)

    return dir_train_outcome, checkpoints_dir, logs_dir
