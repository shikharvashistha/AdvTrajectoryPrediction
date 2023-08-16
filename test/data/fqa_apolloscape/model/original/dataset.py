import os

import sys
sys.path.append('./')
from data.TrajectoryDataset import TrajectoryDataset
from utils.misc_utils import get_files, get_dirs
import numpy as np

tr_dir = "/z/qzzhang/prediction/data/apolloscape/prediction_train"
val_dir = "/z/qzzhang/prediction/data/apolloscape/prediction_val"
te_dir = "/z/qzzhang/prediction/data/apolloscape/prediction_test"

with open("data/apolloscape/scale.txt", 'r') as f:
    scale = float(f.readline())

with open("data/apolloscape/mean.txt", 'r') as f:
    mean = np.array([float(i) for i in f.readline().split(' ')])

tr_data_files = [os.path.join(tr_dir, filepath) for filepath in get_files(tr_dir)]
val_data_files = [os.path.join(val_dir, filepath) for filepath in get_files(val_dir)]
te_data_files = [os.path.join(te_dir, filepath) for filepath in get_files(te_dir)]

def get_dataset(dtype, max_length=12, min_length=12, skip_frames=1, burn_in_steps=8, frame_shift=1, **kwargs):
    if dtype == 'tr':
        return TrajectoryDataset(tr_data_files, max_length=max_length, skip_frames=skip_frames,
                                 frame_shift=frame_shift, burn_in_steps=burn_in_steps, min_length=min_length, mean=mean, scale=scale, **kwargs)
    elif dtype == 'te':
        return TrajectoryDataset(te_data_files, max_length=max_length, skip_frames=skip_frames,
                                 frame_shift=frame_shift, burn_in_steps=burn_in_steps, min_length=min_length, mean=mean, scale=scale, **kwargs)
    elif dtype == 'val':
        return TrajectoryDataset(val_data_files, max_length=max_length, skip_frames=skip_frames,
                                 frame_shift=frame_shift, burn_in_steps=burn_in_steps, min_length=min_length, mean=mean, scale=scale, **kwargs)
