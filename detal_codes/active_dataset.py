import mmcv
import numpy as np
import json

def get_X_L_0(cfg):
    # Load dataset anns
    with open(cfg.data.train.dataset.ann_file) as f:
        anns = json.load(f)