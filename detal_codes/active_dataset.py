import mmcv
import numpy as np
import random
from pycocotools.coco import COCO
import json

def get_X_L_0(cfg):
    # Load dataset anns
    anns = COCO(cfg.data.train.ann_file)
    training_set_size = len(anns.imgs.keys())
    anns_img_inds = np.sort(np.array(list(anns.imgs.keys())))
    X_all = anns.imgToAnns
    np.random.shuffle(anns_img_inds)
    X_L_0_size = int(training_set_size * cfg.X_L_0_ratio)
    X_L_0_inds = anns_img_inds[:X_L_0_size].copy()
    X_U_0_inds = anns_img_inds[X_L_0_size:].copy()

    return X_all, X_L_0_inds, X_U_0_inds