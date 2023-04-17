import numpy as np
import mmcv
from pycocotools.coco import COCO

def acquire_random(cfg, cfg_u):
    anns_u = COCO(cfg_u.data.train.ann_file)
    anns = COCO(cfg.data.train.ann_file)
    budget_size = int(len(anns.imgs.keys()) * cfg.budget_ratio)
    u_img_ids = np.sort(np.array(list(anns_u.imgs.keys())))
    np.random.shuffle(u_img_ids)
    X_L = u_img_ids[:budget_size]
    return X_L