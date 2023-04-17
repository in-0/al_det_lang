import mmcv
import numpy as np
import random
from pycocotools.coco import COCO
import json
import mmcv
import time
import copy

def get_X_L_0(cfg):
    cfg_l = copy.deepcopy(cfg)
    cfg_u = copy.deepcopy(cfg)

    tic = time.time()
    # Load dataset anns
    anns = COCO(cfg.data.train.ann_file)
    training_set_size = len(anns.imgs.keys())
    all_img_ids = np.sort(np.array(list(anns.imgs.keys())))
    np.random.shuffle(all_img_ids)
    X_L_0_size = int(training_set_size * cfg.X_L_0_ratio)
    X_L_0_list = all_img_ids[:X_L_0_size]
    X_U_0_list = all_img_ids[X_L_0_size:]

    # Create X_L and X_U annotation json file
    save_directory = f'{cfg.work_dir}/cycle_0'
    mmcv.mkdir_or_exist(save_directory)
    X_L_0_save_path = f'{save_directory}/X_L_list.json'
    X_U_0_save_path = f'{save_directory}/X_U_list.json'

    X_L_0_dataset, X_L_0_anns, X_L_0_imgs = dict(), [], []
    X_U_0_dataset, X_U_0_anns, X_U_0_imgs = dict(), [], []
    for key in X_L_0_list:
        if len(anns.imgToAnns[key]):
            X_L_0_anns.extend(anns.imgToAnns.pop(key))
        X_L_0_imgs.append(anns.imgs.pop(key))
    X_L_0_dataset['info'] = anns.dataset['info']
    X_L_0_dataset['licenses'] = anns.dataset['licenses']
    X_L_0_dataset['categories'] = anns.dataset['categories']
    X_L_0_dataset['annotations'] = X_L_0_anns
    X_L_0_dataset['images'] = X_L_0_imgs
    
    for key in X_U_0_list:
        if len(anns.imgToAnns[key]):
            X_U_0_anns.extend(anns.imgToAnns.pop(key))
        X_U_0_imgs.append(anns.imgs.pop(key))
    X_U_0_dataset['info'] = anns.dataset['info']
    X_U_0_dataset['licenses'] = anns.dataset['licenses']
    X_U_0_dataset['categories'] = anns.dataset['categories']
    X_U_0_dataset['annotations'] = X_U_0_anns
    X_U_0_dataset['images'] = X_U_0_imgs

    with open(X_L_0_save_path, 'w') as file_1:
        json.dump(X_L_0_dataset, file_1)
    with open(X_U_0_save_path, 'w') as file_2:
        json.dump(X_U_0_dataset, file_2)

    cfg_l.data.train.ann_file = X_L_0_save_path
    cfg_u.data.train.ann_file = X_U_0_save_path

    print(f'Done (t={time.time()-tic:.2f}s)')
    return cfg_l, cfg_u

def update_X_L(cfg_l, cfg_u, X_L, cycle):
    tic = time.time()
    anns_l = COCO(cfg_l.data.train.ann_file)
    anns_u = COCO(cfg_u.data.train.ann_file)

    X_U_dataset, X_U_anns, X_U_imgs = dict(), [], []
    X_L.sort()
    for key in X_L:
        ann_u = anns_u.imgToAnns.pop(key)
        img_u = anns_u.imgs.pop(key)
        if len(ann_u):
            anns_l.dataset['annotations'].extend(ann_u)
            X_U_anns.extend(ann_u)
        anns_l.dataset['images'].append(img_u)
        X_U_imgs.append(img_u)

    X_U_dataset['info'] = anns_u.dataset['info']
    X_U_dataset['licenses'] = anns_u.dataset['licenses']
    X_U_dataset['categories'] = anns_u.dataset['categories']
    X_U_dataset['annotations'] = X_U_anns
    X_U_dataset['images'] = X_U_imgs
    
    save_directory = f'{cfg_l.work_dir}/cycle_{cycle}'
    mmcv.mkdir_or_exist(save_directory)
    X_L_save_path = f'{save_directory}/X_L_list.json'
    X_U_save_path = f'{save_directory}/X_U_list.json'

    with open(X_L_save_path, 'w') as file_1:
        json.dump(anns_l.dataset, file_1)
    with open(X_U_save_path, 'w') as file_2:
        json.dump(X_U_dataset, file_2)

    cfg_l.data.train.ann_file = X_L_save_path
    cfg_u.data.train.ann_file = X_U_save_path

    print(f'{cycle}Done (t={time.time()-tic:.2f}s)')
    return cfg_l, cfg_u