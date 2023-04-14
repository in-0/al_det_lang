_base_ = [
    '../configs/_base_/models/faster_rcnn_r50_fpn.py',
    '../configs/_base_/datasets/coco_detection.py',
    '../configs/_base_/schedules/schedule_1x.py', '../configs/_base_/default_runtime.py'
]

cycles = 5
acquisition_function = 'random'

# Dataset configs
X_L_0_ratio = 0.05
buget_ratio = 0.05