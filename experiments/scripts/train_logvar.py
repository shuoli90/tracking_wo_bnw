import os
import time
from os import path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import motmetrics as mm
mm.lap.default_solver = 'lap'

import torchvision
import yaml
from tqdm import tqdm
import sacred
from sacred import Experiment
from tracktor.frcnn_fpn import FRCNN_FPN, ProbFRCNN_FPN
from tracktor.config import get_output_dir
from tracktor.datasets.factory import Datasets
from tracktor.oracle_tracker import OracleTracker
from tracktor.tracker import Tracker
from tracktor.reid.resnet import resnet50
from tracktor.utils import interpolate, plot_sequence, get_mot_accum, evaluate_mot_accums
from tracktor.finetune import RegressorHead, COCORegLearner, loss_nll_mot
import torchvision
from tracktor.datasets.mot_sequence import *

from tracktor.probfasterrcnn import *

import cv2
from numpy import genfromtxt
# from model.config import cfg as frcnn_cfg

ex = Experiment()

ex.add_config('experiments/cfgs/tracktor.yaml')

# hacky workaround to load the corresponding configs and not having to hardcode paths here
ex.add_config(ex.configurations[0]._conf['tracktor']['reid_config'])
ex.add_named_config('oracle', 'experiments/cfgs/oracle_tracktor.yaml')

dataset = Datasets('mot17_train_FRCNN17')

obj_detect = ProbFRCNN_FPN(num_classes=2)
obj_detect.load_state_dict(torch.load('output/faster_rcnn_fpn_training_mot_17/model_params_best_coco'))

breakpoint()

for seq in dataset:

    folder = f'{seq}'
    gt_file = osp.join('data', 'MOT17Labels', 'train', folder, 'gt', 'gt.txt')
    
    my_data = genfromtxt(gt_file, delimiter=',')
    my_data = my_data[my_data[:, 7]==1, :]

    data_loader = DataLoader(seq, batch_size=1, shuffle=False)
    for i, frame in enumerate(tqdm(data_loader)):
        inputs = frame['img']

        y = tc.tensor(my_data[my_data[:,0]==i+1, :])
        targets = {}
        targets['labels'] = y[:, 7].long()
        y = y[:, 2:6]
        y[:, 2] += y[:, 0]
        y[:, 3] += y[:, 1]
        targets["boxes"] = y
        pred = obj_detect(inputs, [targets])
        breakpoint()