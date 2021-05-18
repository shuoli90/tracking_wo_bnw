# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
from os import path as osp
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import os, sys

import torchvision
import torch as tc
import torch.nn as nn
import torch.nn.functional as F

from sacred import Experiment
from tracktor.frcnn_fpn import FRCNN_FPN, ProbFRCNN_FPN
from tracktor.config import get_output_dir
from tracktor.datasets.factory import Datasets
from tracktor.oracle_tracker import OracleTracker
from tracktor.tracker import Tracker
from tracktor.reid.resnet import resnet50
from tracktor.utils import interpolate, plot_sequence, get_mot_accum, evaluate_mot_accums
from tracktor.finetune import RegressorHead, COCORegLearner, loss_nll_mot

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

# from engine import train_one_epoch, evaluate
# import utils
# import transforms as T

class ProbFastRCNNPredictor(FastRCNNPredictor):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.
    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__(in_channels, num_classes)
        self.bbox_pred_logvar = nn.Linear(in_channels, num_classes * 4)
        

    def forward(self, x):
        scores, bbox_deltas = super().forward(x)
        bbox_delta_logvar = self.bbox_pred_logvar(x)
        return scores, {'bbox_deltas': bbox_deltas, 'bbox_deltas_logvar': bbox_delta_logvar}

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    # model.roi_heads.box_predictor = ProbFastRCNNPredictor(in_features, num_classes)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations

    # move model to the right device
    # model.to(device)
    # model.eval()
    train = True

    ex = Experiment()

    ex.add_config('experiments/cfgs/tracktor.yaml')

    # hacky workaround to load the corresponding configs and not having to hardcode paths here
    ex.add_config(ex.configurations[0]._conf['tracktor']['reid_config'])
    ex.add_named_config('oracle', 'experiments/cfgs/oracle_tracktor.yaml')

    dataset = Datasets('mot17_train_FRCNN17')

    if train:
        model = ProbFRCNN_FPN(num_classes=2)
        model.load_state_dict(torch.load('output/faster_rcnn_fpn_training_mot_17/model_params_best_coco'))
        model.to(device)

        for param in model.backbone.parameters():
            param.requires_grad = False
    
        for param in model.rpn.parameters():
            param.requires_grad = False
        
        for param in model.roi_heads.box_head.parameters():
            param.requires_grad = False
        
        for param in model.roi_heads.box_predictor.cls_score.parameters():
            param.requires_grad = False
        
        for param in model.roi_heads.box_predictor.bbox_pred.parameters():
            param.requires_grad = False

        learning_rate = 1e-4
        optimizer = torch.optim.Adam(model.roi_heads.box_predictor.bbox_pred_logvar.parameters(), lr=learning_rate)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,
                                                    gamma=0.1)

        iterations = 10

        for i in range(iterations):
        
            for seq in dataset:

                folder = f'{seq}'
                gt_file = osp.join('data', 'MOT17Labels', 'train', folder, 'gt', 'gt.txt')
                
                my_data = genfromtxt(gt_file, delimiter=',')
                my_data = my_data[my_data[:, 7]==1, :]
                
                bs = 1

                data_loader = DataLoader(seq, batch_size=bs, shuffle=False)
                for i, frame in enumerate(tqdm(data_loader)):
                    inputs = frame['img'].to(device)
                    y = frame['gt']

                    targets = []

                    # boxes = []
                    for i in range(bs):
                        target = {}
                        boxes = []
                        for key in y.keys():
                            boxes.append(y[key][i,:])
                        boxes = tc.stack(boxes, dim=0)
                        # labels.append(label)
                        target['labels'] = tc.ones(boxes.shape[0]).to(device).long()
                        target['boxes'] = boxes.to(device)
                        targets.append(target)

                    loss = model(inputs, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # breakpoint()
                tc.save(model.state_dict(), 'model_params_best_mot')
            
            print('loss', loss)
    else:
        model = ProbFRCNN_FPN(num_classes=2)
        model.load_state_dict(torch.load('model_params_best_mot'))
        model.eval()

        for seq in dataset:

            folder = f'{seq}'
            gt_file = osp.join('data', 'MOT17Labels', 'train', folder, 'gt', 'gt.txt')
            
            my_data = genfromtxt(gt_file, delimiter=',')
            my_data = my_data[my_data[:, 7]==1, :]
            
            bs = 1

            data_loader = DataLoader(seq, batch_size=bs, shuffle=False)
            for i, frame in enumerate(tqdm(data_loader)):
                inputs = frame['img'].to(device)
                y = frame['gt']

                # targets = []

                # # boxes = []
                # for i in range(bs):
                #     target = {}
                #     boxes = []
                #     for key in y.keys():
                #         boxes.append(y[key][i,:])
                #     target['labels'] = tc.ones(boxes.shape[0]).to(device).long()
                #     target['boxes'] = boxes.to(device)
                #     targets.append(target)

                pred = model(inputs)
                breakpoint()
    
if __name__ == "__main__":
    main()
