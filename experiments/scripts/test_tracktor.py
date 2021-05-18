import os
import time
from os import path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader

import motmetrics as mm
mm.lap.default_solver = 'lap'

import torchvision
import yaml
from tqdm import tqdm
import sacred
from sacred import Experiment
from tracktor.frcnn_fpn import FRCNN_FPN, ProbFRCNN_FPN, FRCNN_FPN_Custom
from tracktor.config import get_output_dir
from tracktor.datasets.factory import Datasets
# from tracktor.datasets.factory_modified import Datasets
from tracktor.oracle_tracker import OracleTracker
from tracktor.tracker import Tracker
from tracktor.reid.resnet import resnet50
from tracktor.utils import interpolate, plot_sequence, get_mot_accum, evaluate_mot_accums

import cv2
from numpy import genfromtxt
import pickle

import motmetrics as mm

# from model.config import cfg as frcnn_cfg

ex = Experiment()

ex.add_config('experiments/cfgs/tracktor.yaml')

# hacky workaround to load the corresponding configs and not having to hardcode paths here
ex.add_config(ex.configurations[0]._conf['tracktor']['reid_config'])
ex.add_named_config('oracle', 'experiments/cfgs/oracle_tracktor.yaml')


@ex.automain
def main(tracktor, reid, _config, _log, _run):
    sacred.commands.print_config(_run)

    # set all seeds
    torch.manual_seed(tracktor['seed'])
    torch.cuda.manual_seed(tracktor['seed'])
    np.random.seed(tracktor['seed'])
    torch.backends.cudnn.deterministic = True

    output_dir = osp.join(get_output_dir(tracktor['module_name']), tracktor['name'])
    sacred_config = osp.join(output_dir, 'sacred_config.yaml')

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(_config, outfile, default_flow_style=False)

    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    _log.info("Initializing object detector.")

    # obj_detect = FRCNN_FPN(num_classes=2)
    # obj_detect.load_state_dict(torch.load(_config['tracktor']['obj_detect_model'],
    #                            map_location=lambda storage, loc: storage))

    ##NEW: fasterrcnn with regression variance
    # obj_detect = ProbFRCNN_FPN(num_classes=2)
    # obj_detect.load_state_dict(torch.load('output/faster_rcnn_fpn_training_mot_17/model_params_best_coco'))
    # obj_detect.load_state_dict(torch.load('output/faster_rcnn_fpn_training_mot_17/model_params_best_mot'))
    # obj_detect.load_state_dict(torch.load('model_params_best_mot'))
    obj_detect = FRCNN_FPN_Custom(num_classes=91, which_class=1)
    obj_detect.load_state_dict(
            torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).state_dict()
        )

    obj_detect.eval()
    obj_detect.cuda()

    # reid
    reid_network = resnet50(pretrained=False, **reid['cnn'])
    reid_network.load_state_dict(torch.load(tracktor['reid_weights'],
                                 map_location=lambda storage, loc: storage))
    reid_network.eval()
    reid_network.cuda()

    # tracktor
    if 'oracle' in tracktor:
        tracker = OracleTracker(obj_detect, reid_network, tracktor['tracker'], tracktor['oracle'])
    else:
        tracker = Tracker(obj_detect, reid_network, tracktor['tracker'])

    time_total = 0
    num_frames = 0
    mot_accums = []
    variances = []
    dataset = Datasets(tracktor['dataset'])

    count = 0

    # breakpoint()

    for i, seq in enumerate(dataset):

        if i > 0:
            continue

        tracker.reset()

        # folder = f'{seq}'[:-1]
        folder = f'{seq}'
        gt_file = osp.join('data', 'MOT17Labels', 'train', folder, 'gt', 'gt.txt')
        # gt_file = osp.join('data', 'MOT17Labels', 'test', folder, 'gt', 'gt.txt')
        
        # my_data = genfromtxt(gt_file, delimiter=',')
        # my_data = my_data[my_data[:, 7]==1, :]

        # breakpoint()

        # gt_file = osp.join('data', 'MOT20', 'train', folder, 'gt', 'gt.txt')
        # det_file = osp.join('data', 'MOT20', 'train', folder, 'det', 'det.txt')
        
        my_data = genfromtxt(gt_file, delimiter=',')
        my_data = my_data[np.isin(my_data[:, -2], [1,2,7]), :]
        # my_data = my_data[np.isin(my_data[:, -2], [3]), :]

        start = time.time()

        _log.info(f"Tracking: {seq}")

        data_loader = DataLoader(seq, batch_size=1, shuffle=False)
        # breakpoint()
        for j, frame in enumerate(tqdm(data_loader)):
            # if j <= 3:
            #     continue
            # if j > 40:
            #     break
            if len(seq) * tracktor['frame_split'][0] <= j <= len(seq) * tracktor['frame_split'][1]:
                with torch.no_grad():
                    frame_detection = my_data[my_data[:,0]==j+1, :]
                    # breakpoint()
                    tracker.step(frame, frame_detection, j, folder)
        # breakpoint()
        data = np.asarray(tracker.data)
        np.savetxt(f'test_results/{folder}_data.csv', data, delimiter=",")
        # breakpoint()
        results = tracker.get_results()

        time_total += time.time() - start

        _log.info(f"Tracks found: {len(results)}")
        _log.info(f"Runtime for {seq}: {time.time() - start :.2f} s.")

        if tracktor['interpolate']:
            results = interpolate(results)

        if seq.no_gt:
            _log.info(f"No GT data for evaluation available.")
        else:
            mot_accums.append(get_mot_accum(results, seq))
        
        # breakpoint()
        
        mh = mm.metrics.create()
        summary = mh.compute(mot_accums[-1], metrics=['num_frames', 'mota', 'motp'], name='acc')
        print(summary)
        
        breakpoint()

        _log.info(f"Writing predictions to: {output_dir}")
        seq.write_results(results, output_dir)

        if tracktor['write_images']:
            plot_sequence(results, seq, osp.join(output_dir, tracktor['dataset'], str(seq)))

    _log.info(f"Tracking runtime for all sequences (without evaluation or image writing): "
              f"{time_total:.2f} s for {num_frames} frames ({num_frames / time_total:.2f} Hz)")
    if mot_accums:
        evaluate_mot_accums(mot_accums, [str(s) for s in dataset if not s.no_gt], generate_overall=True)
