import os, sys
import time
import math
import numpy as np
import argparse

from PIL import Image, ImageDraw

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# from learning import *
# import data
# import util

import torch.nn as nn
import torchvision

from tracktor.probfasterrcnn import *

# from engine import *

from .utils import bbox_overlaps


class RegressorHead(nn.Module):
    """
    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        self.bbox_pred_logvar = nn.Linear(in_channels, num_classes * 4)
        

    def forward(self, x, training=False):
        if training:
            self.train()
        else:
            self.eval()
        bbox_deltas = self.bbox_pred(x)
        bbox_deltas_logvar = self.bbox_pred_logvar(x)

        return {'mu': bbox_deltas, 'logvar': bbox_deltas_logvar}


    def parameters(self):
        return self.bbox_pred_logvar.parameters()


    
def loss_nll_coco(x, y, mdl, reduction='mean', device=tc.device('cpu')):
    x = x.to(device)
    y_cls, y_reg = y['label_cls'].to(device), y['label_reg'].to(device)
    pred = mdl(x)
    n = x.shape[0]

    ##
    mu, logvar = pred['mu'].reshape(n, -1, 4)[tc.arange(0, n, dtype=tc.long), y_cls], pred['logvar'].reshape(n, -1, 4)[tc.arange(0, n, dtype=tc.long), y_cls]
    y = y_reg
    
    breakpoint()

    ## loss 
    #assert(len(mu.size()) == len(logvar.size()) == len(y.size()))
    #assert(all(~tc.isinf(logvar.exp())))
    
    sim_term = 0.5*(y - mu).div(logvar.exp().sqrt()).pow(2).sum(1)
    reg_term = 0.5*logvar.sum(1)
    const_term = 0.5*math.log(2*np.pi)*y.shape[1]
    loss_vec = sim_term + reg_term + const_term  
    # loss = reduce(loss_vec, reduction)
    # loss = tloss_vec

    #print(loss)
    breakpoint()
    return {'loss': loss}

def loss_nll_mot(mu, y, logvar, reduction='mean', device=tc.device('cpu')):
    
    sim_term = 0.5*(y - mu).div(logvar.exp().sqrt()).pow(2).sum(1)
    reg_term = 0.5*logvar.sum(1)
    const_term = 0.5*math.log(2*np.pi)*y.shape[1]
    loss_vec = sim_term + reg_term + const_term  
    loss = torch.mean(loss_vec)

    return loss


class COCORegLearner(RegressorHead):
    def __init__(self, mdl, params=None, name_postfix=None):
        super().__init__(mdl, params, name_postfix)
        self.loss_fn_train = loss_nll_coco
        self.loss_fn_val = loss_nll_coco
        self.loss_fn_test = loss_nll_coco
        

    def test(self, ld, mdl=None, loss_fn=None, ld_name=None, verbose=False):
        t_start = time.time()
        error, *_ = super().test(ld, mdl, loss_fn)
        
        if verbose:
            print('[test%s, %f secs.] nll = %f'%(
                ': %s'%(ld_name if ld_name else ''), time.time()-t_start, error))

        return error, 


def plot_boxes(fn, img, boxes, boxes_cons=[], linewidthrate=0.005, linewidthmin=3):
    size = max(img.width, img.height)
    linewidth = max(linewidthmin, round(size*linewidthrate))

    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle(box, outline='blue', width=linewidth)
    for box_cons in boxes_cons:
        draw.rectangle(box_cons, outline='red', width=linewidth)
    img.save(fn, 'png')
    
    

    
def parse_args():
    ## init a parser
    parser = argparse.ArgumentParser(description='learning')

    ## meta args
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--snapshot_root', type=str, default='snapshots')
    parser.add_argument('--cpu', action='store_true')
    #parser.add_argument('--calibrate', action='store_true')

    ## data args
    parser.add_argument('--data.batch_size', type=int, default=1024)
    #parser.add_argument('--data', type=str, required=True)
    #parser.add_argument('--data.n_labels', type=int)
    #parser.add_argument('--data.img_size', type=int, nargs=3)
    #parser.add_argument('--data.aug_src', type=str, nargs='*')
    #parser.add_argument('--data.aug_tar', type=str, nargs='*')


    # ## predset model args
    # parser.add_argument('--model_predset.eps', type=float, default=0.01)
    # parser.add_argument('--model_predset.alpha', type=float, default=0.01)
    # parser.add_argument('--model_predset.delta', type=float, default=1e-5)
    # parser.add_argument('--model_predset.n', type=int)

    ## train args
    parser.add_argument('--train.rerun', action='store_true')
    parser.add_argument('--train.load_final', action='store_true')
    parser.add_argument('--train.optimizer', type=str, default='SGD')
    parser.add_argument('--train.n_epochs', type=int, default=100)
    parser.add_argument('--train.lr', type=float, default=0.01)
    parser.add_argument('--train.momentum', type=float, default=0.9)
    parser.add_argument('--train.weight_decay', type=float, default=0.0)
    parser.add_argument('--train.lr_decay_epoch', type=int, default=20)
    parser.add_argument('--train.lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--train.val_period', type=int, default=1)

    ## calibration args
    parser.add_argument('--cal.rerun', action='store_true')
    parser.add_argument('--cal.load_final', action='store_true')
    parser.add_argument('--cal.optimizer', type=str, default='SGD')
    parser.add_argument('--cal.n_epochs', type=int, default=100)
    parser.add_argument('--cal.lr', type=float, default=0.01)
    parser.add_argument('--cal.momentum', type=float, default=0.9)
    parser.add_argument('--cal.weight_decay', type=float, default=0.0)
    parser.add_argument('--cal.lr_decay_epoch', type=int, default=20)
    parser.add_argument('--cal.lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--cal.val_period', type=int, default=1)    
    
    
    args = parser.parse_args()
    args = util.to_tree_namespace(args)


    ##TODO: generalize
    ## additional args
    args.train.device = tc.device('cpu') if args.cpu else tc.device('cuda:0')
    args.train.exp_name = args.exp_name
    args.train.snapshot_root = args.snapshot_root

    args.cal.device = tc.device('cpu') if args.cpu else tc.device('cuda:0')
    args.cal.exp_name = args.exp_name
    args.cal.snapshot_root = args.snapshot_root

    
    ## print args
    util.print_args(args)
    
    ## setup logger
    os.makedirs(os.path.join(args.snapshot_root, args.exp_name), exist_ok=True)
    sys.stdout = util.Logger(os.path.join(args.snapshot_root, args.exp_name, 'out'))
    
    
    return args    


if __name__ == '__main__':

    ## parameters
    args = parse_args()
    
    ## init loaders
    dsld = data.COCO_reg('data/coco_reg_wo_bg', args.data.batch_size)
    
    ## init a model
    mdl = RegressorHead(1024, 91)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    mdl.bbox_pred = model.roi_heads.box_predictor.bbox_pred
    
    ## train and test
    l = COCORegLearner(mdl, args.train, name_postfix='coco_reg_head')
    l.train(dsld.train, dsld.val, dsld.test)
    l.test(dsld.test, verbose=True)
    print()

    ## save the new model
    mdl_new = probfasterrcnn_resnet50_fpn(pretrained=True)
    mdl_new.roi_heads.box_predictor.bbox_pred_logvar = mdl.bbox_pred_logvar
    tc.save(mdl_new.state_dict(), 'model_params')
    
    mdl_new.eval()
    mdl_new.to(args.train.device)
    
    ## init coco dataset
    dsld = data.COCO('data/coco', 3, split_ratio={'train': None, 'val': 0.5, 'test': 0.5})
    
    ## eval
    #evaluate(mdl_new, dsld.test, device=args.train.device)

    ## plot bounding boxes
    device = args.train.device
    plot_root = 'plots/boxes'
    os.makedirs(plot_root, exist_ok=True)
    i = 0
    for image, targets in dsld.test:
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with tc.no_grad():
            outputs = mdl_new(image)

        for img, pred in zip(image, outputs):
            img, score_pred, box_pred, box_pred_logvar = img.cpu(), pred['scores'].cpu().detach(), pred['boxes'].cpu().detach(), pred['boxes_logvar'].cpu().detach()
            img_pil, score_pred, box_pred, box_pred_logvar = torchvision.transforms.ToPILImage()(img), score_pred.numpy(), box_pred.numpy(), box_pred_logvar.numpy()

            keep = score_pred > 0.9
            if all(keep == False):
                continue
            box_pred, box_pred_logvar = box_pred[keep], box_pred_logvar[keep]
            box_pred_cons = np.zeros_like(box_pred)
            box_pred_cons[:, :2] = box_pred[:, :2] - np.sqrt(np.exp(box_pred_logvar[:, :2]))
            box_pred_cons[:, 2:] = box_pred[:, 2:] + np.sqrt(np.exp(box_pred_logvar[:, 2:]))

            fn = os.path.join(plot_root, '%d.png'%(i))
            plot_boxes(fn, img_pil, boxes=box_pred, boxes_cons=box_pred_cons)
            i = i + 1
        

