from collections import OrderedDict

import torch
import torch as tc
import torch.nn.functional as F

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import resize_boxes

from .probfasterrcnn import ProbFasterRCNN, get_conservative_box

class FRCNN_FPN(FasterRCNN):

    def __init__(self, num_classes):
        backbone = resnet_fpn_backbone('resnet50', False)
        super(FRCNN_FPN, self).__init__(backbone, num_classes)
        # these values are cached to allow for feature reuse
        self.original_image_sizes = None
        self.preprocessed_images = None
        self.features = None

    def detect(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)

        detections = self(img)[0]

        return detections['boxes'].detach(), detections['scores'].detach()

    def predict_boxes(self, boxes):
        device = list(self.parameters())[0].device
        boxes = boxes.to(device)

        boxes = resize_boxes(boxes, self.original_image_sizes[0], self.preprocessed_images.image_sizes[0])
        proposals = [boxes]

        box_features = self.roi_heads.box_roi_pool(self.features, proposals, self.preprocessed_images.image_sizes)
        box_features = self.roi_heads.box_head(box_features)
        class_logits, box_regression = self.roi_heads.box_predictor(box_features)

        pred_boxes = self.roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)

        # score_thresh = self.roi_heads.score_thresh
        # nms_thresh = self.roi_heads.nms_thresh

        # self.roi_heads.score_thresh = self.roi_heads.nms_thresh = 1.0
        # self.roi_heads.score_thresh = 0.0
        # self.roi_heads.nms_thresh = 1.0
        # detections, detector_losses = self.roi_heads(
        #     features, [boxes.squeeze(dim=0)], images.image_sizes, targets)

        # self.roi_heads.score_thresh = score_thresh
        # self.roi_heads.nms_thresh = nms_thresh

        # detections = self.transform.postprocess(
        #     detections, images.image_sizes, original_image_sizes)

        # detections = detections[0]
        # return detections['boxes'].detach().cpu(), detections['scores'].detach().cpu()

        pred_boxes = pred_boxes[:, 1:].squeeze(dim=1).detach()
        pred_boxes = resize_boxes(pred_boxes, self.preprocessed_images.image_sizes[0], self.original_image_sizes[0])
        pred_scores = pred_scores[:, 1:].squeeze(dim=1).detach()
        return pred_boxes, pred_scores

    def load_image(self, images):
        device = list(self.parameters())[0].device
        images = images.to(device)

        self.original_image_sizes = [img.shape[-2:] for img in images]

        preprocessed_images, _ = self.transform(images, None)
        self.preprocessed_images = preprocessed_images

        self.features = self.backbone(preprocessed_images.tensors)
        if isinstance(self.features, torch.Tensor):
            self.features = OrderedDict([(0, self.features)])


##
## probablistic version
##
class ProbFRCNN_FPN(ProbFasterRCNN):

    def __init__(self, num_classes):
        assert(num_classes == 2)
        print('[ProbFRCNN_FPN] initialized')
        
        backbone = resnet_fpn_backbone('resnet50', False)
        super(ProbFRCNN_FPN, self).__init__(backbone, 91)
        # these values are cached to allow for feature reuse
        self.original_image_sizes = None
        self.preprocessed_images = None
        self.features = None

    def detect(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)

        detections = self(img)[0]

        return detections['boxes'].detach(), detections['scores'].detach()

    def predict_boxes(self, boxes):
        device = list(self.parameters())[0].device
        boxes = boxes.to(device)

        boxes = resize_boxes(boxes, self.original_image_sizes[0], self.preprocessed_images.image_sizes[0])
        proposals = [boxes]

        box_features = self.roi_heads.box_roi_pool(self.features, proposals, self.preprocessed_images.image_sizes)
        box_features = self.roi_heads.box_head(box_features)
        class_logits, box_regression = self.roi_heads.box_predictor(box_features)

        pred_boxes = box_regression['bbox_deltas']
        pred_boxes_cons = get_conservative_box(box_regression['bbox_deltas'], box_regression['bbox_deltas_logvar'])
        
        pred_boxes = self.roi_heads.box_coder.decode(pred_boxes, proposals)
        pred_boxes_cons = self.roi_heads.box_coder.decode(pred_boxes_cons, proposals)
        
        pred_scores = F.softmax(class_logits, -1)

        pred_boxes = pred_boxes[:, 1, :].squeeze(dim=1).detach() ##TODO: pick-up person, idx=1
        pred_boxes = resize_boxes(pred_boxes, self.preprocessed_images.image_sizes[0], self.original_image_sizes[0])
        
        pred_boxes_cons = pred_boxes_cons[:, 1, :].squeeze(dim=1).detach() ##TODO: pick-up person, idx=1
        pred_boxes = resize_boxes(pred_boxes_cons, self.preprocessed_images.image_sizes[0], self.original_image_sizes[0])

        pred_scores = pred_scores[:, 1].detach() ##TODO: pick-up person, idx=1

        ## compute logvar
        boxes, boxes_sigma = pred_boxes, pred_boxes_cons
        sigma = (boxes - boxes_sigma).abs()
        sigma_topleft, sigma_botright = sigma[:, :2], sigma[:, 2:]
        sigma_worst = tc.max(sigma[:, :2], sigma[:, 2:])
        sigma = tc.cat((sigma_worst, sigma_worst), 1)
        pred_boxes_logvar = sigma.pow(2).log()
                
        return pred_boxes, pred_scores, pred_boxes_logvar

    def load_image(self, images):
        device = list(self.parameters())[0].device
        images = images.to(device)

        self.original_image_sizes = [img.shape[-2:] for img in images]

        preprocessed_images, _ = self.transform(images, None)
        self.preprocessed_images = preprocessed_images

        self.features = self.backbone(preprocessed_images.tensors)
        if isinstance(self.features, torch.Tensor):
            self.features = OrderedDict([(0, self.features)])
