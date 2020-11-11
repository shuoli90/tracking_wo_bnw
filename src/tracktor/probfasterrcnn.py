import os, sys

import torchvision
import torch as tc
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.detection.roi_heads import *
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, AnchorGenerator, MultiScaleRoIAlign, RPNHead, RegionProposalNetwork, TwoMLPHead, GeneralizedRCNNTransform, load_state_dict_from_url, model_urls
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import resize_boxes

def get_conservative_box(mu, logvar):
    """
    box convention: xyxy
    """
    n = mu.shape[0]
    mu, logvar = mu.reshape(n, -1, 4), logvar.reshape(n, -1, 4)
    # one sigmal box
    sigma = logvar.exp().sqrt()
    xy_cons_topleft = mu[:, :, :2] - sigma[:, :, :2]
    xy_cons_botright = mu[:, :, 2:] + sigma[:, :, 2:]
    box_cons = tc.cat([xy_cons_topleft, xy_cons_botright], 2)
    box_cons = box_cons.reshape(n, -1)

    return box_cons
    

class ProbRoIHeads(RoIHeads):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
    def postprocess_detections(self,
                               class_logits,    # type: Tensor
                               box_regression,  # type: Tensor
                               proposals,       # type: List[Tensor]
                               image_shapes,     # type: List[Tuple[int, int]]
                               labels_gt, ##NEW
                               boxes_gt, ##NEW
                               ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]

        
        pred_boxes = box_regression['bbox_deltas'] ##NEW
        pred_boxes_cons = get_conservative_box(box_regression['bbox_deltas'], box_regression['bbox_deltas_logvar']) ##NEW
        gt_boxes = tc.cat(boxes_gt, 0)
        
        pred_boxes = self.box_coder.decode(pred_boxes, proposals) ##NEW
        pred_boxes_cons = self.box_coder.decode(pred_boxes_cons, proposals) ##NEW
        gt_boxes = self.box_coder.decode(gt_boxes, proposals) ##NEW
        gt_boxes = gt_boxes.expand((-1, pred_boxes.shape[1], -1))

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_boxes_cons_list = pred_boxes_cons.split(boxes_per_image, 0) ##NEW
        gt_boxes_list = gt_boxes.split(boxes_per_image, 0) ##NEW
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        ##NEW
        if labels_gt is None:
            labels_gt_list = [tc.ones(s.shape[0], dtype=tc.long)*(-1) for s in pred_scores_list]
        else:
            labels_gt_list = labels_gt
            #labels_gt = labels_gt if labels_gt is not None else tc.zeros(len(image_shapes), dtype=tc.long) ##NEW
        ## duplicate gt labels, like one hot vector
        labels_gt_list = [F.one_hot(l, s.shape[-1]).mul(l.unsqueeze(1)) for l, s in zip(labels_gt_list, pred_scores_list)]
        
        all_boxes = []
        all_boxes_cons = [] ##NEW
        all_scores = []
        all_labels = []
        all_labels_gt = [] ##NEW
        all_boxes_gt = [] ##NEW
        for boxes, boxes_cons, scores, image_shape, labels_gt, boxes_gt in zip(pred_boxes_list, pred_boxes_cons_list, pred_scores_list, image_shapes, labels_gt_list, gt_boxes_list): ##NEW

            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            boxes_cons = box_ops.clip_boxes_to_image(boxes_cons, image_shape) ##NEW
            boxes_gt = box_ops.clip_boxes_to_image(boxes_gt, image_shape) ##NEW

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            
            # remove predictions with the background label
            boxes = boxes[:, 1:]
            boxes_cons = boxes_cons[:, 1:] ##NEW
            boxes_gt = boxes_gt[:, 1:] ##NEW
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            labels_gt = labels_gt[:, 1:] ##NEW
            
            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            boxes_cons = boxes_cons.reshape(-1, 4) ##NEW
            boxes_gt = boxes_gt.reshape(-1, 4) ##NEW
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            labels_gt = labels_gt.reshape(-1) ##NEW
            
            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, boxes_cons, boxes_gt, scores, labels, labels_gt = boxes[inds], boxes_cons[inds], boxes_gt[inds], scores[inds], labels[inds], labels_gt[inds] ##NEW

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, boxes_cons, boxes_gt, scores, labels, labels_gt = boxes[keep], boxes_cons[keep], boxes_gt[keep], scores[keep], labels[keep], labels_gt[keep] ##NEW

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, boxes_cons, boxes_gt, scores, labels, labels_gt = boxes[keep], boxes_cons[keep], boxes_gt[keep], scores[keep], labels[keep], labels_gt[keep] ##NEW

            all_boxes.append(boxes)
            all_boxes_cons.append(boxes_cons) ##NEW
            all_scores.append(scores)
            all_labels.append(labels)
            all_labels_gt.append(labels_gt)
            all_boxes_gt.append(boxes_gt)

        return all_boxes, all_boxes_cons, all_scores, all_labels, all_labels_gt, all_boxes_gt ##NEW

    
    def forward(self,
                features,      # type: Dict[str, Tensor]
                proposals,     # type: List[Tensor]
                image_shapes,  # type: List[Tuple[int, int]]
                targets=None   # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            if targets is not None:
                proposals, matched_idxs, labels_gt, regression_targets = self.select_training_samples(proposals, targets) ##NEW
            else:
                labels = None
                regression_targets = None
                matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, boxes_cons, scores, labels, labels_gt, boxes_gt = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes, labels_gt, regression_targets) ##NEW
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "boxes_cons": boxes_cons[i], ##NEW
                        "boxes_gt": boxes_gt[i], ##NEW
                        "labels": labels[i],
                        "labels_gt": labels_gt[i], ##NEW
                        "scores": scores[i],
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                assert matched_idxs is not None
                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                mask_logits = torch.tensor(0)
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None
                assert mask_logits is not None

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals,
                    gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {
                    "loss_mask": rcnn_loss_mask
                }
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if self.keypoint_roi_pool is not None and self.keypoint_head is not None \
                and self.keypoint_predictor is not None:
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                assert matched_idxs is not None
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals,
                    gt_keypoints, pos_matched_idxs)
                loss_keypoint = {
                    "loss_keypoint": rcnn_loss_keypoint
                }
            else:
                assert keypoint_logits is not None
                assert keypoint_proposals is not None

                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps

            losses.update(loss_keypoint)

        return result, losses

    

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

    
class GeneralizedProbRCNNTransform(GeneralizedRCNNTransform):
    def __init__(self, min_size, max_size, image_mean, image_std):
        super().__init__(min_size, max_size, image_mean, image_std)

        
    def postprocess(self,
                    result,               # type: List[Dict[str, Tensor]]
                    image_shapes,         # type: List[Tuple[int, int]]
                    original_image_sizes  # type: List[Tuple[int, int]]
                    ):
        # type: (...) -> List[Dict[str, Tensor]]
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes, boxes_cons = pred["boxes"], pred['boxes_cons'] ##NEW
            boxes, boxes_cons = resize_boxes(boxes, im_s, o_im_s), resize_boxes(boxes_cons, im_s, o_im_s) ##NEW
            
            result[i]["boxes"] = boxes
            result[i]['boxes_cons'] = boxes_cons ##NEW
            
            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks
            if "keypoints" in pred:
                keypoints = pred["keypoints"]
                keypoints = resize_keypoints(keypoints, im_s, o_im_s)
                result[i]["keypoints"] = keypoints
        return result
        

class ProbFasterRCNN(GeneralizedRCNN):
    """
    Implements Faster R-CNN.
    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.
    The behavior of the model changes depending if it is in training or evaluation mode.
    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values of x
          between 0 and W and values of y between 0 and H
        - labels (Int64Tensor[N]): the class label for each ground-truth box
    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN.
    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values of x
          between 0 and W and values of y between 0 and H
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
    Example::
        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import FasterRCNN
        >>> from torchvision.models.detection.rpn import AnchorGenerator
        >>> # load a pre-trained model for classification and return
        >>> # only the features
        >>> backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        >>> # FasterRCNN needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280
        >>> # so we need to add it here
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the RPN generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
        >>>                                    aspect_ratios=((0.5, 1.0, 2.0),))
        >>>
        >>> # let's define what are the feature maps that we will
        >>> # use to perform the region of interest cropping, as well as
        >>> # the size of the crop after rescaling.
        >>> # if your backbone returns a Tensor, featmap_names is expected to
        >>> # be ['0']. More generally, the backbone should return an
        >>> # OrderedDict[Tensor], and in featmap_names you can choose which
        >>> # feature maps to use.
        >>> roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
        >>>                                                 output_size=7,
        >>>                                                 sampling_ratio=2)
        >>>
        >>> # put the pieces together inside a FasterRCNN model
        >>> model = FasterRCNN(backbone,
        >>>                    num_classes=2,
        >>>                    rpn_anchor_generator=anchor_generator,
        >>>                    box_roi_pool=roi_pooler)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """

    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = ProbFastRCNNPredictor( ##NEW
                representation_size,
                num_classes)

        roi_heads = ProbRoIHeads( ##NEW
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedProbRCNNTransform(min_size, max_size, image_mean, image_std)

        super(ProbFasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)

        
    def forward(self, images, targets=None):
        out = super().forward(images, targets)

        out_new = []
        if not self.training:
            ## agument logvar
            for pred in out:
                boxes, boxes_sigma = pred['boxes'], pred['boxes_cons']
                sigma = (boxes - boxes_sigma).abs()
                sigma_topleft, sigma_botright = sigma[:, :2], sigma[:, 2:]
                sigma_worst = tc.max(sigma[:, :2], sigma[:, 2:])
                sigma = tc.cat((sigma_worst, sigma_worst), 1)
                pred['boxes_logvar'] = sigma.pow(2).log()
                out_new.append(pred)
                
        return out_new


def probfasterrcnn_resnet50_fpn(pretrained=False, progress=True,
                            num_classes=91, pretrained_backbone=True, trainable_backbone_layers=3, **kwargs):
    """
    Constructs a Faster R-CNN model with a ResNet-50-FPN backbone.
    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.
    The behavior of the model changes depending if it is in training or evaluation mode.
    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with values of ``x``
          between ``0`` and ``W`` and values of ``y`` between ``0`` and ``H``
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box
    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN.
    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with values of ``x``
          between ``0`` and ``W`` and values of ``y`` between ``0`` and ``H``
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction
    Faster R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.
    Example::
        >>> model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        >>> # For training
        >>> images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
        >>> labels = torch.randint(1, 91, (4, 11))
        >>> images = list(image for image in images)
        >>> targets = []
        >>> for i in range(len(images)):
        >>>     d = {}
        >>>     d['boxes'] = boxes[i]
        >>>     d['labels'] = labels[i]
        >>>     targets.append(d)
        >>> output = model(images, targets)
        >>> # For inference
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
        >>>
        >>> # optionally, if you want to export the model to ONNX:
        >>> torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)
    Arguments:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        num_classes (int): number of output classes of the model (including the background)
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    assert trainable_backbone_layers <= 5 and trainable_backbone_layers >= 0
    # dont freeze any layers if pretrained model or backbone is not used
    if not (pretrained or pretrained_backbone):
        trainable_backbone_layers = 5
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone, trainable_layers=trainable_backbone_layers)
    model = ProbFasterRCNN(backbone, num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def ProbFasterRCNN_resnet50_fpn(pretrained=False, progress=True,
                            n_labels=91, pretrained_backbone=True, trainable_backbone_layers=3, **kwargs):
    assert trainable_backbone_layers <= 5 and trainable_backbone_layers >= 0
    # dont freeze any layers if pretrained model or backbone is not used
    if not (pretrained or pretrained_backbone):
        trainable_backbone_layers = 5
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone, trainable_layers=trainable_backbone_layers)
    model = ProbFasterRCNN(backbone, n_labels, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


class ObjectnessView(nn.Module):
    def __init__(self, mdl):
        super().__init__()
        self.mdl = mdl

        
    def forward(self, images, targets=None):
        pred_list = self.mdl.forward(images, targets)

        print("##TODO: visualized results")
        sys.exit()
        ph = tc.cat([tc.cat(((1 - p['scores']).unsqueeze(1), p['scores'].unsqueeze(1)), 1).unsqueeze(0) for p in pred_list], 0)
        
        pred = {'ph': ph, 'yh_top': ph.argmax(-1), 'ph_top': ph.max(-1)[0]}

        if targets is None:
            return pred
        else:
            y = [p['labels_gt'] for p in pred_list]
            return pred, y
        

        

class ClassifierView(nn.Module):
    def __init__(self, mdl):
        super().__init__()
        self.mdl = mdl

        
    def forward(self, images, targets=None):
        pred_list = self.mdl.forward(images, targets)
        x = tc.cat([pred['logits'] for pred in pred_list], 0)
        
        print(x.shape)
        raise NotImplementedError
        return {'fh': x, 'ph': F.softmax(x, -1), 'yh_top': x.argmax(-1), 'ph_top': F.softmax(x, -1).max(-1)[0]}

    
class RegressorView(nn.Module):
    def __init__(self, mdl):
        super().__init__()
        self.mdl = mdl

    def forward(self, images, targets=None):
        pred_list = self.mdl.forward(images, targets)
        mu_list = [pred['boxes'] for pred in pred_list]
        logvar_list = [pred['boxes_logvar'] for pred in pred_list]
    
