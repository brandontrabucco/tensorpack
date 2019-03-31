#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py

import argparse
import itertools
import numpy as np
import os
import shutil
import cv2
import six
assert six.PY3, "FasterRCNN requires Python 3!"
import tensorflow as tf
import tqdm

import tensorpack.utils.viz as tpviz
from tensorpack import *
from tensorpack.tfutils import optimizer
from tensorpack.tfutils.common import get_tf_version_tuple
from tensorpack.tfutils.summary import add_moving_summary

import FasterRCNN.model_frcnn as model_frcnn
import FasterRCNN.model_mrcnn as model_mrcnn
from FasterRCNN.basemodel import image_preprocess, resnet_c4_backbone, resnet_conv5, resnet_fpn_backbone
from FasterRCNN.dataset import DetectionDataset
from FasterRCNN.config import finalize_configs, config as cfg
from FasterRCNN.data import get_all_anchors, get_all_anchors_fpn, get_eval_dataflow, get_train_dataflow
from FasterRCNN.eval import DetectionResult, predict_image, multithread_predict_dataflow, EvalCallback
from FasterRCNN.model_box import RPNAnchors, clip_boxes, crop_and_resize, roi_align
from FasterRCNN.model_cascade import CascadeRCNNHead
from FasterRCNN.model_fpn import fpn_model, generate_fpn_proposals, multilevel_roi_align, multilevel_rpn_losses
from FasterRCNN.model_frcnn import BoxProposals, FastRCNNHead, fastrcnn_outputs, fastrcnn_predictions, sample_fast_rcnn_targets
from FasterRCNN.model_mrcnn import maskrcnn_loss, maskrcnn_upXconv_head
from FasterRCNN.model_rpn import generate_rpn_proposals, rpn_head, rpn_losses
from FasterRCNN.viz import draw_annotation, draw_final_outputs, draw_predictions, draw_proposal_recall

try:
    import horovod.tensorflow as hvd
except ImportError:
    pass


class DefaultArgs():
    def __init__(self):
        self.predict = ("D:/research/tensorpack/examples/FasterRCNN/image.jpg")
        self.load = ("D:/research/tensorpack/examples/FasterRCNN" + 
            "/ckpts/COCO-R101FPN-MaskRCNN-ScratchGN.npz")
        self.config = ['MODE_FPN=True', 
            'FPN.CASCADE=True', 
            'BACKBONE.RESNET_NUM_BLOCKS=[3,4,23,3]', 
            'FPN.NORM=GN', 
            'BACKBONE.NORM=GN', 
            'FPN.FRCNN_HEAD_FUNC=fastrcnn_4conv1fc_gn_head', 
            'FPN.MRCNN_HEAD_FUNC=maskrcnn_up4conv_gn_head', 
            'PREPROC.TRAIN_SHORT_EDGE_SIZE=[640,800]', 
            'TRAIN.LR_SCHEDULE=[1500000,1580000,1620000]', 
            'BACKBONE.FREEZE_AT=0']


def build_r101fpn_mask_rcnn_model(image):
    num_anchors = len(cfg.RPN.ANCHOR_RATIOS)
    c2345 = resnet_fpn_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCKS)
    p23456 = fpn_model('fpn', c2345)
    assert len(cfg.RPN.ANCHOR_SIZES) == len(cfg.FPN.ANCHOR_STRIDES)
    image_shape2d = tf.shape(image)[2:]
    all_anchors_fpn = get_all_anchors_fpn()
    multilevel_anchors = [RPNAnchors(all_anchors_fpn[i],
        tf.zeros([tf.shape(all_anchors_fpn[i])[0], 
            tf.shape(all_anchors_fpn[i])[1], num_anchors]),
        tf.zeros([tf.shape(all_anchors_fpn[i])[0], 
            tf.shape(all_anchors_fpn[i])[1], num_anchors, 4])) for i in range(len(
                all_anchors_fpn))]
    for i, stride in enumerate(cfg.FPN.ANCHOR_STRIDES):
        with tf.name_scope('FPN_slice_lvl{}'.format(i)):
            multilevel_anchors[i] = multilevel_anchors[i].narrow_to(p23456[i])
    rpn_outputs = [rpn_head('rpn', pi, cfg.FPN.NUM_CHANNEL, len(
        cfg.RPN.ANCHOR_RATIOS)) for pi in p23456]
    multilevel_label_logits = [k[0] for k in rpn_outputs]
    multilevel_box_logits = [k[1] for k in rpn_outputs]
    multilevel_pred_boxes = [anchor.decode_logits(logits) for anchor, logits in zip(
        multilevel_anchors, multilevel_box_logits)]
    proposal_boxes, proposal_scores = generate_fpn_proposals(
        multilevel_pred_boxes, multilevel_label_logits, image_shape2d)
    proposals = BoxProposals(proposal_boxes)
    fastrcnn_head_func = getattr(model_frcnn, cfg.FPN.FRCNN_HEAD_FUNC)
    def roi_func(boxes):
        return multilevel_roi_align(p23456[:4], boxes, 7)
    fastrcnn_head = CascadeRCNNHead(
        proposals, roi_func, fastrcnn_head_func,
        (None, None), image_shape2d, cfg.DATA.NUM_CLASS)
    decoded_boxes = fastrcnn_head.decoded_output_boxes()
    decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')
    label_scores = fastrcnn_head.output_scores(name='fastrcnn_all_scores')
    final_boxes, final_scores, final_labels = fastrcnn_predictions(
        decoded_boxes, label_scores, name_scope='output')
    roi_feature = multilevel_roi_align(p23456[:4], final_boxes, 14)
    # Cascade inference needs roi transform with refined boxes.
    maskrcnn_head_func = getattr(model_mrcnn, cfg.FPN.MRCNN_HEAD_FUNC)
    mask_logits = maskrcnn_head_func(
        'maskrcnn', roi_feature, cfg.DATA.NUM_CATEGORY)   # #fg x #cat x 28 x 28
    indices = tf.stack([tf.range(tf.size(final_labels)), tf.cast(final_labels, tf.int32) - 1], axis=1)
    final_mask_logits = tf.gather_nd(mask_logits, indices)   # #resultx28x28
    final_mask = tf.sigmoid(final_mask_logits, name='output/masks')
    # Special computation for an image captioning model
    roi_feature_large = multilevel_roi_align(p23456[:4], final_boxes, 28)
    roi_feature_c2345 = [roi_align(x, final_boxes, 28) for x in c2345]
    roi_feature_p2345 = [roi_align(x, final_boxes, 28) for x in p23456[:4]]
    region_features_1 = tf.reduce_mean(roi_feature_c2345[-1], [2, 3])
    region_features_2 = tf.reduce_mean(roi_feature_large, [2, 3])
    region_features_3 = tf.concat([tf.reduce_mean(x, [2, 3]) for x in roi_feature_c2345], 1)
    region_features_4 = tf.concat([tf.reduce_mean(x, [2, 3]) for x in roi_feature_p2345], 1)
    roi_masks = tf.expand_dims(final_mask, 1)
    masks_sum = tf.reduce_sum(roi_masks, [2, 3])
    mask_features_1 = tf.reduce_sum(roi_feature_c2345[-1] * roi_masks, [2, 3]) / masks_sum
    mask_features_2 = tf.reduce_sum(roi_feature_large * roi_masks, [2, 3]) / masks_sum
    mask_features_3 = tf.concat([tf.reduce_sum(x * roi_masks, [2, 3]) for x in roi_feature_c2345], 1) / masks_sum
    mask_features_4 = tf.concat([tf.reduce_sum(x * roi_masks, [2, 3]) for x in roi_feature_p2345], 1) / masks_sum
    return { "boxes": final_boxes, "masks": final_mask, 
        "scores": final_scores, "labels": final_labels, 
        #"roi_feature_c2345": roi_feature_c2345, 
        #"roi_feature_p2345": roi_feature_p2345,
        #"c2345": c2345, 
        #"p2345": p23456[:4],
        "region_features_1": region_features_1,
        "region_features_2": region_features_2,
        "region_features_3": region_features_3,
        "region_features_4": region_features_4,
        "mask_features_1": mask_features_1, 
        "mask_features_2": mask_features_2, 
        "mask_features_3": mask_features_3, 
        "mask_features_4": mask_features_4 }


def create_r101fpn_mask_rcnn_model_function():
    args = DefaultArgs()
    if args.config:
        cfg.update_args(args.config)
    cfg.DATA.NUM_CATEGORY = 80 # Number of MSCOCO classes without background
    finalize_configs(is_training=False)
    image = tf.placeholder(tf.float32, (None, 3, None, None), 'image')
    results = build_r101fpn_mask_rcnn_model(image)
    loader = get_model_loader(args.load)
    loader._setup_graph()
    sess = tf.Session()
    loader._run_init(sess)
    def run_function(fetch, np_image):
        return sess.run(fetch, feed_dict={image: np_image})
    return results, run_function


def create_r101fpn_mask_rcnn_model_graph(image, sess):
    args = DefaultArgs()
    if args.config:
        cfg.update_args(args.config)
    cfg.DATA.NUM_CATEGORY = 80 # Number of MSCOCO classes without background
    finalize_configs(is_training=False)
    results = build_r101fpn_mask_rcnn_model(image)
    loader = get_model_loader(args.load)
    loader._setup_graph()
    loader._run_init(sess)
    return results


if __name__ == '__main__':
    results, run_function = create_r101fpn_mask_rcnn_model_function()
    img = cv2.imread(DefaultArgs().predict, cv2.IMREAD_COLOR)
    r = run_function(
        [results["region_features_1"],
            results["region_features_2"],
            results["region_features_3"],
            results["region_features_4"],
            results["mask_features_1"],
            results["mask_features_2"],
            results["mask_features_3"],
            results["mask_features_4"],
        ], 
        np.transpose(img[np.newaxis, ...], [0, 3, 1, 2]))
    for x in r:
        print(x.shape)
    print("Finished building model!")
