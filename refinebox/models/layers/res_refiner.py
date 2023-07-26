"""
Box refiner with res block.

Author:
    Yiqun Chen
"""


from typing import List, Dict
from copy import deepcopy

from torch import nn, Tensor
from detectron2.structures import Boxes
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.backbone.resnet import ResNet, BottleneckBlock
from detrex.layers.box_ops import box_cxcywh_to_xyxy
from refinebox.models.layers.base_refiner import BaseBboxRefiner


class ResBboxRefiner(BaseBboxRefiner):
    def __init__(self,
                 num_classes: int,
                 d_model: int,
                 roi_layer: ROIPooler,
                 num_reg_layers: int,
                 num_cls_layers: int = None,
                 cls_as_obj_score: bool = True,
                 res_block_cfg: dict = None) -> None:
        super().__init__(num_classes=num_classes,
                         d_model=d_model,
                         num_reg_layers=num_reg_layers,
                         num_cls_layers=num_cls_layers,
                         cls_as_obj_score=cls_as_obj_score)
        self.roi_layer = roi_layer
        default_res_block_cfg = dict(block_class=BottleneckBlock,
                                     num_blocks=3,
                                     stride_per_block=[1, 1, 1],
                                     in_channels=256,
                                     bottleneck_channels=1024,
                                     out_channels=256,
                                     num_groups=8,  # groups of 3x3 conv.
                                     norm='GN',
                                     stride_in_1x1=False)
        if res_block_cfg is not None:
            default_res_block_cfg.update(res_block_cfg)

        self.res_block = None
        if num_reg_layers is not None and num_reg_layers > 0:
            res_block = ResNet.make_stage(**default_res_block_cfg)
            self.res_block = nn.Sequential(*res_block)
        self.cls_block = None
        if num_cls_layers is not None and num_cls_layers > 0:
            res_block = ResNet.make_stage(**default_res_block_cfg)
            self.cls_block = nn.Sequential(*res_block)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self,
                features: List[Tensor],
                predictions: Dict,
                image_whwh: Tensor) -> Dict:
        if isinstance(features, dict):
            features = features.values()
        features = list(features)
        assert isinstance(features, list), \
               f'Expect features is instances of `list` or `tuple`, ' \
               f'but got type {type(features)}.'
        logits: Tensor = predictions['pred_logits']
        bboxes: Tensor = predictions['pred_boxes']
        unnorm_bboxes = bboxes * image_whwh.unsqueeze(1)
        unnorm_bboxes = box_cxcywh_to_xyxy(unnorm_bboxes)
        roi_boxes = [Boxes(b) for b in unnorm_bboxes]
        # (B*N, D, H, W)
        _roi_feats: Tensor = self.roi_layer(features, roi_boxes)

        pred_bboxes = bboxes
        if self.reg_module is not None:
            # (B*N, D, H, W) -> (B*N, D, 1, 1)
            roi_feats = self.res_block(_roi_feats)
            roi_feats = self.avg_pool(roi_feats)
            # (B*N, D)
            roi_feats = roi_feats.squeeze(-1).squeeze(-1)
            # (B*N, D) -> (B, N, D)
            roi_feats = roi_feats.unflatten(0, bboxes.shape[:2])
            reg_feature = roi_feats
            for reg_layer in self.reg_module:
                reg_feature = reg_layer(reg_feature)
            pred_bboxes_deltas = self.bboxes_deltas(reg_feature)
            pred_bboxes = self.apply_bboxes_deltas(pred_bboxes_deltas, bboxes)

        pred_logits = logits
        if self.cls_module is not None:
            # (B*N, D, H, W) -> (B*N, D, 1, 1)
            roi_feats = self.cls_block(_roi_feats)
            roi_feats = self.avg_pool(roi_feats)
            # (B*N, D)
            roi_feats = roi_feats.squeeze(-1).squeeze(-1)
            # (B*N, D) -> (B, N, D)
            roi_feats = roi_feats.unflatten(0, bboxes.shape[:2])
            cls_feature = roi_feats
            for cls_layer in self.cls_module:
                cls_feature = cls_layer(cls_feature)
            pred_logits_deltas = self.logits_deltas(cls_feature)
            pred_logits = self.apply_logits_deltas(pred_logits_deltas, logits)

        predictions = dict()
        predictions['pred_logits'] = pred_logits
        predictions['pred_boxes'] = pred_bboxes.clamp(min=0.0, max=1.0)
        return predictions
