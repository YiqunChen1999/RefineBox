"""
RefineBox for DAB-DETR.

Author:
    Yiqun Chen
"""


import logging
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor
from detectron2.structures import ImageList
from detrex.utils.misc import inverse_sigmoid
from refinebox.models.layers.mixin import RefinerMixin
from .dab_detr import DABDETR
from refinebox.utils.utils import init_detrs_with_mixin


logger = logging.getLogger(__name__)


class RBDABDETR(RefinerMixin, DABDETR):
    def __init__(self, **kwargs):
        init_detrs_with_mixin(self, RefinerMixin, DABDETR, kwargs)

    def forward(self, batched_inputs: Dict) -> Dict:
        images, images_whwh = RefinerMixin.preprocess_image(
            self, batched_inputs)
        detector_outputs, backbone_features, *_ = \
            self.forward_detector(batched_inputs, images)
        return self.forward_refinement(
            batched_inputs=batched_inputs,
            detector_outputs=detector_outputs,
            images_whwh=images_whwh,
            image_sizes=images.image_sizes,
            backbone_features=backbone_features)

    def forward_detector(self, batched_inputs: List[Dict], images: ImageList):
        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = \
                    batched_inputs[img_id]["instances"].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)

        # only use last level feature in DAB-DETR
        backbone_features: Dict[str, Tensor] = self.backbone(images.tensor)
        # features = self.backbone(images.tensor)[self.in_features[-1]]
        features = backbone_features[self.in_features[-1]]
        features = self.input_proj(features)
        img_masks = F.interpolate(
            img_masks[None], size=features.shape[-2:]).to(torch.bool)[0]
        pos_embed = self.position_embedding(img_masks)

        # dynamic anchor boxes
        dynamic_anchor_boxes = self.anchor_box_embed.weight

        # hidden_states: transformer output hidden feature
        # reference_boxes: the refined dynamic anchor boxes in format
        # (x, y, w, h) with normalized coordinates in range of [0, 1].
        hidden_states, reference_boxes = self.transformer(
            features, img_masks, dynamic_anchor_boxes, pos_embed
        )
        # hidden_states, reference_boxes, memory = self.forward_transformer(
        #     features, img_masks, dynamic_anchor_boxes, pos_embed
        # )

        # Calculate output coordinates and classes.
        reference_boxes = inverse_sigmoid(reference_boxes)
        anchor_box_offsets = self.bbox_embed(hidden_states)
        outputs_coord = (reference_boxes + anchor_box_offsets).sigmoid()
        outputs_class = self.class_embed(hidden_states)

        output = {
            "pred_logits": outputs_class[self.predict_from_decoder_layer],
            "pred_boxes": outputs_coord[self.predict_from_decoder_layer]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class,
                                                       outputs_coord)
        return output, backbone_features
