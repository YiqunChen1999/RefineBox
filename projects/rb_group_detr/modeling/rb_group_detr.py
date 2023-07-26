
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor
from detectron2.structures import ImageList
from detrex.utils.misc import inverse_sigmoid
from .group_detr import GroupDETR
from refinebox.models.layers.mixin import RefinerMixin
from refinebox.utils.utils import init_detrs_with_mixin


class RBGroupDETR(RefinerMixin, GroupDETR):
    def __init__(self, **kwargs):
        init_detrs_with_mixin(self, RefinerMixin, GroupDETR, kwargs)

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
                img_h, img_w = batched_inputs[img_id]["instances"].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)

        # only use last level feature in Conditional-DETR
        backbone_features: Dict[str, Tensor] = self.backbone(images.tensor)
        # features = self.backbone(images.tensor)[self.in_features[-1]]
        features = backbone_features[self.in_features[-1]]
        features = self.input_proj(features)
        img_masks = F.interpolate(img_masks[None],
                                  size=features.shape[-2:]).to(torch.bool)[0]
        pos_embed = self.position_embedding(img_masks)

        # training with multi-groups and inference in one group
        if self.training:
            query_embed_weight = self.query_embed.weight
        else:
            query_embed_weight = self.query_embed.weight[: self.num_queries]

        # hidden_states: transformer output hidden feature
        # reference: reference points in format (x, y)
        # with normalized coordinates in range of [0, 1].
        hidden_states, reference = self.transformer(
            features, img_masks, query_embed_weight, pos_embed
        )

        reference_before_sigmoid = inverse_sigmoid(reference)
        outputs_coords = []
        for lvl in range(hidden_states.shape[0]):
            tmp = self.bbox_embed(hidden_states[lvl])
            tmp[..., :2] += reference_before_sigmoid
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
        outputs_coord = torch.stack(outputs_coords)
        outputs_class = self.class_embed(hidden_states)

        output = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
        }
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class,
                                                       outputs_coord)

        return (output,
                backbone_features,
                img_masks,
                features,
                pos_embed)
