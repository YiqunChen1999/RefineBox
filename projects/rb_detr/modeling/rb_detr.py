
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor
from detectron2.structures import ImageList, Instances, Boxes
from detrex.layers.box_ops import box_cxcywh_to_xyxy
from refinebox.models.layers.mixin import RefinerMixin
from .detr import DETR
from refinebox.utils.utils import init_detrs_with_mixin


class RBDETR(RefinerMixin, DETR):
    def __init__(self, **kwargs):
        init_detrs_with_mixin(self, RefinerMixin, DETR, kwargs)

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

        # only use last level feature in DETR
        backbone_features: Dict[str, Tensor] = self.backbone(images.tensor)
        # features = self.backbone(images.tensor)[self.in_features[-1]]
        features = backbone_features[self.in_features[-1]]
        features = self.input_proj(features)
        img_masks = F.interpolate(img_masks[None],
                                  size=features.shape[-2:]).to(torch.bool)[0]
        pos_embed = self.position_embedding(img_masks)

        hidden_states, _ = self.transformer(features,
                                            img_masks,
                                            self.query_embed.weight,
                                            pos_embed)

        outputs_class: Tensor = self.class_embed(hidden_states)
        outputs_coord: Tensor = self.bbox_embed(hidden_states).sigmoid()
        output = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
        }
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class,
                                                       outputs_coord)
        return output, backbone_features

    def inference(self,
                  box_cls: Tensor,
                  box_pred: Tensor,
                  image_sizes: list[torch.Size]) -> list[Instances]:
        """
        Arguments:
            box_cls: tensor of shape (batch_size, num_queries, K). The tensor
                predicts the classification probability for each query.
            box_pred: tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes: the input image sizes

        Returns:
            results: a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # For each box we assign the best class or the second best
        # if the best on is `no_object`.
        scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (
                scores_per_image,
                labels_per_image,
                box_pred_per_image,
                image_size) in enumerate(zip(scores,
                                             labels,
                                             box_pred,
                                             image_sizes)):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1],
                                    scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results
