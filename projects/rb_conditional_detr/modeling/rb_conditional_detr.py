
import logging
from typing import List, Dict
import torch
from torch import Tensor
import torch.nn.functional as F

from detectron2.structures import ImageList
from detrex.utils import inverse_sigmoid
from .conditional_detr import ConditionalDETR
from refinebox.models.layers.mixin import RefinerMixin
from refinebox.utils.utils import init_detrs_with_mixin


logger = logging.getLogger(__name__)


class RBConditionalDETR(RefinerMixin, ConditionalDETR):
    def __init__(self, **kwargs):
        init_detrs_with_mixin(self, RefinerMixin, ConditionalDETR, kwargs)

    def forward(self, batched_inputs):
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
        """
        Forward function of `DAB-DETR` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each
                instance dict must consists of:
                    - dict["image"] (torch.Tensor): The unnormalized image
                        tensor.
                    - dict["height"] (int): The original image height.
                    - dict["width"] (int): The original image width.
                    - dict["instance"] (detectron2.structures.Instances):
                        Image meta informations and ground truth boxes and
                        labels during training. Please refer to
                        https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                        for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all
                    queries. Shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all
                    queries in format ``(x, y, w, h)``. These values are
                    normalized in [0, 1] relative to the size of each
                    individual image (disregarding possible padding). See
                    PostProcess for information on how to retrieve the
                    unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary
                    losses are activated. It is a list of dictionnaries
                    containing the two above keys for each decoder layer.
        """
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
        img_masks = F.interpolate(
            img_masks[None], size=features.shape[-2:]).to(torch.bool)[0]
        pos_embed = self.position_embedding(img_masks)

        # hidden_states: transformer output hidden feature
        # reference: reference points in format (x, y) with normalized
        # coordinates in range of [0, 1].
        hidden_states, reference = self.transformer(
            features, img_masks, self.query_embed.weight, pos_embed
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

        return output, backbone_features
