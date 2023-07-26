
from typing import List, Dict
import os
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from detectron2.structures import ImageList
from detrex.utils.misc import inverse_sigmoid
from .dn_detr import DNDETR
from refinebox.models.layers.mixin import RefinerMixin
from refinebox.utils.utils import init_detrs_with_mixin


class RBDNDETR(RefinerMixin, DNDETR):
    def __init__(self, **kwargs):
        init_detrs_with_mixin(self, RefinerMixin, DNDETR, kwargs)

    def forward(self, batched_inputs: List[Dict]):
        images, images_whwh = RefinerMixin.preprocess_image(self,
                                                            batched_inputs)
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

        # only use last level feature as DAB-DETR
        backbone_features: Dict[str, Tensor] = self.backbone(images.tensor)
        # features = self.backbone(images.tensor)[self.in_features[-1]]
        features = backbone_features[self.in_features[-1]]
        features = self.input_proj(features)
        img_masks = F.interpolate(img_masks[None],
                                  size=features.shape[-2:]).to(torch.bool)[0]
        pos_embed = self.position_embedding(img_masks)

        # collect ground truth for denoising generation
        if self.training:
            gt_instances = [x["instances"].to(self.device)
                            for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            gt_labels_list = [t["labels"] for t in targets]
            gt_boxes_list = [t["boxes"] for t in targets]
        else:
            # set to None during inference
            targets = None

        # for vallina dn-detr, label queries in the matching part is encoded
        # as "no object" (the last class) in the label encoder.
        matching_label_query = self.denoising_generator.label_encoder(
            torch.tensor(self.num_classes).to(self.device)
        ).repeat(self.num_queries, 1)
        indicator_for_matching_part = \
            torch.zeros([self.num_queries, 1]).to(self.device)
        matching_label_query = torch.cat(
            [matching_label_query, indicator_for_matching_part], 1
        ).repeat(batch_size, 1, 1)
        matching_box_query = self.anchor_box_embed.weight.repeat(
            batch_size, 1, 1)

        if targets is None:
            # (num_queries, bs, embed_dim)
            input_label_query = matching_label_query.transpose(0, 1)
            # (num_queries, bs, 4)
            input_box_query = matching_box_query.transpose(0, 1)
            attn_mask = None
            denoising_groups = self.denoising_groups
            max_gt_num_per_image = 0
        else:
            # generate denoising queries and attention masks
            (
                noised_label_queries,
                noised_box_queries,
                attn_mask,
                denoising_groups,
                max_gt_num_per_image,
            ) = self.denoising_generator(gt_labels_list, gt_boxes_list)

            # concate dn queries and matching queries as input
            input_label_query = torch.cat(
                [noised_label_queries, matching_label_query], 1
            ).transpose(0, 1)
            input_box_query = torch.cat(
                [noised_box_queries, matching_box_query], 1).transpose(0, 1)

        hidden_states, reference_boxes = self.transformer(
            features,
            img_masks,
            input_box_query,
            pos_embed,
            target=input_label_query,
            attn_mask=[attn_mask, None],  # None mask for cross attention
        )

        # Calculate output coordinates and classes.
        reference_boxes = inverse_sigmoid(reference_boxes)
        anchor_box_offsets = self.bbox_embed(hidden_states)
        outputs_coord = (reference_boxes + anchor_box_offsets).sigmoid()
        outputs_class = self.class_embed(hidden_states)

        # denoising post process
        output = {
            "denoising_groups":
                torch.tensor(denoising_groups).to(self.device),
            "max_gt_num_per_image":
                torch.tensor(max_gt_num_per_image).to(self.device),
        }
        outputs_class, outputs_coord = self.dn_post_process(outputs_class,
                                                            outputs_coord,
                                                            output)

        output.update(
            {
                "pred_logits": outputs_class[-1],
                "pred_boxes": outputs_coord[-1],
            }
        )
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class,
                                                       outputs_coord)
        return output, backbone_features
