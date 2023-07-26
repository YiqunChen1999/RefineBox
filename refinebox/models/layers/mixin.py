

import os
import logging
from copy import deepcopy
from typing import List, Dict, Union, Mapping, Any

import torch
from torch import nn, Tensor

from detectron2.modeling import detector_postprocess
from detectron2.structures import Instances, ImageList, Boxes
from detrex.utils import get_world_size, is_dist_avail_and_initialized
from detrex.layers.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.modeling import SetCriterion

from refinebox.models.layers.base_refiner import BaseBboxRefiner
from refinebox.models.layers.bboxes_filter import TopKBboxesFilter
from refinebox.utils.utils import (extract_positive_samples,
                                   update_positive_samples)


logger = logging.getLogger(__name__)


class RefinerMixin(nn.Module):
    def __init__(self,
                 refine_steps: int,
                 channel_mapper: nn.Module,
                 bboxes_refiner: BaseBboxRefiner,
                 bboxes_filter: TopKBboxesFilter,
                 refine_only_matched: bool,
                 predict_from_decoder_layer: int = -1,
                 visualize: bool = False,
                 replace_bboxes: bool = False,
                 replace_labels: bool = False,
                 output_dir: str | os.PathLike = 'output/debug'):
        self.criterion: SetCriterion
        self.num_classes: int
        self.device: torch.device
        self.refine_steps = refine_steps
        self.channel_mapper = channel_mapper
        self.bboxes_refiner = bboxes_refiner
        self.refine_only_matched = refine_only_matched
        self.bboxes_filter = bboxes_filter
        self.select_box_nums_for_evaluation: int
        self.predict_from_decoder_layer = predict_from_decoder_layer
        self.visualize = visualize
        if replace_bboxes or replace_labels:
            logger.warn(
                'Never set `replace_bboxes` and `replace_labels` as True '
                'when reporting results. '
                'Make sure you know what you are doing if you set '
                'them as True.')
        self.replace_bboxes = replace_bboxes
        self.replace_labels = replace_labels
        self.output_dir = output_dir
        self._trainable_modules = [channel_mapper,
                                   bboxes_refiner,
                                   bboxes_filter]
        self.freeze_parameters()

    def state_dict(self,
                   *args,
                   destination=None,
                   prefix='',
                   keep_vars=False) -> dict[str, Tensor]:
        # Only save parameters of RefineBox modules.
        KEYS = ('bboxes_refiner', 'bboxes_filter', 'channel_mapper')
        state_dict = super().state_dict(*args,
                                        destination=destination,
                                        prefix=prefix,
                                        keep_vars=keep_vars)
        state_dict_keys = list(state_dict.keys())
        for key in state_dict_keys:
            if not any([k in key for k in KEYS]):
                state_dict.pop(key)
        return state_dict

    def freeze_parameters(self):
        freeze_detector = getattr(self, 'freeze_detector', True)
        if not freeze_detector:
            logger.warn('YOU CHOOSE TO TRAIN THE DETECTOR.')
            return
        logger.info('Freezing parameters of detector...')
        if len(self._trainable_modules) == 0:
            raise RuntimeError('No trainable modules.')
        for name, module in self.named_children():
            if module in self._trainable_modules:
                continue
            module.requires_grad_(False)

    def forward_refinement(
            self,
            batched_inputs: list[dict],
            detector_outputs: dict[str, Tensor],
            images_whwh: Tensor,
            image_sizes: Tensor,
            backbone_features: dict[str, Tensor],
            **kwargs) -> Union[dict[str, Tensor], list[dict]]:
        """
        `forward_refinement` is a part of `forward`.
        """
        targets = None
        if len(kwargs):
            logger.warn(f'The following kwargs for refinement are not used: '
                        f'{list(kwargs.keys())}')

        if self.training:
            targets = self._get_targets(batched_inputs)
        features_for_refinement = backbone_features

        refiner_outputs = self.refine_bbox(
            features_for_refinement, detector_outputs, images_whwh,
            targets=targets)

        if self.training:
            loss_dict = self.calc_refine_loss(
                detector_outputs=detector_outputs,
                refiner_outputs=refiner_outputs,
                targets=targets)

            freeze_detector = getattr(self, 'freeze_detector', True)
            if not freeze_detector:
                det_loss_dict: dict[str, Tensor]
                det_loss_dict = self.criterion(detector_outputs, targets)
                weight_dict = self.criterion.weight_dict
                for k in det_loss_dict.keys():
                    if k in weight_dict:
                        det_loss_dict[k] *= weight_dict[k]
                loss_dict.update(det_loss_dict)

            return loss_dict
        else:
            if self.refine_steps > 0:
                detector_outputs = refiner_outputs[-1]
            detector_outputs = self._replace_annotations(batched_inputs,
                                                         detector_outputs)
            return self._process_results(batched_inputs,
                                         detector_outputs,
                                         image_sizes)

    def refine_bbox(self,
                    feature: Union[Tensor, Dict[str, Tensor]],
                    detector_outputs: Dict,
                    image_whwh: Tensor,
                    targets: List[Dict] = None) -> List[Dict]:
        # refinebox.utils.utils.parse_positive_samples_indices
        # can parse indices
        indices = None
        if self.training:
            indices = self.criterion.matcher(detector_outputs, targets)

        features = [feature] if isinstance(feature, Tensor) else feature
        if self.channel_mapper is not None:
            features: List[Tensor] = self.channel_mapper(features)

        outputs_list = list()
        refiner_inputs, detector_outputs = self._before_refinement(
            indices, detector_outputs, targets)

        for refine_step in range(self.refine_steps):

            bboxes_refiner = self.bboxes_refiner
            if isinstance(self.bboxes_refiner, nn.ModuleList):
                bboxes_refiner = self.bboxes_refiner[refine_step]

            # refiner_outputs contains only positive samples.
            refiner_outputs = bboxes_refiner(
                features, refiner_inputs, image_whwh)
            # _refiner_outputs contains both positive and negative samples.
            _refiner_outputs = self._after_refinement(
                indices, refiner_outputs, detector_outputs)

            refiner_inputs = refiner_outputs
            outputs_list.append(_refiner_outputs)

        return outputs_list

    def _before_refinement(self,
                           indices: List[Tensor],
                           outputs: Dict[str, Tensor],
                           targets: List[Dict]) -> Dict[str, Tensor]:
        """
        Extract positive samples for refinement.

        Args:
            indices: matched positive samples indices. Can be parsed by
                `refinebox.utils.utils.parse_positive_samples_indices`.
            outputs: detector (e.g., DETR, DN-DETR) outputs.
            targets: ground-truth targets.
        """
        if (not self.refine_only_matched) \
                or (not self.training
                    and self.bboxes_filter is None):
            return outputs, outputs

        detector_outputs = deepcopy(outputs)
        detector_outputs['pred_boxes'].detach_()
        detector_outputs['pred_logits'].detach_()

        if (not self.training) and self.bboxes_filter is not None:
            refiner_inputs = dict()
            bboxes = detector_outputs['pred_boxes']
            logits = detector_outputs['pred_logits']
            bboxes, logits = self.bboxes_filter(bboxes, logits)
            refiner_inputs['pred_boxes'] = bboxes
            refiner_inputs['pred_logits'] = logits
            return refiner_inputs, detector_outputs

        refiner_inputs = extract_positive_samples(
            detector_outputs, targets, indices)
        refiner_inputs['pred_boxes'] = refiner_inputs.pop('positive_bboxes')
        refiner_inputs['pred_logits'] = refiner_inputs.pop('positive_logits')

        return refiner_inputs, detector_outputs

    def _after_refinement(
            self,
            indices: List[Tensor],
            refiner_outputs: Dict[str, Tensor],
            detector_outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Update positive samples' bboxes of detector outputs."""
        if not self.refine_only_matched:
            return refiner_outputs

        if not self.training and self.bboxes_filter is not None:
            bboxes, logits = self.bboxes_filter.reverse(
                detector_outputs['pred_boxes'],
                detector_outputs['pred_logits'],
                refiner_outputs['pred_boxes'],
                refiner_outputs['pred_logits'])
            refiner_outputs = dict(pred_boxes=bboxes, pred_logits=logits)
            return refiner_outputs

        if self.training:
            refiner_outputs = update_positive_samples(detector_outputs,
                                                      refiner_outputs,
                                                      indices)
            return refiner_outputs

        return refiner_outputs

    def calc_refine_loss(self,
                         detector_outputs: Dict,
                         refiner_outputs: List[Dict],
                         targets: List[Dict]) -> Dict:
        """
        Args:
            detector_outputs: The outputs of DAB-DETR.
            refiner_outputs: Refined outputs.
        """
        indices: List[Tensor] = self.criterion.matcher(detector_outputs,
                                                       targets)
        num_bboxes = self._get_num_bboxes(detector_outputs, targets)
        loss_dict_list = list()
        bboxes_refiner = self.bboxes_refiner
        if isinstance(bboxes_refiner, nn.ModuleList):
            bboxes_refiner = bboxes_refiner[0]
        for pred in refiner_outputs:
            loss_dict = dict()
            if bboxes_refiner.bboxes_deltas is not None:
                loss_dict.update(self.criterion.get_loss('boxes',
                                                         pred,
                                                         targets,
                                                         indices,
                                                         num_bboxes))
            if bboxes_refiner.logits_deltas is not None:
                # Only used for ablation study.
                loss_dict.update(self.criterion.get_loss('class',
                                                         pred,
                                                         targets,
                                                         indices,
                                                         num_bboxes))
            loss_dict = self._assign_losses_weights(loss_dict)
            loss_dict_list.append(loss_dict)
        return self.format_loss_dict(loss_dict_list)

    def format_loss_dict(
            self,
            loss_dict_list: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        loss_dict = dict()
        for idx, loss in enumerate(loss_dict_list):
            for key, val in loss.items():
                loss_dict[f'loss.refine.{idx}.{key}'] = val
        return loss_dict

    def _assign_losses_weights(
            self, loss_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        weight_dict = self.criterion.weight_dict
        for k in loss_dict.keys():
            if k in weight_dict:
                loss_dict[k] *= weight_dict[k]
        return loss_dict

    def _get_num_bboxes(self,
                        outputs: Dict[str, Tensor],
                        targets: List[Dict]) -> float:
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes],
            dtype=torch.float,
            device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

    def _get_targets(self, batched_inputs: List[Dict]) -> List[Dict]:
        gt_instances = [x["instances"].to(self.device)
                        for x in batched_inputs]
        targets = self.prepare_targets(gt_instances)
        return targets

    def _process_results(self,
                         batched_inputs: List[Dict],
                         output: Dict[str, Tensor],
                         image_sizes: Tensor) -> List[Dict[str, Instances]]:
        box_cls = output["pred_logits"]
        box_pred = output["pred_boxes"]
        results = self.inference(box_cls, box_pred, image_sizes)
        # results = self.inference(box_cls, box_pred, image_sizes)
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            results, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

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

        # Select top-k confidence boxes for inference
        prob = box_cls.sigmoid()
        if not hasattr(self, 'select_box_nums_for_evaluation'):
            self.select_box_nums_for_evaluation = prob.shape[1]
            logger.info(f'Automatically set select_box_nums_for_evaluation '
                        f'as {self.select_box_nums_for_evaluation}')
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1),
            self.select_box_nums_for_evaluation,
            dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes,
                               box_cls.shape[2],
                               rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]

        boxes = torch.gather(box_pred,
                             1,
                             topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        for i, (scores_per_image,
                labels_per_image,
                box_pred_per_image,
                image_size) in enumerate(zip(scores,
                                             labels,
                                             boxes,
                                             image_sizes)):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1],
                                    scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h],
                                              dtype=torch.float,
                                              device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        return new_targets

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device))
                  for x in batched_inputs]
        images = ImageList.from_tensors(images)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(
                torch.tensor([w, h, w, h],
                             dtype=torch.float32,
                             device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh

    def _replace_annotations(self,
                             batched_inputs: List[Dict],
                             output: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if self.replace_bboxes or self.replace_labels:
            logger.warn(
                'This method can only be used for getting upper bound. '
                'Make sure you know what you are doing if you set '
                '`replace_bboxes` or `replace_labels` as True.'
                'Never set them as True when reporting results.')
            targets = self._get_targets(batched_inputs)
        if self.replace_bboxes:
            output = self._replace_bboxes(output, targets)
        if self.replace_labels:
            output = self._replace_labels(output, targets)
        return output

    def _replace_bboxes(self, prds: Dict[str, Tensor], trgs: Dict) -> Dict:
        logger.warn('This method can only be used for getting upper bound.')
        _preds = dict(pred_logits=prds['pred_logits'].clone(),
                      pred_boxes=prds['pred_boxes'].clone())
        indices = self.criterion.matcher(_preds, trgs)
        idx = self.criterion._get_src_permutation_idx(indices)

        trg_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(trgs, indices)], dim=0)
        _preds["pred_boxes"][idx] = trg_boxes

        return _preds

    def _replace_labels(self, prds: Dict[str, Tensor], trgs: Dict) -> Dict:
        logger.warn('This method can only be used for getting upper bound.')
        _preds = dict(pred_logits=prds['pred_logits'].clone(),
                      pred_boxes=prds['pred_boxes'].clone())
        indices = self.criterion.matcher(_preds, trgs)
        idx = self.criterion._get_src_permutation_idx(indices)

        src_logits = _preds['pred_logits']
        target_classes_o = torch.cat([t["labels"][J]
                                      for t, (_, J) in zip(trgs, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        # src_logits: (b, num_queries, num_classes) = (2, 300, 80)
        # target_classes_one_hot = (2, 300, 81)
        target_classes_onehot = torch.zeros(
            [
                src_logits.shape[0],
                src_logits.shape[1],
                src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1] * 100
        target_classes_onehot[target_classes == 80] = \
            src_logits[target_classes == 80]
        _preds['pred_logits'] = target_classes_onehot

        return _preds
