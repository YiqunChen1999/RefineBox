
import os
import json
import inspect
import logging
from typing import List, Any
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn, Tensor


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedIndices:
    batch_ids: Tensor = None
    index_in_batch: Tensor = None
    matched_ids_gt: Tensor = None
    matched_ids_pred: Tensor = None


def get_output_dir(path2config: os.PathLike) -> os.PathLike:
    return f"./output/{path2config.split('/')[-1].replace('.py', '')}"


def init_detrs_with_mixin(self,
                          mixin_class: nn.Module,
                          base_class: nn.Module,
                          kwargs: dict[str, Any]):
    _kwargs = extract_kwargs_by_signature(
        inspect.signature(base_class.__init__), kwargs)
    base_class.__init__(self, **_kwargs)
    _kwargs = extract_kwargs_by_signature(
        inspect.signature(mixin_class.__init__), kwargs)
    mixin_class.__init__(self, **_kwargs)
    if len(kwargs) > 0:
        logger.warn(f'Following arguments are not used: \n{kwargs}')


def extract_kwargs_by_signature(signature: inspect.Signature,
                                kwargs: dict[str, Any]) -> dict[str, Any]:
    _kwargs = dict()
    OLD_KEYS = ('bbox_refiner', )
    NEW_KEYS = ('bboxes_refiner', )
    modules = kwargs.pop('modules', None)
    if modules is not None:
        kwargs.update(modules)
    for old_key, new_key in zip(OLD_KEYS, NEW_KEYS):
        if old_key in kwargs.keys():
            logger.info(f'Rename argument {old_key} to {new_key}.')
            kwargs[new_key] = kwargs.pop(old_key)
    _kwargs = dict()
    for key in signature.parameters.keys():
        if key == 'self':
            continue
        if key in kwargs.keys():
            _kwargs[key] = kwargs.pop(key)
    return _kwargs


def extract_positive_samples(
        outputs: dict[str, Tensor],
        targets: list[dict],
        indices: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
    batch = len(targets)
    device = outputs["pred_boxes"].device
    parsed_indices = parse_positive_samples_indices(indices)
    batch_ids = parsed_indices.batch_ids
    matched_ids_pred = parsed_indices.matched_ids_pred
    # matched_ids_gt = parsed_indices.matched_ids_gt
    index_in_batch = parsed_indices.index_in_batch
    num_gts_per_img = calc_num_gts_per_img(targets=targets)
    if num_gts_per_img.numel() == 0:
        num_gts_per_img = torch.as_tensor([1], type=torch.long)
    num_cats = outputs['pred_logits'].shape[-1]

    positive_bboxes = torch.zeros((batch, num_gts_per_img.max(), 4),
                                  device=device)
    positive_logits = torch.zeros((batch, num_gts_per_img.max(), num_cats),
                                  device=device)

    # (batch, num_gts, 4)
    positive_bboxes[batch_ids, index_in_batch] = \
        outputs['pred_boxes'][batch_ids, matched_ids_pred]
    positive_logits[batch_ids, index_in_batch] = \
        outputs['pred_logits'][batch_ids, matched_ids_pred]

    positive_samples = dict(positive_bboxes=positive_bboxes,
                            positive_logits=positive_logits,
                            indices=indices,
                            parsed_indices=parsed_indices)
    return positive_samples


def update_positive_samples(
        outputs: dict[str, Tensor],
        predictions: dict[str, Tensor],
        indices: tuple[str, Tensor] = None,
        parsed_indices: ParsedIndices = None) -> dict[str, Tensor]:
    if parsed_indices is None:
        assert indices is not None
        parsed_indices = parse_positive_samples_indices(indices)
    batch_ids = parsed_indices.batch_ids
    matched_ids_pred = parsed_indices.matched_ids_pred
    index_in_batch = parsed_indices.index_in_batch

    outputs = deepcopy(outputs)
    outputs['pred_boxes'][batch_ids, matched_ids_pred] = \
        predictions['pred_boxes'][batch_ids, index_in_batch]
    outputs['pred_logits'][batch_ids, matched_ids_pred] = \
        predictions['pred_logits'][batch_ids, index_in_batch]

    return outputs


def calc_num_gts_per_img(indices: tuple[Tensor, Tensor] = None,
                         targets: list[dict] = None,
                         parsed_indices: ParsedIndices = None) -> Tensor:
    if indices is not None:
        parsed_indices = parse_positive_samples_indices(indices)
    if parsed_indices is not None:
        return torch.unique(parsed_indices.batch_ids, return_counts=True)[1]

    num_gts_per_img = list()
    for trg in targets:
        num_gts_per_img.append(len(trg['labels']))
    num_gts_per_img = torch.as_tensor(num_gts_per_img)
    if num_gts_per_img.numel() == 0:
        num_gts_per_img = torch.zeros((len(targets), ), dtype=torch.int)

    return num_gts_per_img


def parse_positive_samples_indices(
        indices: List[tuple[Tensor, Tensor]]) -> ParsedIndices:
    """
    Format the indices of positive samples, including gt ids and preds ids.

    Args:
        indices: Each element records the indices information of one image.
            The format of each element is (matched_preds_idx, matched_gt_idx).

    Returns:
        batch_ids: Indicate which image the corresponding matched indices
            belong to.
        matched_ids_pred: The indices in the predicted results of matched
            bboxes.
        matched_ids_gt: The indices in the ground truth of the corresponding
            matched gt bboxes.
    """
    batch_ids = [torch.full_like(src, i)
                 for i, (src, _) in enumerate(indices)]
    matched_ids_pred = [src for (src, _) in indices]
    matched_ids_gt = [src for (_, src) in indices]
    index_in_batch = [torch.arange(len(b)) for b in batch_ids]

    # sort_ids = [ids.sort(stable=True)[1] for ids in matched_ids_gt]
    # matched_ids_gt = [ids.sort(stable=True)[0] for ids in matched_ids_gt]
    # matched_ids_pred = [p[idx] for p, idx in zip(matched_ids_pred, sort_ids)]

    batch_ids = torch.cat(batch_ids)
    index_in_batch = torch.cat(index_in_batch)
    matched_ids_gt = torch.cat(matched_ids_gt)
    matched_ids_pred = torch.cat(matched_ids_pred)

    return ParsedIndices(batch_ids=batch_ids,
                         index_in_batch=index_in_batch,
                         matched_ids_gt=matched_ids_gt,
                         matched_ids_pred=matched_ids_pred)


def load_class_freq(
        path='datasets/lvis/lvis_v1_train_cat_info.json',
        freq_weight=1.0):
    cat_info = json.load(open(path, 'r'))
    cat_info = torch.tensor(
        [c['image_count'] for c in sorted(cat_info, key=lambda x: x['id'])])
    freq_weight = cat_info.float() ** freq_weight
    return freq_weight


def get_fed_loss_inds(gt_classes, num_sample_cats, C, weight=None):
    appeared = torch.unique(gt_classes)
    prob = appeared.new_ones(C + 1).float()
    prob[-1] = 0
    if len(appeared) < num_sample_cats:
        if weight is not None:
            prob[:C] = weight.float().clone()
        prob[appeared] = 0
        more_appeared = torch.multinomial(
            prob, num_sample_cats - len(appeared),
            replacement=False)
        appeared = torch.cat([appeared, more_appeared])
    return appeared


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
