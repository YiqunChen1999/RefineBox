
import logging
from typing import List

import torch
from torch import Tensor
import torch.nn.functional as F
from detrex.modeling import SetCriterion
from refinebox.utils.utils import load_class_freq, get_fed_loss_inds, accuracy


logger = logging.getLogger(__name__)


def sigmoid_focal_loss(inputs: Tensor,
                       targets: Tensor,
                       num_boxes: float,
                       alpha: float = 0.25,
                       gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection:
    https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
            The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the
            binary classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
            positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
            balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs,
                                                 targets,
                                                 reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class SetCriterionWithFedLoss(SetCriterion):
    # Adapted from
    # https://github.com/facebookresearch/Detic/blob/main/detic/modeling/meta_arch/d2_deformable_detr.py
    def __init__(self,
                 num_classes,
                 matcher,
                 weight_dict,
                 losses: List[str] = ['class', 'boxes'],
                 eos_coef: float = 0.1,
                 loss_class_type: str = "focal_loss",
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 use_fed_loss: bool = False):
        super().__init__(num_classes=num_classes,
                         matcher=matcher,
                         weight_dict=weight_dict,
                         losses=losses,
                         eos_coef=eos_coef,
                         loss_class_type=loss_class_type,
                         alpha=alpha,
                         gamma=gamma)
        self.use_fed_loss = use_fed_loss
        if self.use_fed_loss:
            self.register_buffer(
                'fed_loss_weight', load_class_freq(freq_weight=0.5))

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim
        [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J]
                                      for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2],
                                    self.num_classes,
                                    dtype=torch.int64,
                                    device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [
                src_logits.shape[0],
                src_logits.shape[1],
                src_logits.shape[2] + 1,
            ],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]  # B x N x C
        if self.use_fed_loss:
            inds = get_fed_loss_inds(
                gt_classes=target_classes_o,
                num_sample_cats=50,
                weight=self.fed_loss_weight,
                C=target_classes_onehot.shape[2])
            loss_ce = sigmoid_focal_loss(
                src_logits[:, :, inds],
                target_classes_onehot[:, :, inds],
                num_boxes=num_boxes,
                alpha=self.alpha,
                gamma=self.gamma) * src_logits.shape[1]
        else:
            loss_ce = sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                num_boxes=num_boxes,
                alpha=self.alpha,
                gamma=self.gamma) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this
            # one here
            losses['class_error'] = 100 - accuracy(src_logits[idx],
                                                   target_classes_o)[0]
        return losses
