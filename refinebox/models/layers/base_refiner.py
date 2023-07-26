
import warnings

import torch
from torch import nn, Tensor

from detrex.utils import inverse_sigmoid


INF = 1E8


class BaseBboxRefiner(nn.Module):
    def __init__(
            self,
            num_classes: int,
            d_model: int,
            num_reg_layers: int,
            num_cls_layers: int = None,
            cls_as_obj_score: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.cls_as_obj_score = cls_as_obj_score
        self.d_model = d_model

        self.cls_module = None
        self.logits_deltas = None
        if num_cls_layers is not None and num_cls_layers > 0:
            warnings.warn('You choose to refine class logits.')
            cls_module = list()
            for _ in range(num_cls_layers):
                cls_module.append(nn.Linear(d_model, d_model, False))
                cls_module.append(nn.LayerNorm(d_model))
                cls_module.append(nn.ReLU(inplace=True))
            self.cls_module = nn.ModuleList(cls_module)
            if cls_as_obj_score:
                num_classes = 1
            self.logits_deltas = nn.Linear(d_model, num_classes)

        self.reg_module = None
        self.bboxes_deltas = None
        if num_reg_layers is not None and num_reg_layers > 0:
            reg_module = list()
            for _ in range(num_reg_layers):
                reg_module.append(nn.Linear(d_model, d_model, False))
                reg_module.append(nn.LayerNorm(d_model))
                reg_module.append(nn.ReLU(inplace=True))
            self.reg_module = nn.ModuleList(reg_module)
            self.bboxes_deltas = nn.Linear(d_model, 4)

    def apply_bboxes_deltas(self, deltas: Tensor, bboxes: Tensor) -> Tensor:
        """
        Args:
            deltas:
            bboxes: Normalized (cx, cy, w, h) with shape (B, N, 4)
        """
        bboxes = inverse_sigmoid(bboxes) + deltas
        bboxes = bboxes.sigmoid()
        return bboxes

    def apply_logits_deltas(self, deltas: Tensor, logits: Tensor) -> Tensor:
        """
        Args:
            deltas:
            bboxes: Unnormalized scores with shape (B, N, K)
        """
        # logits = logits + deltas
        # return logits
        # use YOLOF style.
        normed_logits = (logits
                         + deltas
                         - torch.log(1.0
                                     + torch.clamp(logits.exp(), max=INF)
                                     + torch.clamp(deltas.exp(), max=INF)))
        return normed_logits
