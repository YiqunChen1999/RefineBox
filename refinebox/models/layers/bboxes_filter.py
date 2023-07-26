
from typing import List, Dict, Tuple

import torch
from torch import nn, Tensor


class TopKBboxesFilter(nn.Module):
    def __init__(self, topk: int = 300) -> None:
        super().__init__()
        self.topk = topk
        self.topk_ids: Tensor = None

    def forward(self, bboxes: Tensor, logits: Tensor):
        """
        Args:
            logits (torch.Tensor): tensor of shape
                ``(batch_size, num_queries, K)``. The tensor predicts the
                classification probability for each query.
            bboxes (torch.Tensor): tensors of shape
                ``(batch_size, num_queries, 4)``. The tensor predicts
                4-vector ``(x, y, w, h)`` box regression values for every
                query.
        """
        """I don't know why DAB-DETR choose such method.
        # Select top-k confidence boxes for inference
        prob = logits.sigmoid()
        _, topk_indexes = torch.topk(
            prob.view(logits.shape[0], -1),
            self.topk,
            dim=1,
        )
        topk_boxes = torch.div(topk_indexes,
                               logits.shape[2],
                               rounding_mode="floor")
        bboxes = torch.gather(bboxes,
                              1,
                              topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        logits = torch.gather(logits,
                              1,
                              topk_boxes.unsqueeze(-1).repeat(1, 1, 4))"""

        scores, _ = logits.max(dim=-1)
        num_classes = logits.shape[-1]
        topk_ids: Tensor = torch.topk(scores, self.topk, dim=-1)[1]
        self.topk_ids = topk_ids
        bboxes = torch.gather(bboxes,
                              1,
                              topk_ids.unsqueeze(-1).repeat(1, 1, 4))
        logits = torch.gather(logits,
                              1,
                              topk_ids.unsqueeze(-1).repeat(1, 1, num_classes))
        return bboxes, logits

    def reverse(self,
                original_bboxes: Tensor,
                original_logits: Tensor,
                filtered_bboxes: Tensor,
                filtered_logits: Tensor):
        num_classes = original_logits.shape[-1]
        original_bboxes = torch.scatter(
            original_bboxes,
            1,
            self.topk_ids.unsqueeze(-1).repeat(1, 1, 4),
            filtered_bboxes)
        original_logits = torch.scatter(
            original_logits,
            1,
            self.topk_ids.unsqueeze(-1).repeat(1, 1, num_classes),
            filtered_logits)
        return original_bboxes, original_logits
