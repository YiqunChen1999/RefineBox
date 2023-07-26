
from typing import List, Dict
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from fvcore.nn import weight_init
from detectron2.modeling.backbone.fpn import FPN, Backbone
from detectron2.layers import Conv2d, get_norm
from detectron2.modeling.backbone.resnet import BottleneckBlock, ResNet
from detrex.layers import ShapeSpec


class RBFPN(FPN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.bottom_up

    def forward(self, bottom_up_features: Dict[str, Tensor]) -> List[Tensor]:
        results = []
        prev_features = self.lateral_convs[0](
            bottom_up_features[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))

        # Reverse feature maps into top-down order
        # (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
                zip(self.lateral_convs, self.output_convs)):
            # Slicing of ModuleList is not supported
            # https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = self.in_features[-idx - 1]
                features = bottom_up_features[features]
                lateral_features: Tensor = lateral_conv(features)
                top_down_features: Tensor = F.interpolate(
                    prev_features,
                    size=lateral_features.shape[-2:],
                    mode="nearest")
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = \
                    bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[
                    self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        # return {f: res for f, res in zip(self._out_features, results)}
        return results
