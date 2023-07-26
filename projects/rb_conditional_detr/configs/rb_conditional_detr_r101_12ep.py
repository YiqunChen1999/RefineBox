import os
import os.path as osp
from .rb_conditional_detr_r50_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)
from refinebox.utils.utils import get_output_dir

# modify model config
model.backbone.stages.depth = 101

# modify training config
train.init_checkpoint = None  # Replace it with pre-trained model.
train.output_dir = get_output_dir(__file__)
train.refinebox_checkpoint = osp.join(train.output_dir, 'model_final.pth')
