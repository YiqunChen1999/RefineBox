import os.path as osp
from detectron2.config import LazyConfig
from .rb_dab_detr_r50_12ep import (
    train,
    optimizer,
    lr_multiplier,
    model)
from refinebox.models.layers.bboxes_filter import TopKBboxesFilter
from refinebox.utils.utils import get_output_dir

dataloader = LazyConfig.load('configs/common/data/lvis_detr.py').dataloader

# initialize checkpoint to be loaded
train.init_checkpoint = None  # Replace it with pre-trained model.
train.output_dir = get_output_dir(__file__)
train.refinebox_checkpoint = osp.join(train.output_dir, 'model_final.pth')

# run evaluation every 15000 iters, LVIS eval takes a long time
train.eval_period = 15000

# Refiner configs
model.num_classes = 1203
model.criterion.num_classes = 1203

model.modules = dict(
    bboxes_filter=TopKBboxesFilter(topk=300),
)
