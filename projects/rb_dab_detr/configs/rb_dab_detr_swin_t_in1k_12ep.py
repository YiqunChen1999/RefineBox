import os.path as osp
from detectron2.config import LazyCall as L
from detectron2.modeling.poolers import ROIPooler
from refinebox.models.layers.fpn import RBFPN
from refinebox.models.layers.res_refiner import ResBboxRefiner
from refinebox.models.layers.bboxes_filter import TopKBboxesFilter
from refinebox.utils.utils import get_output_dir
from .rb_dab_detr_r50_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
)
from .models.rb_dab_detr_swin_tiny import model

# modify training config
train.init_checkpoint = None  # Replace it with pre-trained model.
train.output_dir = get_output_dir(__file__)
train.refinebox_checkpoint = osp.join(train.output_dir, 'model_final.pth')

# Refiner configs
model.refine_only_matched = True
model.refine_steps = 3
model.bboxes_refiner = L(ResBboxRefiner)(
    num_classes=80,
    d_model=64,
    # # num_cls_layers=2,
    num_reg_layers=2,
    roi_layer=L(ROIPooler)(
        output_size=(7, 7),
        scales=[1/(2**i) for i in range(2, 6)],
        sampling_ratio=2,
        pooler_type='ROIAlignV2',
    ),
    # Only update some configs, default configs are available in
    # refinebox/models/layers/res_resfiner.py
    res_block_cfg=dict(in_channels=64,
                       bottleneck_channels=64,
                       out_channels=64,
                       num_groups=1),  # num_groups is the groups of 3x3 conv.
)
model.channel_mapper = L(RBFPN)(
    bottom_up=model.backbone,
    in_features=['p0', 'p1', 'p2', 'p3'],
    out_channels=64,
)
model.modules = dict(
    bboxes_filter=TopKBboxesFilter(topk=100),
)
