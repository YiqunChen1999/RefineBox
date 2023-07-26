import os.path as osp
from detectron2.config import LazyCall as L
from detectron2.config import LazyConfig
from detectron2.modeling.poolers import ROIPooler
from detrex.config import get_config
from refinebox.models.layers.fpn import RBFPN
from refinebox.models.layers.res_refiner import ResBboxRefiner
from refinebox.models.layers.bboxes_filter import TopKBboxesFilter
from refinebox.utils.utils import get_output_dir
from .models.rb_group_detr_r50 import model

dataloader = LazyConfig.load('configs/common/data/coco_detr.py').dataloader
optimizer = get_config('common/optim.py').AdamW
lr_multiplier = get_config('common/coco_schedule.py').lr_multiplier_12ep
train = get_config('common/train.py').train

train.init_checkpoint = None  # Replace it with pre-trained model.
train.output_dir = get_output_dir(__file__)
train.refinebox_checkpoint = osp.join(train.output_dir, 'model_final.pth')

# max training iterations
train.max_iter = 90000

# run evaluation every 5000 iters
train.eval_period = 5000

# log training infomation every 20 iters
train.log_period = 20

# save checkpoint every 5000 iters
train.checkpointer.period = 5000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = 'cuda'
model.device = train.device

# Refiner configs
model.refine_only_matched = True
model.refine_by_backbone_features = True
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
    in_features=['res2', 'res3', 'res4', 'res5'],
    out_channels=64,
)
model.modules = dict(
    bboxes_filter=TopKBboxesFilter(topk=100),
)

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = \
    lambda module_name: 0.1 if 'backbone' in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 16

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 16

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
