import os.path as osp
from detrex.config import get_config
from detectron2.config import LazyConfig
from refinebox.utils.utils import get_output_dir
from .models.dab_detr_r50 import model

dataloader = LazyConfig.load("configs/common/data/lvis_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep
train = get_config("common/train.py").train

# initialize checkpoint to be loaded
# train.init_checkpoint = \
#   "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.init_checkpoint = \
    'detectron2://ImageNetPretrained/torchvision/R-50.pkl'
train.output_dir = get_output_dir(__file__)

# max training iterations
train.max_iter = 375000

# run evaluation every 5000 iters
train.eval_period = 15000

# log training infomation every 20 iters
train.log_period = 20

# save checkpoint every 5000 iters
train.checkpointer.period = 5000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device
model.num_classes = 1203
model.criterion.num_classes = 1203

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 \
    if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 16

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 16

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
