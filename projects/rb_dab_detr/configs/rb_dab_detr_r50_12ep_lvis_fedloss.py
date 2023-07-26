import os.path as osp
from detectron2.config import LazyCall as L
from detectron2.config import LazyConfig
from detrex.modeling import HungarianMatcher
from refinebox.models.layers.criterion import SetCriterionWithFedLoss
from refinebox.models.layers.bboxes_filter import TopKBboxesFilter
from refinebox.utils.utils import get_output_dir
from .rb_dab_detr_r50_12ep import train, optimizer, lr_multiplier, model

dataloader = LazyConfig.load('configs/common/data/lvis_detr.py').dataloader

# initialize checkpoint to be loaded
train.init_checkpoint = None  # Replace it with pre-trained model.
train.output_dir = get_output_dir(__file__)
train.refinebox_checkpoint = osp.join(train.output_dir, 'model_final.pth')

# run evaluation every 15000 iters, LVIS eval takes a long time
train.eval_period = 15000

# Refiner configs
model.num_classes = 1203
model.criterion = L(SetCriterionWithFedLoss)(
    num_classes=1203,
    matcher=L(HungarianMatcher)(
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        cost_class_type='focal_loss_cost',
        alpha=0.25,
        gamma=2.0,
    ),
    weight_dict={
        'loss_class': 1,
        'loss_bbox': 5.0,
        'loss_giou': 2.0,
    },
    loss_class_type='focal_loss',
    alpha=0.25,
    gamma=2.0,
    use_fed_loss=True,
)
model.modules = dict(
    bboxes_filter=TopKBboxesFilter(topk=300),
)

dataloader.evaluator.output_dir = train.output_dir
