from detectron2.config import LazyCall as L
from detrex.modeling import HungarianMatcher
from refinebox.utils.utils import get_output_dir
from refinebox.models.layers.criterion import SetCriterionWithFedLoss
from .dab_detr_r50_50ep_lvis_v1 import (
    model,
    train,
    dataloader,
    optimizer,
    lr_multiplier)


# initialize checkpoint to be loaded
# train.init_checkpoint = \
#   "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.init_checkpoint = \
    'detectron2://ImageNetPretrained/torchvision/R-50.pkl'
train.output_dir = get_output_dir(__file__)

# run evaluation every 5000 iters
train.eval_period = 370000

# set model
model.num_classes = 1203
model.criterion = L(SetCriterionWithFedLoss)(
    num_classes=1203,
    matcher=L(HungarianMatcher)(
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        cost_class_type="focal_loss_cost",
        alpha=0.25,
        gamma=2.0,
    ),
    weight_dict={
        "loss_class": 1,
        "loss_bbox": 5.0,
        "loss_giou": 2.0,
    },
    loss_class_type="focal_loss",
    alpha=0.25,
    gamma=2.0,
    use_fed_loss=True,
)

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
