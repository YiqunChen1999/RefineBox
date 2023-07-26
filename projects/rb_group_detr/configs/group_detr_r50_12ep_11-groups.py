from refinebox.utils.utils import get_output_dir
from .group_detr_r50_50ep import (
    model, dataloader, optimizer, lr_multiplier, train)

train.output_dir = get_output_dir(__file__)

# max training iterations
train.max_iter = 90000

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
