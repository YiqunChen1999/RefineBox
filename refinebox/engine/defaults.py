
import os
import os.path as osp
from datetime import datetime

import torch
from detectron2.config import CfgNode, LazyConfig
from detectron2.utils import comm
from detectron2.utils.env import seed_all_rng
from detectron2.utils.file_io import PathManager
from detectron2.utils.collect_env import collect_env_info
from detectron2.engine.defaults import _try_get_key, setup_logger, _highlight


def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode or omegaconf.DictConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = _try_get_key(cfg,
                              "OUTPUT_DIR",
                              "output_dir",
                              "train.output_dir")
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    now = datetime.now()
    now = now.strftime('%Y%m%d_%H%M%S')
    log_dir = osp.join(output_dir, f'logdir_{now}')
    setup_logger(log_dir, distributed_rank=rank, name="fvcore")
    setup_logger(log_dir, distributed_rank=rank, name="refinebox")
    cfg.train.log_dir = log_dir
    logger = setup_logger(log_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(
        rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file,
                _highlight(PathManager.open(args.config_file, "r").read(),
                           args.config_file),
            )
        )

    if comm.is_main_process() and log_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(log_dir, "config.yaml")
        if isinstance(cfg, CfgNode):
            logger.info("Running with full config:\n{}".format(
                _highlight(cfg.dump(), ".yaml")))
            with PathManager.open(path, "w") as f:
                f.write(cfg.dump())
        else:
            LazyConfig.save(cfg, path)
        logger.info("Full config saved to {}".format(path))

    # make sure each worker has a different, yet deterministic seed
    # if specified
    seed = _try_get_key(cfg, "SEED", "train.seed", default=-1)
    seed_all_rng(None if seed < 0 else seed + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering
    # the small size of typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = _try_get_key(
            cfg, "CUDNN_BENCHMARK", "train.cudnn_benchmark", default=False)
