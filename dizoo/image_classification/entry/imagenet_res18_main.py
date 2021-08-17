from typing import Union, Optional, Tuple
import os
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from ding.worker import BaseLearner
from ding.config import read_config, compile_config
from ding.torch_utils import resnet18
from ding.utils import set_pkg_seed, get_rank
from dizoo.image_classification.policy import ImageClassificationPolicy
from dizoo.image_classification.data import IterableImageNetDataset
from dizoo.image_classification.entry.imagenet_res18_config import imagenet_res18_config


def main(cfg: dict, seed: int) -> None:
    cfg = compile_config(cfg, seed=seed, policy=ImageClassificationPolicy)

    # Random seed
    rank = get_rank()
    set_pkg_seed(cfg.seed + rank, use_cuda=cfg.policy.cuda)

    model = resnet18()
    policy = ImageClassificationPolicy(cfg.policy, model=model, enable_field=['learn', 'eval'])
    if cfg.policy.learn.multi_gpu:
        raise NotImplementedError
    else:
        learn_sampler, eval_sampler = None, None
    learn_dataset = IterableImageNetDataset(cfg.policy.collect.learn_data_path, is_training=True)
    eval_dataset = IterableImageNetDataset(cfg.policy.collect.eval_data_path, is_training=False)
    learn_dataloader = DataLoader(learn_dataset, cfg.policy.learn.batch_size, sampler=learn_sampler)
    eval_dataloader = DataLoader(eval_dataset, cfg.policy.eval.batch_size, sampler=eval_sampler)

    # Main components
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    evaluator = None
    # ==========
    # Main loop
    # ==========
    learner.call_hook('before_run')

    for epoch in range(cfg.policy.learn.train_epoch):
        for i, train_data in enumerate(learn_dataloader):
            learner.train(train_data)
        learner.policy.get_attribute('lr_scheduler').step()
        # Evaluate policy performance
        # evaluator.eval()

    learner.call_hook('after_run')


if __name__ == "__main__":
    main(imagenet_res18_config, 0)
