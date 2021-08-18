from typing import Union, Optional, Tuple
import time
import os
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from ding.worker import BaseLearner, LearnerHook
from ding.config import read_config, compile_config
from ding.torch_utils import resnet18
from ding.utils import set_pkg_seed, get_rank
from dizoo.image_classification.policy import ImageClassificationPolicy
from dizoo.image_classification.data import IterableImageNetDataset
from dizoo.image_classification.entry.imagenet_res18_config import imagenet_res18_config


class ImageClsLogShowHook(LearnerHook):

    def __init__(self, *args, freq: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._freq = freq

    def __call__(self, engine: 'BaseLearner') -> None:  # noqa
        # Only show log for rank 0 learner
        if engine.rank != 0:
            for k in engine.log_buffer:
                engine.log_buffer[k].clear()
            return
        # For 'scalar' type variables: log_buffer -> tick_monitor -> monitor_time.step
        for k, v in engine.log_buffer['scalar'].items():
            setattr(engine.monitor, k, v)
        engine.monitor.time.step()

        iters = engine.last_iter.val
        if iters % self._freq == 0:
            # For 'scalar' type variables: tick_monitor -> var_dict -> text_logger & tb_logger
            var_dict = {}
            log_vars = engine.policy.monitor_vars()
            attr = 'avg'
            for k in log_vars:
                k_attr = k + '_' + attr
                var_dict[k_attr] = getattr(engine.monitor, attr)[k]()
            # user-defined variable
            var_dict['data_time_val'] = engine.data_time
            epoch_info = engine.epoch_info
            var_dict['epoch_val'] = epoch_info[0]
            engine.logger.info(
                'Epoch: {} [{:>4d}/{}]\t'
                'Loss: {:>6.4f}\t'
                'Data Time: {:.3f}\t'
                'Forward Time: {:.3f}\t'
                'Backward Time: {:.3f}\t'
                'GradSync Time: {:.3f}\t'
                'LR: {:.3e}'.format(
                    var_dict['epoch_val'], epoch_info[1], epoch_info[2], var_dict['total_loss_avg'],
                    var_dict['data_time_val'], var_dict['forward_time_avg'], var_dict['backward_time_avg'],
                    var_dict['sync_time_avg'], var_dict['cur_lr_avg']
                )
            )
            for k, v in var_dict.items():
                engine.tb_logger.add_scalar('{}/'.format(engine.instance_name) + k, v, iters)
            # For 'histogram' type variables: log_buffer -> tb_var_dict -> tb_logger
            tb_var_dict = {}
            for k in engine.log_buffer['histogram']:
                new_k = '{}/'.format(engine.instance_name) + k
                tb_var_dict[new_k] = engine.log_buffer['histogram'][k]
            for k, v in tb_var_dict.items():
                engine.tb_logger.add_histogram(k, v, iters)
        for k in engine.log_buffer:
            engine.log_buffer[k].clear()


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
    log_show_hook = ImageClsLogShowHook(
        name='image_cls_log_show_hook', priority=0, position='after_iter', freq=cfg.policy.learn.learner.log_show_freq
    )
    learner.register_hook(log_show_hook)
    evaluator = None
    # ==========
    # Main loop
    # ==========
    learner.call_hook('before_run')
    end = time.time()

    for epoch in range(cfg.policy.learn.train_epoch):
        for i, train_data in enumerate(learn_dataloader):
            learner.data_time = time.time() - end
            learner.epoch_info = (epoch, i, len(learn_dataloader))
            learner.train(train_data)
            end = time.time()
        learner.policy.get_attribute('lr_scheduler').step()
        # Evaluate policy performance
        # evaluator.eval()

    learner.call_hook('after_run')


if __name__ == "__main__":
    main(imagenet_res18_config, 0)
