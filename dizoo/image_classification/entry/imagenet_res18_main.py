from typing import Union, Optional, Tuple, List
import time
import os
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from ding.worker import BaseLearner, LearnerHook, MetricSerialEvaluator, IMetric
from ding.config import read_config, compile_config
from ding.torch_utils import resnet18
from ding.utils import set_pkg_seed, get_rank, dist_init
from dizoo.image_classification.policy import ImageClassificationPolicy
from dizoo.image_classification.data import ImageNetDataset, DistributedSampler
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


class ImageClassificationMetric(IMetric):

    def __init__(self) -> None:
        self.loss = torch.nn.CrossEntropyLoss()

    @staticmethod
    def accuracy(inputs: torch.Tensor, label: torch.Tensor, topk: Tuple = (1, 5)) -> dict:
        """Computes the accuracy over the k top predictions for the specified values of k"""
        maxk = max(topk)
        batch_size = label.size(0)
        _, pred = inputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(label.reshape(1, -1).expand_as(pred))
        return {'acc{}'.format(k): correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk}

    def eval(self, inputs: torch.Tensor, label: torch.Tensor) -> dict:
        """
        Returns:
            - eval_result (:obj:`dict`): {'loss': xxx, 'acc1': xxx, 'acc5': xxx}
        """
        loss = self.loss(inputs, label)
        output = self.accuracy(inputs, label)
        output['loss'] = loss
        for k in output:
            output[k] = output[k].item()
        return output

    def reduce_mean(self, inputs: List[dict]) -> dict:
        L = len(inputs)
        output = {}
        for k in inputs[0].keys():
            output[k] = sum([t[k] for t in inputs]) / L
        return output

    def gt(self, metric1: dict, metric2: dict) -> bool:
        if metric2 is None:
            return True
        for k in metric1:
            if metric1[k] < metric2[k]:
                return False
        return True


def main(cfg: dict, seed: int) -> None:
    cfg = compile_config(cfg, seed=seed, policy=ImageClassificationPolicy, evaluator=MetricSerialEvaluator)
    if cfg.policy.multi_gpu:
        rank, world_size = dist_init()
    else:
        rank, world_size = 0, 1

    # Random seed
    set_pkg_seed(cfg.seed + rank, use_cuda=cfg.policy.cuda)

    model = resnet18()
    policy = ImageClassificationPolicy(cfg.policy, model=model, enable_field=['learn', 'eval'])
    learn_dataset = ImageNetDataset(cfg.policy.collect.learn_data_path, is_training=True)
    eval_dataset = ImageNetDataset(cfg.policy.collect.eval_data_path, is_training=False)
    if cfg.policy.multi_gpu:
        learn_sampler = DistributedSampler(learn_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    else:
        learn_sampler, eval_sampler = None, None
    learn_dataloader = DataLoader(learn_dataset, cfg.policy.learn.batch_size, sampler=learn_sampler, num_workers=3)
    eval_dataloader = DataLoader(eval_dataset, cfg.policy.eval.batch_size, sampler=eval_sampler, num_workers=2)

    # Main components
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    log_show_hook = ImageClsLogShowHook(
        name='image_cls_log_show_hook', priority=0, position='after_iter', freq=cfg.policy.learn.learner.log_show_freq
    )
    learner.register_hook(log_show_hook)
    eval_metric = ImageClassificationMetric()
    evaluator = MetricSerialEvaluator(
        cfg.policy.eval.evaluator, [eval_dataloader, eval_metric], policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    # ==========
    # Main loop
    # ==========
    learner.call_hook('before_run')
    end = time.time()

    for epoch in range(cfg.policy.learn.train_epoch):
        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, epoch, 0)
            if stop:
                break
        for i, train_data in enumerate(learn_dataloader):
            learner.data_time = time.time() - end
            learner.epoch_info = (epoch, i, len(learn_dataloader))
            learner.train(train_data)
            end = time.time()
        learner.policy.get_attribute('lr_scheduler').step()

    learner.call_hook('after_run')


if __name__ == "__main__":
    main(imagenet_res18_config, 0)
