from typing import Optional, Callable, Tuple, Any, List
from abc import ABC, abstractmethod
from collections import namedtuple
import numpy as np
import torch
from torch.utils.data import DataLoader

from ding.torch_utils import to_tensor, to_ndarray
from ding.utils import build_logger, EasyTimer, SERIAL_EVALUATOR_REGISTRY, allreduce
from .base_serial_evaluator import ISerialEvaluator, VectorEvalMonitor


class IMetric(ABC):

    @abstractmethod
    def eval(self, inputs: Any, label: Any) -> dict:
        raise NotImplementedError

    @abstractmethod
    def reduce_mean(self, inputs: List[Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def gt(self, metric1: Any, metric2: Any) -> bool:
        """
        Overview:
            Whether metric1 is greater than metric2 (>=)

        .. note::
            If metric2 is None, return True
        """
        raise NotImplementedError


@SERIAL_EVALUATOR_REGISTRY.register('metric')
class MetricSerialEvaluator(ISerialEvaluator):
    """
    Overview:
        Metric serial evaluator class, policy is evaluated by objective metric(env).
    Interfaces:
        __init__, reset, reset_policy, reset_env, close, should_eval, eval
    Property:
        env, policy
    """

    config = dict(
        # Evaluate every "eval_freq" training iterations.
        eval_freq=50,
    )

    def __init__(
            self,
            cfg: dict,
            env: Tuple[DataLoader, IMetric] = None,
            policy: namedtuple = None,
            tb_logger: 'SummaryWriter' = None,  # noqa
            exp_name: Optional[str] = 'default_experiment',
            instance_name: Optional[str] = 'evaluator',
    ) -> None:
        """
        Overview:
            Init method. Load config and use ``self._cfg`` setting to build common serial evaluator components,
            e.g. logger helper, timer.
        Arguments:
            - cfg (:obj:`EasyDict`): Configuration EasyDict.
        """
        self._cfg = cfg
        self._exp_name = exp_name
        self._instance_name = instance_name
        if tb_logger is not None:
            self._logger, _ = build_logger(
                path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name, need_tb=False
            )
            self._tb_logger = tb_logger
        else:
            self._logger, self._tb_logger = build_logger(
                path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name
            )
        self.reset(policy, env)

        self._timer = EasyTimer()
        self._stop_value = cfg.stop_value

    def reset_env(self, _env: Optional[Tuple[DataLoader, IMetric]] = None) -> None:
        """
        Overview:
            Reset evaluator's environment. In some case, we need evaluator use the same policy in different \
                environments. We can use reset_env to reset the environment.
            If _env is not None, replace the old environment in the evaluator with the new one
        Arguments:
            - env (:obj:`Optional[Tuple[DataLoader, IMetric]]`): Instance of the DataLoader and Metric
        """
        if _env is not None:
            self._dataloader, self._metric = _env

    def reset_policy(self, _policy: Optional[namedtuple] = None) -> None:
        """
        Overview:
            Reset evaluator's policy. In some case, we need evaluator work in this same environment but use\
                different policy. We can use reset_policy to reset the policy.
            If _policy is None, reset the old policy.
            If _policy is not None, replace the old policy in the evaluator with the new passed in policy.
        Arguments:
            - policy (:obj:`Optional[namedtuple]`): the api namedtuple of eval_mode policy
        """
        if _policy is not None:
            self._policy = _policy
        self._policy.reset()

    def reset(self, _policy: Optional[namedtuple] = None, _env: Optional[Tuple[DataLoader, IMetric]] = None) -> None:
        """
        Overview:
            Reset evaluator's policy and environment. Use new policy and environment to collect data.
            If _env is not None, replace the old environment in the evaluator with the new one
            If _policy is None, reset the old policy.
            If _policy is not None, replace the old policy in the evaluator with the new passed in policy.
        Arguments:
            - policy (:obj:`Optional[namedtuple]`): the api namedtuple of eval_mode policy
            - env (:obj:`Optional[Tuple[DataLoader, IMetric]]`): Instance of the DataLoader and Metric
        """
        if _env is not None:
            self.reset_env(_env)
        if _policy is not None:
            self.reset_policy(_policy)
        self._max_avg_eval_result = None
        self._last_eval_iter = -1
        self._end_flag = False

    def close(self) -> None:
        """
        Overview:
            Close the evaluator. If end_flag is False, close the environment, flush the tb_logger\
                and close the tb_logger.
        """
        if self._end_flag:
            return
        self._end_flag = True
        self._tb_logger.flush()
        self._tb_logger.close()

    def __del__(self):
        """
        Overview:
            Execute the close command and close the evaluator. __del__ is automatically called \
                to destroy the evaluator instance when the evaluator finishes its work
        """
        self.close()

    def should_eval(self, train_iter: int) -> bool:
        """
        Overview:
            Determine whether you need to start the evaluation mode, if the number of training has reached\
                the maximum number of times to start the evaluator, return True
        """
        if train_iter == self._last_eval_iter:
            return False
        if (train_iter - self._last_eval_iter) < self._cfg.eval_freq and train_iter != 0:
            return False
        self._last_eval_iter = train_iter
        return True

    def eval(
            self,
            save_ckpt_fn: Callable = None,
            train_iter: int = -1,
            envstep: int = -1,
    ) -> Tuple[bool, Any]:
        '''
        Overview:
            Evaluate policy and store the best policy based on whether it reaches the highest historical reward.
        Arguments:
            - save_ckpt_fn (:obj:`Callable`): Saving ckpt function, which will be triggered by getting the best reward.
            - train_iter (:obj:`int`): Current training iteration.
            - envstep (:obj:`int`): Current env interaction step.
        Returns:
            - stop_flag (:obj:`bool`): Whether this training program can be ended.
            - eval_metric (:obj:`float`): Current evaluation metric result.
        '''
        self._policy.reset()
        eval_results = []

        with self._timer:
            self._logger.info("Evaluation begin...")
            for batch_idx, batch_data in enumerate(self._dataloader):
                inputs, label = to_tensor(batch_data)
                policy_output = self._policy.forward(inputs)
                eval_results.append(self._metric.eval(policy_output, label))
            avg_eval_result = self._metric.reduce_mean(eval_results)
            if self._cfg.multi_gpu:
                device = self._policy.get_attribute('device')
                for k in avg_eval_result.keys():
                    value_tensor = torch.FloatTensor([avg_eval_result[k]]).to(device)
                    allreduce(value_tensor)
                    avg_eval_result[k] = value_tensor.item()

        duration = self._timer.value
        info = {
            'train_iter': train_iter,
            'ckpt_name': 'iteration_{}.pth.tar'.format(train_iter),
            'data_length': len(self._dataloader),
            'evaluate_time': duration,
            'avg_time_per_data': duration / len(self._dataloader),
        }
        info.update(avg_eval_result)
        self._logger.info(self._logger.get_tabulate_vars_hor(info))
        # self._logger.info(self._logger.get_tabulate_vars(info))
        for k, v in info.items():
            if k in ['train_iter', 'ckpt_name']:
                continue
            if not np.isscalar(v):
                continue
            self._tb_logger.add_scalar('{}_iter/'.format(self._instance_name) + k, v, train_iter)
            self._tb_logger.add_scalar('{}_step/'.format(self._instance_name) + k, v, envstep)
        if self._metric.gt(avg_eval_result, self._max_avg_eval_result):
            if save_ckpt_fn:
                save_ckpt_fn('ckpt_best.pth.tar')
            self._max_avg_eval_result = avg_eval_result
        stop_flag = self._metric.gt(avg_eval_result, self._stop_value) and train_iter > 0
        if stop_flag:
            self._logger.info(
                "[DI-engine serial pipeline] " +
                "Current eval_reward: {} is greater than stop_value: {}".format(avg_eval_result, self._stop_value) +
                ", so your RL agent is converged, you can refer to 'log/evaluator/evaluator_logger.txt' for details."
            )
        return stop_flag, avg_eval_result
