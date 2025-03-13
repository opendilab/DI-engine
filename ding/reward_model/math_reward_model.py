from typing import Tuple, Optional, List, Dict
from easydict import EasyDict
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import re

from ding.utils import REWARD_MODEL_REGISTRY
from .base_reward_model import BaseRewardModel


@REWARD_MODEL_REGISTRY.register('math')
class MathRewardModel(BaseRewardModel):
    config = dict(
        # (str) The type of the reward model.
        type='math',
        # (str) The name of the tokenizer and model
        model_name='Qwen/Qwen2.5-Math-PRM-7B',
    )

    def __init__(self, config: EasyDict, device: str, logger, tb_logger: 'SummaryWriter') -> None:  # noqa
        self.cfg = config
        self.device = device
        self.logger = logger
        self.tb_logger = tb_logger

        # 初始化tokenizer和model
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.cfg.model_name, device_map=self.device, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        self.model.eval()

    def make_step_rewards(self, logits: torch.Tensor, token_masks: torch.Tensor) -> List[List[float]]:
        """Calculate step-wise rewards from model outputs"""
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)  # bs, seq_len, num_labels

        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]  # seq_len, num_labels
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]  # valid_tokens, num_labels
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res

    def estimate(self, data: List[Dict]) -> List[Dict]:
        """
        Overview:
            Estimate rewards for mathematical reasoning steps using Qwen2.5-Math-PRM-7B model.
        Arguments:
            - data (:obj:`List[Dict]`): List of dictionaries containing:
                - system (:obj:`str`): System prompt for the model
                - query (:obj:`str`): The mathematical query to be evaluated
                - response (:obj:`List[str]`): List of reasoning steps
        Returns:
            - reward (:obj:`List[Dict]`): List of dictionaries containing:
                - reward (:obj:`float`): Final reward (last step reward)
                - metadata (:obj:`Dict`): Additional information including:
                    - query (:obj:`str`): Original query
                    - step_rewards (:obj:`List[float]`): Rewards for each reasoning step
                    - num_steps (:obj:`int`): Number of reasoning steps
        Shapes:
            - input_ids (:obj:`torch.LongTensor`): :math:`(B, L)`, where B is batch size and L is sequence length
            - outputs (:obj:`torch.FloatTensor`): :math:`(B, L, H)`, where H is hidden size
            - token_masks (:obj:`torch.BoolTensor`): :math:`(B, L)`
            - step_rewards (:obj:`List[List[float]]`): List of length B, each containing S rewards where S is num steps
        Examples:
            >>> data = [{
            >>>     "system": "Please reason step by step...",
            >>>     "query": "What is 1 + 1?",
            >>>     "response": ["First, we have 1", "Then add 1", "Therefore, 1 + 1 = 2"]
            >>> }]
            >>> results = model.estimate(data)
            >>> print(results[0]["reward"])  # 1.0
            >>> print(results[0]["metadata"]["step_rewards"])  # [0.8, 0.9, 1.0]
        """
        # 批量处理所有样本
        all_messages = []
        for item in data:
            messages = [
                {
                    "role": "system",
                    "content": item['system']
                },
                {
                    "role": "user",
                    "content": item['query']
                },
                {
                    "role": "assistant",
                    "content": "<extra_0>".join(item['response']) + "<extra_0>"
                },
            ]
            all_messages.append(messages)

        # 批量转换为模型输入格式
        conversation_strs = [
            self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            for messages in all_messages
        ]

        # 批量编码输入
        input_ids = self.tokenizer(
            conversation_strs, return_tensors="pt", padding=True, truncation=True
        )["input_ids"].to(self.model.device)

        # 批量获取模型输出
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)

        # 计算每个样本的步骤奖励
        step_sep_id = self.tokenizer.encode("<extra_0>")[0]
        token_masks = (input_ids == step_sep_id)
        batch_rewards = self.make_step_rewards(outputs[0], token_masks)

        # 构建详细的结果字典
        results = []
        for item, step_rewards in zip(data, batch_rewards):
            results.append(
                {
                    "reward": step_rewards[-1] if step_rewards else 0.0,  # 最后一步的奖励作为总体奖励
                    "metadata": {
                        "query": item['query'],
                        "step_rewards": step_rewards,  # 每个步骤的奖励
                        "num_steps": len(item['response']),
                    }
                }
            )

        return results

    def train(self):
        """
        Training is not implemented for this reward model as it uses a pre-trained model
        """
        self.logger.warning("Training is not implemented for this reward model")
        pass

    def collect_data(self, data: list) -> None:
        """
        Data collection is not needed for this reward model
        """
        pass

    def clear_data(self) -> None:
        """
        Data clearing is not needed for this reward model
        """
        pass
