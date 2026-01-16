from typing import List, Dict
from easydict import EasyDict
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from ding.utils import REWARD_MODEL_REGISTRY
from .base_reward_model import BaseRewardModel


@REWARD_MODEL_REGISTRY.register('multi_modal')
class MultiModalRewardModel(BaseRewardModel):
    config = dict(
        type='multi_modal',
        model_name='internlm/internlm-xcomposer2d5-7b-reward',
        hd_num=9,  # Number of high-definition patches for image processing
    )

    def __init__(self, config: EasyDict, device: str, logger, tb_logger: 'SummaryWriter') -> None:
        self.cfg = config
        self.device = device
        self.logger = logger
        self.tb_logger = tb_logger

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name, trust_remote_code=True, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name, torch_dtype=torch.float16, trust_remote_code=True
        )

        self.model.tokenizer = self.tokenizer
        self.model.cuda().eval()

    def estimate(self, data: List[Dict], image: List[str], output_mode: str = 'score') -> List[Dict]:
        """
        Estimate rewards for multi-modal inputs using internlm-xcomposer model.

        Arguments:
            data (List[Dict]): List of chat dictionaries, each containing:
                - chat (List[Dict]): List of messages, each message is a dict with:
                    - role (str): Either "user" or "assistant"
                    - content (str): The message content
            image (List[str]): List of image paths. If fewer images than chats, last image will be reused
            output_mode (str, optional): Evaluation mode. Defaults to 'score'.
                - 'score': Return reward scores for each chat
                - 'rank': Return ranking indices (0 is best) for all chats
                - 'compare': Compare first two chats (returns 1.0 for better, 0.0 for worse)

        Returns:
            List[Dict]: Results depending on output_mode:
            - For 'score' mode:
                [{
                    'reward': float,  # Reward score
                    'metadata': {
                        'mode': 'score',
                        'chat_idx': int,  # Index of the chat
                        'image_path': str  # Path of the image used
                    }
                }, ...]
            - For 'rank' mode:
                [{
                    'rank': int,  # Ranking position (0 is best)
                    'metadata': {
                        'mode': 'rank',
                        'chat_idx': int,
                        'image_path': str
                    }
                }, ...]
            - For 'compare' mode:
                [{
                    'reward': float,  # 1.0 for better, 0.0 for worse
                    'metadata': {
                        'mode': 'compare',
                        'chat_idx': int,
                        'image_path': str,
                        'compared_with': int  # Index of the compared chat
                    }
                }, ...]
        """
        # Get chat data
        chats = [item['chat'] for item in data]

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            if output_mode == 'score':
                # Ensure each chat has a corresponding image, use the last image if not enough
                if len(image) < len(chats):
                    image = image + [image[-1]] * (len(chats) - len(image))

                # Get scores for each chat
                scores = []
                for chat, img in zip(chats, image):
                    score = self.model.get_score(chat, [img], hd_num=self.cfg.hd_num)
                    scores.append(score)

                return [
                    {
                        'reward': float(score),
                        'metadata': {
                            'mode': 'score',
                            'chat_idx': idx,
                            'image_path': img
                        }
                    } for idx, (score, img) in enumerate(zip(scores, image))
                ]

            elif output_mode == 'rank':
                # Use the first image for ranking
                img = image[0]
                ranks = self.model.rank(chats, [[img]] * len(chats), hd_num=self.cfg.hd_num)

                return [
                    {
                        'rank': int(rank),
                        'metadata': {
                            'mode': 'rank',
                            'chat_idx': idx,
                            'image_path': img
                        }
                    } for idx, rank in enumerate(ranks)
                ]

            elif output_mode == 'compare':
                if len(data) < 2:
                    raise ValueError("Compare mode requires at least 2 samples")

                # Use the first image for comparison
                img = image[0]
                is_better = self.model.compare(chats[0], [img], chats[1], [img], hd_num=self.cfg.hd_num)

                return [
                    {
                        'reward': 1.0 if is_better else 0.0,
                        'metadata': {
                            'mode': 'compare',
                            'chat_idx': 0,
                            'image_path': img,
                            'compared_with': 1
                        }
                    }, {
                        'reward': 0.0 if is_better else 1.0,
                        'metadata': {
                            'mode': 'compare',
                            'chat_idx': 1,
                            'image_path': img,
                            'compared_with': 0
                        }
                    }
                ]
            else:
                raise ValueError(f"Invalid output mode: {output_mode}")

    def train(self):
        """Training is not implemented for this reward model"""
        self.logger.warning("Training is not implemented for this reward model")
        pass

    def collect_data(self, data: list) -> None:
        """Data collection is not needed for this reward model"""
        pass

    def clear_data(self) -> None:
        """Data clearing is not needed for this reward model"""
        pass
