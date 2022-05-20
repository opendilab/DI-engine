from typing import TYPE_CHECKING, Dict
# easydict的作用：可以使得以属性的方式去访问字典的值
from easydict import EasyDict
import copy
import torch
import torch.nn as nn

# 为了防止循环引用出现的差错
if TYPE_CHECKING:
    from tensorboardX import SummaryWriter

class encoder(nn.Module):
    pass

class CURL:
    @classmethod
    def default_config(cls: type) -> EasyDict:
        # copy.deepcopy 深拷贝 拷贝对象及其子对象
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg
        #cls指类，config为一个类
    config = dict(
        batch_size,
        z_dim,
        learning_rate=1e-3,
        encoder_feature_size=50,
        target_theta=0.005,
    )

    def __init__(
            self, cfg: EasyDict, encoder: nn.Module, encoder_target: nn.Module, tb_logger: "SummaryWriter"
    ) -> None:
        """
        Overview:
            Create main components, such as paramater, optimizer, loss function and so on.
        """
        super(CURL,self).__init__()

        self.cfg = cfg
        self.encoder = encoder
        self.encoder_target = encoder_target
        self.tb_logger = tb_logger

        # parameter
        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        # optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )
        self.encoder_optimizer = torch.optim.Adam(
            self.critic.encoder.parameters(), lr=encoder_lr
        )
        self.cpc_optimizer = torch.optim.Adam(
            self.CURL.parameters(), lr=encoder_lr
        )
        # loss function
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def compute_logits(self,z_a,z_pos):
        """
            Uses logits trick for CURL:
            - compute (B,B) matrix z_a (W z_pos.T)
            - positives are all diagonal elements
            - negatives are all other elements
            - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Encode original observation into more compact embedding feature.
        Arguments:
            - obs (:obj:`torch.Tensor`): Original observation returned from environment, e.g.: stacked image in Atari.
        Returns:
            - embedding (:obj:`torch.Tensor`): Embedded feature learner by CURL.
        Shapes:
            - obs: :math:`(B, C, H, W)`, where ``B = batch_size`` and ``C = frame_stack``.
            - embedding: :math:`(B, N)`, where ``N = embedding_size``
            模型训练好后怎么给rl用
        """
        # k和q怎么编码
        embedding= self.encoder(obs)

        return embedding

    def train(self, data: Dict) -> None:
        """
        Overview:
            Compute CURL contrastive learning loss to update encoder and W.
        Arguments:
            - data (:obj:`Dict`): CURL training data, including keywords ``obs_anchor`` (:obj:`torch.Tensor`) and \
                ``obs_positive`` (:obj:`torch.Tensor`).
        """
        # prepare pair data
        obs_anchor, obs_positive = data['obs_anchor'], data['obs_positive']

        # update_cpc
        z_a = self.CURL.encode(obs_anchor)
        z_pos = self.CURL.encode(obs_positive)
        logits = self.CURL.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)

        self.encoder_optimizer.zero_grad()
        self.cpc_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.cpc_optimizer.step()
        # if step % self.log_interval == 0:
        #     L.log('train/curl_loss', loss, step)


    def save(self, _state_dict: Dict, step) -> Dict:
        """
        Overview:
            Return state_dict including W and optimizer for saving checkpoint.
        """
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (_state_dict, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (_state_dict, step)
        )
        torch.save(
            self.CURL.state_dict(), '%s/curl_%s.pt' % (_state_dict, step)
        )

    def load(self, _state_dict: Dict,step) -> None:
        """
        Overview:
            Load state_dict including W and optimizer in order to recover.
        """
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (_state_dict, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (_state_dict, step))
        )
