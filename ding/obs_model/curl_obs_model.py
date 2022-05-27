from typing import TYPE_CHECKING, Dict
# easydict的作用：可以使得以属性的方式去访问字典的值
from easydict import EasyDict
import copy
import torch
import torch.nn as nn
import os
# 为了防止循环引用出现的差错
if TYPE_CHECKING:
    from tensorboardX import SummaryWriter

# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}

class Encoder(nn.Module):
    def __init__(self, obs_shape, encoder_feature_size, num_layers=2, num_filters=32):
        super().__init__()
        # x : shape : [B, C, H, W]
        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.encoder_feature_size = encoder_feature_size
        print(encoder_feature_size)
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)] #??
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        out_dim = OUT_DIM[num_layers] #39
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.encoder_feature_size)
        self.ln = nn.LayerNorm(self.encoder_feature_size)
        self.outputs = dict() # initialize

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs
        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv
        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv
        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)
        if detach:
            h = h.detach() #切断分支反向传播
        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc
        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm
        out = torch.tanh(h_norm)
        self.outputs['tanh'] = out
        return out

class CurlObsModel(nn.Module):
    @classmethod
    def default_config(cls: type) -> EasyDict:
        # copy.deepcopy 深拷贝 拷贝对象及其子对象
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg
    config = dict(
        batch_size=64,
        encoder_lr=1e-3,
        w_lr=1e-4,
        target_theta=0.005,
        encoder_feature_size=50,
        frame_stack = 3,
        image_size = 84,
        num_layers = 2,
        num_filters = 32,
        train_steps = 10,
    )

    def __init__(
            self, cfg: EasyDict, tb_logger: "SummaryWriter") -> None:
        # delete encoder & encoder_target: nn.Module,
        """
        Overview:
            Create main components, such as paramater, optimizer, loss function and so on.
        """
        super(CurlObsModel, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.cfg = cfg
        self.obs_shape = (3*cfg.frame_stack, cfg.image_size, cfg.image_size)
        self.encoder = Encoder(self.obs_shape, cfg.encoder_feature_size, cfg.num_layers, cfg.num_filters)
        # self.encoder_target = encoder_target
        self.tb_logger = tb_logger

        # parameter
        self.W = nn.Parameter(torch.rand(self.cfg.encoder_feature_size, self.cfg.encoder_feature_size))
        # optimizer
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=self.cfg.encoder_lr
        )

        self.cpc_optimizer = torch.optim.Adam(
            [self.W], lr=self.cfg.w_lr,
        )
        # loss function
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def compute_logits(self, z_anc, z_pos):
        """
            Uses logits trick for CURL:
            - compute (B,B) matrix z_anc (W z_pos.T)
            - positives are all diagonal elements
            - negatives are all other elements
            - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        proj_k = torch.matmul(self.W, z_pos.T)  # (encoder_feature_size, B)
        logits = torch.matmul(z_anc, proj_k)  # (B,B)
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
        """
        embedding = self.encoder(obs)
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
        z_anc = self.encode(obs_anchor)
        z_pos = self.encode(obs_positive)
        logits = self.compute_logits(z_anc, z_pos)

        labels = torch.arange(logits.shape[0]).long() #.to(self.device) 有two devices 报错
        loss = self.cross_entropy_loss(logits, labels)

        self.encoder_optimizer.zero_grad() # 将模型的参数梯度初始化为0
        self.cpc_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.cpc_optimizer.step() # 更新所有参数
        # if step % self.log_interval == 0:
        #     L.log('train/curl_loss', loss, step)

    def save(self) -> Dict:
        """
        Overview:
            Returneezs state_dict including W and optimizer for saving checkpoint.
        """
        torch.save(
            self.encoder.state_dict(), 'curl.pt'
        )


    def load(self) -> None:
        """
        Overview:
            Load state_dict including W and optimizer in order to recover.
        """
        self.encoder.load_state_dict(
            torch.load('curl.pt')
        )

