import numpy as np
import copy
import torch
from torch import nn

from ding.utils import WORLD_MODEL_REGISTRY, lists_to_dicts
from ding.utils.data import default_collate
from ding.model import ConvEncoder, FCEncoder
from ding.world_model.base_world_model import WorldModel
from ding.world_model.model.networks import RSSM, ConvDecoder
from ding.torch_utils import to_device, one_hot
from ding.torch_utils.network.dreamer import DenseHead


@WORLD_MODEL_REGISTRY.register('dreamer')
class DREAMERWorldModel(WorldModel, nn.Module):
    config = dict(
        pretrain=100,
        train_freq=2,
        model=dict(
            state_size=None,
            action_size=None,
            model_lr=1e-4,
            reward_size=1,
            hidden_size=200,
            batch_size=256,
            max_epochs_since_update=5,
            dyn_stoch=32,
            dyn_deter=512,
            dyn_hidden=512,
            dyn_input_layers=1,
            dyn_output_layers=1,
            dyn_rec_depth=1,
            dyn_shared=False,
            dyn_discrete=32,
            act='SiLU',
            norm='LayerNorm',
            grad_heads=['image', 'reward', 'discount'],
            units=512,
            image_dec_layers=2,
            reward_layers=2,
            discount_layers=2,
            value_layers=2,
            actor_layers=2,
            cnn_depth=32,
            encoder_kernels=[4, 4, 4, 4],
            decoder_kernels=[4, 4, 4, 4],
            reward_head='twohot_symlog',
            kl_lscale=0.1,
            kl_rscale=0.5,
            kl_free=1.0,
            kl_forward=False,
            pred_discount=True,
            dyn_mean_act='none',
            dyn_std_act='sigmoid2',
            dyn_temp_post=True,
            dyn_min_std=0.1,
            dyn_cell='gru_layer_norm',
            unimix_ratio=0.01,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            obs_type='RGB',
            action_type='continuous',
            encoder_hidden_size_list=[64, 128, 128],
        ),
    )

    def __init__(self, cfg, env, tb_logger):
        WorldModel.__init__(self, cfg, env, tb_logger)
        nn.Module.__init__(self)

        self.pretrain_flag = True
        self._cfg = cfg.model
        #self._cfg.act = getattr(torch.nn, self._cfg.act),
        #self._cfg.norm = getattr(torch.nn, self._cfg.norm),
        self._cfg.act = nn.modules.activation.SiLU  # nn.SiLU
        self._cfg.norm = nn.modules.normalization.LayerNorm  # nn.LayerNorm
        self.state_size = self._cfg.state_size
        self.obs_type = self._cfg.obs_type
        self.action_size = self._cfg.action_size
        self.action_type = self._cfg.action_type
        self.reward_size = self._cfg.reward_size
        self.hidden_size = self._cfg.hidden_size
        self.batch_size = self._cfg.batch_size
        if self.obs_type == 'vector':
            self.encoder = FCEncoder(self.state_size, self._cfg.encoder_hidden_size_list, activation=torch.nn.SiLU())
            self.embed_size = self._cfg.encoder_hidden_size_list[-1]
        elif self.obs_type == 'RGB':
            self.encoder = ConvEncoder(
                self.state_size,
                hidden_size_list=[32, 64, 128, 256, 4096],  # to last layer 128?
                activation=torch.nn.SiLU(),
                kernel_size=self._cfg.encoder_kernels,
                layer_norm=True
            )
            self.embed_size = (
                (self.state_size[1] // 2 ** (len(self._cfg.encoder_kernels))) ** 2 * self._cfg.cnn_depth *
                2 ** (len(self._cfg.encoder_kernels) - 1)
            )

        self.dynamics = RSSM(
            self._cfg.dyn_stoch,
            self._cfg.dyn_deter,
            self._cfg.dyn_hidden,
            self._cfg.action_type,
            self._cfg.dyn_input_layers,
            self._cfg.dyn_output_layers,
            self._cfg.dyn_rec_depth,
            self._cfg.dyn_shared,
            self._cfg.dyn_discrete,
            self._cfg.act,
            self._cfg.norm,
            self._cfg.dyn_mean_act,
            self._cfg.dyn_std_act,
            self._cfg.dyn_temp_post,
            self._cfg.dyn_min_std,
            self._cfg.dyn_cell,
            self._cfg.unimix_ratio,
            self._cfg.action_size,
            self.embed_size,
            self._cfg.device,
        )
        self.heads = nn.ModuleDict()
        if self._cfg.dyn_discrete:
            feat_size = self._cfg.dyn_stoch * self._cfg.dyn_discrete + self._cfg.dyn_deter
        else:
            feat_size = self._cfg.dyn_stoch + self._cfg.dyn_deter

        if isinstance(self.state_size, int):
            self.heads['image'] = DenseHead(
                feat_size,
                (self.state_size, ),
                self._cfg.image_dec_layers,
                self._cfg.units,
                'SiLU',  # self._cfg.act
                'LN',  # self._cfg.norm
                dist='binary',
                outscale=0.0,
                device=self._cfg.device,
            )
        elif len(self.state_size) == 3:
            self.heads["image"] = ConvDecoder(
                feat_size,  # pytorch version
                self._cfg.cnn_depth,
                self._cfg.act,
                self._cfg.norm,
                self.state_size,
                self._cfg.decoder_kernels,
            )
        self.heads["reward"] = DenseHead(
            feat_size,  # dyn_stoch * dyn_discrete + dyn_deter
            (255, ),
            self._cfg.reward_layers,
            self._cfg.units,
            'SiLU',  # self._cfg.act
            'LN',  # self._cfg.norm
            dist=self._cfg.reward_head,
            outscale=0.0,
            device=self._cfg.device,
        )
        if self._cfg.pred_discount:
            self.heads["discount"] = DenseHead(
                feat_size,  # pytorch version
                [],
                self._cfg.discount_layers,
                self._cfg.units,
                'SiLU',  # self._cfg.act
                'LN',  # self._cfg.norm
                dist="binary",
                device=self._cfg.device,
            )

        if self._cuda:
            self.cuda()
        # to do
        # grad_clip, weight_decay
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self._cfg.model_lr)

    def step(self, obs, act):
        pass

    def eval(self, env_buffer, envstep, train_iter):
        pass

    def should_pretrain(self):
        if self.pretrain_flag:
            self.pretrain_flag = False
            return True
        return False

    def train(self, env_buffer, envstep, train_iter, batch_size, batch_length):
        self.last_train_step = envstep
        data = env_buffer.sample(
            batch_size, batch_length, train_iter
        )  # [len=B, ele=[len=T, ele={dict_key: Tensor(any_dims)}]]
        data = default_collate(data)  # -> [len=T, ele={dict_key: Tensor(B, any_dims)}]
        data = lists_to_dicts(data, recursive=True)  # -> {some_key: T lists}, each list is [B, some_dim]
        data = {k: torch.stack(data[k], dim=1) for k in data}  # -> {dict_key: Tensor([B, T, any_dims])}

        data['discount'] = data.get('discount', 1.0 - data['done'].float())
        data['weight'] = data.get('weight', None)
        if self.obs_type == 'RGB':
            data['image'] = data['obs'] - 0.5
        else:
            data['image'] = data['obs']
        if self.action_type == 'continuous':
            data['action'] *= (1.0 / torch.clip(torch.abs(data['action']), min=1.0))
        else:
            data['action'] = one_hot(data['action'], self.action_size)
        data = to_device(data, self._cfg.device)
        if len(data['reward'].shape) == 2:
            data['reward'] = data['reward'].unsqueeze(-1)
        if len(data['action'].shape) == 2:
            data['action'] = data['action'].unsqueeze(-1)
        if len(data['discount'].shape) == 2:
            data['discount'] = data['discount'].unsqueeze(-1)

        self.requires_grad_(requires_grad=True)

        image = data['image'].reshape([-1] + list(data['image'].shape[2:]))
        embed = self.encoder(image)
        embed = embed.reshape(list(data['image'].shape[:2]) + [embed.shape[-1]])

        post, prior = self.dynamics.observe(embed, data["action"])
        kl_loss, kl_value, loss_lhs, loss_rhs = self.dynamics.kl_loss(
            post, prior, self._cfg.kl_forward, self._cfg.kl_free, self._cfg.kl_lscale, self._cfg.kl_rscale
        )
        losses = {}
        likes = {}
        for name, head in self.heads.items():
            grad_head = name in self._cfg.grad_heads
            feat = self.dynamics.get_feat(post)
            feat = feat if grad_head else feat.detach()
            pred = head(feat)
            like = pred.log_prob(data[name])
            likes[name] = like
            losses[name] = -torch.mean(like)
        model_loss = sum(losses.values()) + kl_loss

        # ====================
        # world model update
        # ====================
        self.optimizer.zero_grad()
        model_loss.backward()
        self.optimizer.step()

        self.requires_grad_(requires_grad=False)
        # log
        if self.tb_logger is not None:
            for name, loss in losses.items():
                self.tb_logger.add_scalar(name + '_loss', loss.detach().cpu().numpy().item(), envstep)
        self.tb_logger.add_scalar('kl_free', self._cfg.kl_free, envstep)
        self.tb_logger.add_scalar('kl_lscale', self._cfg.kl_lscale, envstep)
        self.tb_logger.add_scalar('kl_rscale', self._cfg.kl_rscale, envstep)
        self.tb_logger.add_scalar('loss_lhs', loss_lhs.detach().cpu().numpy().item(), envstep)
        self.tb_logger.add_scalar('loss_rhs', loss_rhs.detach().cpu().numpy().item(), envstep)
        self.tb_logger.add_scalar('kl', torch.mean(kl_value).detach().cpu().numpy().item(), envstep)

        prior_ent = torch.mean(self.dynamics.get_dist(prior).entropy()).detach().cpu().numpy()
        post_ent = torch.mean(self.dynamics.get_dist(post).entropy()).detach().cpu().numpy()

        self.tb_logger.add_scalar('prior_ent', prior_ent.item(), envstep)
        self.tb_logger.add_scalar('post_ent', post_ent.item(), envstep)

        context = dict(
            embed=embed,
            feat=self.dynamics.get_feat(post),
            kl=kl_value,
            postent=self.dynamics.get_dist(post).entropy(),
        )
        post = {k: v.detach() for k, v in post.items()}
        return post, context

    def _save_states(self, ):
        self._states = copy.deepcopy(self.state_dict())

    def _save_state(self, id):
        state_dict = self.state_dict()
        for k, v in state_dict.items():
            if 'weight' in k or 'bias' in k:
                self._states[k].data[id] = copy.deepcopy(v.data[id])

    def _load_states(self):
        self.load_state_dict(self._states)

    def _save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                self._save_state(i)
                # self._save_state(i)
                updated = True
                # improvement = (best - current) / best

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        return self._epochs_since_update > self.max_epochs_since_update
