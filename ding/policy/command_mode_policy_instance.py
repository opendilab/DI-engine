from ding.utils import POLICY_REGISTRY
from ding.rl_utils import get_epsilon_greedy_fn
from .base_policy import CommandModePolicy

from .dqn import DQNPolicy
from .c51 import C51Policy
from .qrdqn import QRDQNPolicy
from .iqn import IQNPolicy
from .rainbow import RainbowDQNPolicy
from .r2d2 import R2D2Policy
from .sqn import SQNPolicy
from .ppo import PPOPolicy
from .ppg import PPGPolicy
from .a2c import A2CPolicy
from .impala import IMPALAPolicy
from .ddpg import DDPGPolicy
from .td3 import TD3Policy
from .sac import SACPolicy
from .qmix import QMIXPolicy
from .collaq import CollaQPolicy
from .coma import COMAPolicy
from .atoc import ATOCPolicy


class EpsCommandModePolicy(CommandModePolicy):

    def _init_command(self) -> None:
        r"""
        Overview:
            Command mode init method. Called by ``self.__init__``.
            Set the eps_greedy rule according to the config for command
        """
        eps_cfg = self._cfg.other.eps
        self.epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    def _get_setting_collect(self, command_info: dict) -> dict:
        r"""
        Overview:
            Collect mode setting information including eps
        Arguments:
            - command_info (:obj:`dict`): Dict type, including at least ['learner_step']
        Returns:
           - collect_setting (:obj:`dict`): Including eps in collect mode.
        """
        # Decay according to `learner_step`
        # step = command_info['learner_step']
        # Decay according to `envstep`
        step = command_info['envstep']
        return {'eps': self.epsilon_greedy(step)}

    def _get_setting_learn(self, command_info: dict) -> dict:
        return {}

    def _get_setting_eval(self, command_info: dict) -> dict:
        return {}


class DummyCommandModePolicy(CommandModePolicy):

    def _init_command(self) -> None:
        pass

    def _get_setting_collect(self, command_info: dict) -> dict:
        return {}

    def _get_setting_learn(self, command_info: dict) -> dict:
        return {}

    def _get_setting_eval(self, command_info: dict) -> dict:
        return {}


@POLICY_REGISTRY.register('dqn_command')
class DQNCommandModePolicy(DQNPolicy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('c51_command')
class C51CommandModePolicy(C51Policy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('qrdqn_command')
class QRDQNCommandModePolicy(QRDQNPolicy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('iqn_command')
class IQNCommandModePolicy(IQNPolicy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('rainbow_command')
class RainbowDQNCommandModePolicy(RainbowDQNPolicy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('r2d2_command')
class R2D2CommandModePolicy(R2D2Policy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('sqn_command')
class SQNCommandModePolicy(SQNPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('ppo_command')
class PPOCommandModePolicy(PPOPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('a2c_command')
class A2CCommandModePolicy(A2CPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('impala_command')
class IMPALACommandModePolicy(IMPALAPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('ppg_command')
class PPGCommandModePolicy(PPGPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('ddpg_command')
class DDPGCommandModePolicy(DDPGPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('td3_command')
class TD3CommandModePolicy(TD3Policy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('sac_command')
class SACCommandModePolicy(SACPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('qmix_command')
class QMIXCommandModePolicy(QMIXPolicy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('collaq_command')
class CollaQCommandModePolicy(CollaQPolicy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('coma_command')
class COMACommandModePolicy(COMAPolicy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('atoc_command')
class ATOCCommandModePolicy(ATOCPolicy, DummyCommandModePolicy):
    pass
