import torch
import torch.nn as nn
import torch.nn.functional as F
from sc2learner.nn_utils import Transformer, fc_block, build_activation


class EntityEncoder(nn.Module):
    def __init__(self, cfg):
        super(EntityEncoder, self).__init__()
        self.act = build_activation(cfg.activation)
        self.transformer = Transformer(
            input_dim=cfg.input_dim,
            head_dim=cfg.head_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim,
            head_num=cfg.head_num,
            mlp_num=cfg.mlp_num,
            layer_num=cfg.layer_num,
            dropout_ratio=cfg.dropout_ratio,
            activation=self.act,
        )
        self.entity_fc = fc_block(cfg.output_dim, cfg.output_dim, activation=self.act)
        self.embed_fc = fc_block(cfg.output_dim, cfg.output_dim, activation=self.act)

    def forward(self, x):
        '''
        Input:
            x: [batch_size, entity_num, input_dim]
        Output:
            entity_embeddings: [batch_size, entity_num, output_dim]
            embedded_entity: [batch_size, output_dim]
        '''
        x = self.act(self.transformer(x))
        entity_embeddings = self.entity_fc(x)
        embedded_entity = self.embed_fc(x.mean(dim=1))  # TODO masked
        return entity_embeddings, embedded_entity


def transform_entity_data(entity_list, pad_value=-1e9):
    template = [
        {'key': 'unit_type', 'dim': 256, 'other': 'one-hot'},
        {'key': 'unit_attr', 'dim': 13, 'other': 'each one boolean'},
        {'key': 'alliance', 'dim': 5, 'other': 'one-hot'},
        {'key': 'current_health', 'dim': 38, 'other': 'one-hot, sqrt(1500), floor'},  # doubt
        {'key': 'current_shields', 'dim': 31, 'other': 'one-hot, sqrt(1000), floor'},  # doubt
        {'key': 'current_energy', 'dim': 14, 'other': 'one-hot, sqrt(200), floor'},  # doubt
        {'key': 'cargo_space_used', 'dim': 9, 'other': 'one-hot'},
        {'key': 'cargo_space_maximum', 'dim': 9, 'other': 'one-hot'},
        {'key': 'build_progress', 'dim': 1, 'other': 'float [0, 1]'},
        {'key': 'current_health_ratio', 'dim': 1, 'other': 'float [0, 1]'},
        {'key': 'current_shield_ratio', 'dim': 1, 'other': 'float [0, 1]'},
        {'key': 'current_energy_ratio', 'dim': 1, 'other': 'float [0, 1]'},
        {'key': 'display_type', 'dim': 5, 'other': 'one-hot'},
        {'key': 'x_position', 'dim': 1, 'other': 'binary encoding'},  # doubt
        {'key': 'y_position', 'dim': 1, 'other': 'binary encoding'},  # doubt
        {'key': 'is_cloaked', 'dim': 5, 'other': 'one-hot'},
        {'key': 'is_powered', 'dim': 2, 'other': 'one-hot'},
        {'key': 'is_hallucination', 'dim': 2, 'other': 'one-hot'},
        {'key': 'is_active', 'dim': 2, 'other': 'one-hot'},
        {'key': 'is_on_screen', 'dim': 2, 'other': 'one-hot'},
        {'key': 'is_in_cargo', 'dim': 2, 'other': 'one-hot'},
        {'key': 'current_minerals', 'dim': 19, 'other': 'one-hot, 1900/100'},
        {'key': 'current_vespene', 'dim': 26, 'other': 'one-hot, 2600/100'},
        {'key': 'mined_minerals', 'dim': 42, 'other': 'one-hot, sqrt(1800), floor'},
        {'key': 'mined_vespene', 'dim': 50, 'other': 'one-hot, sqrt(2500), floor'},
        {'key': 'assigned_harvesters', 'dim': 24, 'other': 'one-hot'},
        {'key': 'ideal_harvesters', 'dim': 17, 'other': 'one-hot'},
        {'key': 'weapon_cooldown', 'dim': 32, 'other': 'one-hot, game steps'},
        {'key': 'order_queue_length', 'dim': 9, 'other': 'one-hot'},
        {'key': 'order_1', 'dim': 1, 'other': 'one-hot'},  # TODO
        {'key': 'order_2', 'dim': 1, 'other': 'one-hot'},  # TODO
        {'key': 'order_3', 'dim': 1, 'other': 'one-hot'},  # TODO
        {'key': 'order_4', 'dim': 1, 'other': 'one-hot'},  # TODO
        {'key': 'buffers', 'dim': 2, 'other': 'each one boolean'},  # doubt
        {'key': 'addon_type', 'dim': 1, 'other': 'one-hot'},  # TODO
        {'key': 'order_progress_1', 'dim': 11, 'other': '1+10, float([0, 1]) + one-hot(100/10)'},  # doubt
        {'key': 'order_progress_2', 'dim': 11, 'other': '1+10, float([0, 1]) + one-hot(100/10)'},  # doubt
        {'key': 'weapon_upgrades', 'dim': 4, 'other': 'one-hot'},
        {'key': 'armor_upgrades', 'dim': 4, 'other': 'one-hot'},
        {'key': 'shield_upgrades', 'dim': 4, 'other': 'one-hot'},
        {'key': 'was_selected', 'dim': 2, 'other': 'one-hot, last action'},
        {'key': 'was_targeted', 'dim': 2, 'other': 'one-hot, last action'},
    ]


def test_entity_encoder():
    class CFG(object):
        def __init__(self):
            self.input_dim = 256  # origin 512
            self.head_dim = 128
            self.hidden_dim = 1024
            self.output_dim = 256
            self.head_num = 2
            self.mlp_num = 2
            self.layer_num = 3
            self.dropout_ratio = 0.1
            self.activation = 'relu'

    model = EntityEncoder(CFG()).cuda()
    input = torch.randn(2, 14, 256).cuda()
    entity_embeddings, embedded_entity = model(input)
    print(model)
    print(entity_embeddings.shape)
    print(embedded_entity.shape)


if __name__ == "__main__":
    test_entity_encoder()
