import torch
import os
from sc2learner.nn_utils import one_hot
from pysc2.lib.static_data import NUM_BEGIN_ACTIONS, BEGIN_ACTIONS_REORDER, NUM_UNIT_BUILD_ACTIONS, UNIT_BUILD_ACTIONS_REORDER, \
    NUM_EFFECT_ACTIONS, EFFECT_ACTIONS_REORDER, NUM_RESEARCH_ACTIONS, RESEARCH_ACTIONS_REORDER


def div_one_hot(v, max_val, ratio):
    num = int(max_val / ratio) + 1
    v = v.float()
    v = torch.floor(torch.clamp(v, 0, max_val) / ratio).long()
    return one_hot(v, num)


def reorder_one_hot(v, dictionary, num):
    assert(len(v.shape) == 1)
    assert(isinstance(v, torch.Tensor))
    new_v = torch.zeros_like(v)
    for idx in range(v.shape[0]):
        new_v[idx] = dictionary[v[idx].item()]
    return one_hot(new_v, num)


def binary_encode(v, bit_num):
    bin_v = '{:b}'.format(int(v))
    bin_v = [int(i) for i in bin_v]
    bit_diff = len(bin_v) - bit_num
    if bit_diff > 0:
        bin_v = bin_v[-bit_num:]
    elif bit_diff < 0:
        bin_v = [0 for _ in range(-bit_diff)] + bin_v
    return torch.FloatTensor(bin_v)


def batch_binary_encode(v, bit_num):
    B = v.shape[0]
    ret = []
    for b in range(B):
        ret.append(binary_encode(v[b], bit_num))
    return torch.stack(ret, dim=0)


def main():
    data_dir = '/mnt/lustre/niuyazhe/data/sl_data_test_multi_sp'
    location_num = 8
    for item in os.listdir(data_dir):
        name, suffix = item.split('.')
        if suffix == 'stat':
            stat = torch.load(os.path.join(data_dir, item))
            beginning_build_order = stat['begin_statistics']
            beginning_build_order_tensor = []
            for item in beginning_build_order:
                action_type, location = item['action_type'], item['location']
                action_type = torch.LongTensor([action_type])
                action_type = reorder_one_hot(action_type, BEGIN_ACTIONS_REORDER, num=NUM_BEGIN_ACTIONS)
                if location == 'none':
                    location = torch.zeros(location_num*2)
                else:
                    x = binary_encode(torch.LongTensor([location[0]]), bit_num=location_num)
                    y = binary_encode(torch.LongTensor([location[1]]), bit_num=location_num)
                    location = torch.cat([x, y], dim=0)
                beginning_build_order_tensor.append(torch.cat([action_type.squeeze(0), location], dim=0))
            beginning_build_order_tensor = torch.stack(beginning_build_order_tensor, dim=0)

            cumulative_stat = stat['cumulative_statistics']
            cumulative_stat_tensor = {
                'unit_build': torch.zeros(NUM_UNIT_BUILD_ACTIONS),
                'effect': torch.zeros(NUM_EFFECT_ACTIONS),
                'research': torch.zeros(NUM_RESEARCH_ACTIONS)
            }
            for k, v in cumulative_stat.items():
                if v['goal'] in ['unit', 'build']:
                    cumulative_stat_tensor['unit_build'][UNIT_BUILD_ACTIONS_REORDER[k]] = 1
                elif v['goal'] in ['effect']:
                    cumulative_stat_tensor['effect'][EFFECT_ACTIONS_REORDER[k]] = 1
                elif v['goal'] in ['research']:
                    cumulative_stat_tensor['research'][RESEARCH_ACTIONS_REORDER[k]] = 1
            for k, v in cumulative_stat_tensor.items():
                print(k, v.shape, v.sum())
            meta = torch.load(os.path.join(data_dir, name+'.meta'))
            mmr = meta['home_mmr']
            mmr = torch.LongTensor([mmr])
            mmr = div_one_hot(mmr, 6000, 1000).squeeze(0)
            torch.save({'mmr': mmr, 'beginning_build_order': beginning_build_order_tensor, 'cumulative_stat': cumulative_stat_tensor},
                       os.path.join(data_dir, name+'.stat_processed'))
            print(name)


if __name__ == "__main__":
    main()
