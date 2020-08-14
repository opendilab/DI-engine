import torch
import enum
import os
from nervex.torch_utils import one_hot
from pysc2.lib.static_data import NUM_BEGIN_ACTIONS, BEGIN_ACTIONS_REORDER, NUM_UNIT_BUILD_ACTIONS,\
    UNIT_BUILD_ACTIONS_REORDER, NUM_EFFECT_ACTIONS, EFFECT_ACTIONS_REORDER, NUM_RESEARCH_ACTIONS,\
    RESEARCH_ACTIONS_REORDER


class State(enum.IntEnum):
    init = 0,
    add = 1,


def remove_repeat_data(min_delay=16, max_move=3):
    def merge(selected_list):

        if len(selected_list) == 1:
            return selected_list

        def single_action_merge(start, end, check_delay=True):
            part = selected_list[start:end]
            if len(part) <= 1:
                return part
            actions = [p['actions'] for p in part]
            # high delay
            if check_delay:
                high_delay_step = [idx for idx, a in enumerate(actions) if a['delay'] >= min_delay]
                result = []
                cur = start
                for i in high_delay_step:
                    result.extend(single_action_merge(cur, start + i, False))
                    cur = start + i
                if cur < end:
                    result.extend(single_action_merge(cur, end, False))
            else:

                def equal(a, b):
                    if type(a) != type(b):
                        return False
                    if isinstance(a, torch.Tensor):
                        if a.shape != b.shape:
                            return False
                        return (a == b).all()
                    else:
                        return a == b

                # target units
                if isinstance(actions[0]['target_units'], torch.Tensor):
                    # same selected units and target_units
                    a0_s_units = actions[0]['selected_units']
                    not_same_s = [idx for idx, a in enumerate(actions) if not equal(a['selected_units'], a0_s_units)]
                    a0_t_units = actions[0]['target_units']
                    not_same_t = [idx for idx, a in enumerate(actions) if not equal(a['target_units'], a0_t_units)]
                    not_same = list(set(not_same_s).union(set(not_same_t)))
                    result = [part[0]]
                    if len(not_same) > 0:
                        # print('not same selected_units and target_units\n', actions)
                        result.extend(single_action_merge(start + not_same[0], end, False))
                # target location
                else:
                    # same selected_units units
                    a0_s_units = actions[0]['selected_units']
                    not_same = [idx for idx, a in enumerate(actions) if not equal(a['selected_units'], a0_s_units)]
                    if len(not_same) > 0:
                        # print('not same selected_units\n', actions, not_same)
                        result = []
                        result.extend(single_action_merge(start, start + not_same[0], False))
                        result.extend(single_action_merge(start + not_same[0] + 1, end, False))
                    else:
                        location = torch.stack([a['target_location'] for a in actions], dim=0).float()
                        x, y = torch.chunk(location, 2, dim=1)
                        x_flag = torch.abs(x - x.mean()).max() > max_move
                        y_flag = torch.abs(y - y.mean()).max() > max_move
                        if x_flag or y_flag:
                            result = [part[0], part[-1]]
                        else:
                            part[0]['actions']['target_location'] = torch.FloatTensor([x.mean(),
                                                                                       y.mean()]).round().long()  # noqa
                            result = [part[0]]
            return result

        start = 0
        start_action_type = selected_list[start]['actions']['action_type']
        result = []
        for idx in range(len(selected_list)):
            if start_action_type != selected_list[idx]['actions']['action_type']:
                result.extend(single_action_merge(start, idx))
                start = idx
                start_action_type = selected_list[start]['actions']['action_type']
        if start < len(selected_list):
            result.extend(single_action_merge(start, len(selected_list)))
        '''
        print('-'*60 + '\nnum:{}\n'.format(len(selected_list)))
        for item in selected_list:
            print(item['actions'])
        print('*'*60 + '\nnum:{}\n'.format(len(result)))
        for item in result:
            print(item['actions'])
        '''
        return result

    data_dir = '/mnt/lustre/niuyazhe/data/sl_data_R3'
    location_num = 8
    target_action_type_list = [168, 12, 1, 4]  # camera_move, smart_unit, smart_pt, attack_attack_pt
    for item in os.listdir(data_dir):
        name, suffix = item.split('.')
        if suffix == 'step' and name[:4] == 'Zerg':
            count = 0
            data = torch.load(os.path.join(data_dir, item))
            new_data = []
            state = State.init
            selected_list = []
            for step in data:
                action = step['actions']
                action_type = action['action_type']
                if state == State.init:
                    if action_type in target_action_type_list:
                        state = State.add
                        assert (len(selected_list) == 0)
                        selected_list.append(step)
                    else:
                        new_data.append(step)
                elif state == State.add:
                    if action_type in target_action_type_list:
                        selected_list.append(step)
                    else:
                        state = State.init
                        new_data.extend(merge(selected_list))
                        selected_list = []
                        new_data.append(step)
            torch.save(new_data, os.path.join(data_dir, name + '.step_processed'))
            meta = torch.load(os.path.join(data, name + '.meta'))
            meta['step_num'] = len(new_data)
            torch.save(meta, os.path.join(data, name + '.meta_processed'))
            print('replay: {}\ndata len: {}\tnew_data len: {}'.format(name, len(data), len(new_data)))
            count += 1
            if count % 10 == 0:
                print(count)


if __name__ == "__main__":
    remove_repeat_data()
