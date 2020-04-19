import os
import sys
import subprocess
import torch
import zlib
import pickle


# get all info of cluster
def get_cls_info():
    ret_dict = {}
    info = subprocess.getoutput('sinfo -Nh').split('\n')
    for line in info:
        line = line.strip().split()
        if len(line) != 4:
            continue
        node, _, partition, state = line
        if partition not in ret_dict:
            ret_dict[partition] = []
        assert node not in ret_dict[partition]
        if state in ['idle', 'mix']:
            ret_dict[partition].append([node, state])

    return ret_dict


def generate():
    p = 'nohup srun -p {} -w {} python -u replay_fix.py {} > {} 2>&1 &'
    list_p = '/mnt/lustre/zhangming/data/listtempfix_2/todo.list.'
    log_p = '/mnt/lustre/zhangming/data/logfix_2/log.'
    d = get_cls_info()
    partitions = ['VI_SP_Y_V100_A', 'VI_SP_VA_V100', 'VI_SP_VA_1080TI', 'VI_ID_1080TI']

    count = 0
    for partition in partitions:
        workstations = d[partition]
        for workstation_state in workstations:
            workstation = workstation_state[0]
            print(p.format(partition, workstation, list_p + "{:02d}".format(count), log_p + "{:02d}".format(count)))
            count += 1
            if count >= 50:
                exit()


def get_dict():
    p = '/mnt/lustre/zhangming/data/replay_decode_410_clean_2.list'
    save_path = '/mnt/lustre/zhangming/data/replay_decode_410_clean_2.meta'
    from sc2learner.utils import read_file_ceph
    import torch
    result = {}
    lines = open(p, 'r').readlines()
    for index, line in enumerate(lines):
        if index % 1000 == 0:
            print(index)
        meta = torch.load(read_file_ceph(line.strip() + ".meta"))
        del meta['replay_path']
        k = line.strip().split('/')[-1]
        assert k not in result
        result[k] = meta
    torch.save(result, save_path)


def filter():
    p = '/mnt/lustre/zhangming/data/replay_decode_410_clean_2.meta'
    save_path = '/mnt/lustre/zhangming/data/Zerg_Zerg_KJ.list'
    import torch
    metas = torch.load(p)
    prefix = 's3://replay_decode_410_clean_2/'
    result = []
    for k, v in metas.items():
        if v['map_name'] in ['Kairos Junction LE', 'KairosJunction'] \
           and v['home_race'] == 'Zerg' and v['away_race'] == 'Zerg':
            result.append(prefix + k)
    with open(save_path, 'w') as f:
        for item in result:
            f.write(item + '\n')


def process():
    p = '/mnt/lustre/zhangming/data/replay_decode_410_clean_2.list'
    meta_file = '/mnt/lustre/zhangming/data/replay_decode_410_clean_2.meta'
    import torch
    metas = torch.load(meta_file)
    from sc2learner.utils import read_file_ceph
    lines = open(p, 'r').readlines()
    count = 0
    for line in lines:
        name = line.strip().split('/')[-1]
        if name in metas:
            home_race = metas[name]['home_race']
            if home_race == 'Protoss':
                print(line.strip())
                count += 1
    print(count)


def zlib_decompressor_save(local_path, save_path):
    data = torch.load(local_path)
    new_data = []
    for item in data:
        new_data.append(pickle.loads(zlib.decompress(item)))
    torch.save(new_data, save_path)


def decompress(todolist):
    # import multiprocessing
    local_prefix = '/mnt/lustre/zhangming/data/replay_decode_410_clean_compress/'
    save_prefix = '/mnt/lustre/zhangming/data/replay_decode_410_clean_decompress/'
    # pool = multiprocessing.Pool(1)
    lines = open(todolist, 'r').readlines()
    for index, line in enumerate(lines):
        if index % 10 == 0:
            print(index)
        p1 = line.strip().split('/')[-1]
        local_path = local_prefix + p1 + ".step"
        save_path = save_prefix + p1 + ".step"
        # r = pool.apply_async(zlib_decompressor, args=(local_path, save_path))
        zlib_decompressor_save(local_path, save_path)
    # pool.close()
    # pool.join()


def clean(todolist):
    from sc2learner.envs import action_unit_id_transform
    local_prefix = '/mnt/lustre/zhangming/data/replay_decode_410_clean_compress/'
    save_prefix = '/mnt/lustre/zhangming/data/replay_decode_410_clean_decompress/'

    lines = open(todolist, 'r').readlines()
    for index, line in enumerate(lines):
        if index % 10 == 0:
            print(index)
        p1 = line.strip().split('/')[-1]
        local_path = local_prefix + p1 + ".step"
        save_path = save_prefix + p1 + ".step"
        data = torch.load(local_path)
        new_data = []
        for item in data:
            new_data.append(pickle.loads(zlib.decompress(item)))
        try:
            sample_data = action_unit_id_transform(new_data)
        except:
            print('[bad replay] {}'.format(line))
        finally:
            torch.save(new_data, save_path)


def clean_for_manyuan(todolist):
    def check_steps(replay):
        new_replay = []
        for i, step in enumerate(replay):
            entity_raw = step['entity_raw']
            actions = step['actions']
            id_list = entity_raw['id']
            flag = True
            if isinstance(actions['selected_units'], torch.Tensor):
                for val in actions['selected_units']:
                    if val not in id_list:
                        flag = False
                        break
            if isinstance(actions['target_units'], torch.Tensor):
                for val in actions['target_units']:
                    if val not in id_list:
                        flag = False
                        break
            if flag:
                new_replay.append(step)
        # print("bad_vals = {} {} \n delete from {} to {}".format(bad_vals, len(bad_vals), len(replay), len(new_replay)))
        # print("selected_units {}, target_units {}".format(count1, count2))
        return new_replay

    from sc2learner.envs import action_unit_id_transform
    save_prefix = '/mnt/lustre/zhangming/data/replay_decode_410_clean_manyuan/'

    lines = open(todolist, 'r').readlines()
    for index, line in enumerate(lines):
        if index % 10 == 0:
            print(index)
        name = line.strip().split('/')[-1]
        local_path = line.strip() + ".step"
        save_path = os.path.join(save_prefix, name + ".step")
        data = torch.load(local_path)
        new_data = check_steps(data)
        # try:
        #     sample_data = action_unit_id_transform(new_data)
        # except:
        #     print('[bad replay] {}'.format(line.strip()))
        # else:
        torch.save(new_data, save_path)


def auto():
    workers_num_std = 1
    workers_num_my = 1
    # todolist = '/mnt/lustre/zhangming/data/Zerg_None_None_3500_train_5200.clean_2.txt'
    todolist = '/mnt/lustre/zhangming/data/todo.list'
    save_dir = '/mnt/lustre/zhangming/data/replay_decode_410_clean_decompress/'
    list_dir = '/mnt/lustre/zhangming/data/list_decompress_temp'
    delete_workstations = []
    partitions = ['VI_SP_Y_V100_A', 'VI_SP_Y_V100_B', 'VI_SP_VA_V100', 'VI_ID_1080TI', 'VI_Face_1080TI']
    info = get_cls_info()
    total_list_num = 0
    for k, v in info.items():
        if k in partitions:
            total_list_num += len(v)
    replays = open(todolist, 'r').readlines()
    print('total: {}'.format(len(replays)))
    replays = [x.strip() for x in replays]
    print('todo: {}'.format(len(replays)))
    len_per_workstation = len(replays) // total_list_num + 1
    print('split into {} lists, each list has {} items'.format(total_list_num, len_per_workstation))
    replays_index = 0

    for partition in partitions:
        replay_list_path = os.path.join(list_dir, partition)
        if not os.path.isdir(replay_list_path):
            os.mkdir(replay_list_path)
        workstations = []
        if partition != 'VI_SP_Y_V100_A':
            workers_num_std_spe = workers_num_std
            workstations = info[partition]
        else:
            workers_num_std_spe = workers_num_my
            workstations = info[partition]
        for workstation_state in workstations:
            workstation = workstation_state[0]
            state = workstation_state[0]
            if workstation not in delete_workstations:
                replay_list_path_now = os.path.join(replay_list_path, workstation)
                with open(replay_list_path_now, 'w') as f:
                    for replay in replays[replays_index * len_per_workstation:(replays_index + 1) *
                                          len_per_workstation]:
                        f.write(replay + '\n')
                print(
                    'srun -p {} -w {} python -u test.py clean {} >> /mnt/lustre/zhangming/data/clean_2.log 2>&1 &'.
                    format(partition, workstation, replay_list_path_now)
                )
                replays_index += 1


def auto_clean_for_manyuan():
    workers_num_std = 1
    workers_num_my = 1
    # todolist = '/mnt/lustre/zhangming/data/Zerg_None_None_3500_train_5200.clean_2.txt'
    # todolist = '/mnt/lustre/zhangmanyuan/nature-agi/sl_data/0412.finish.list'
    todolist = '/mnt/lustre/zhangmanyuan/nature-agi/sl_data/Protoss_1708.txt'
    list_dir = '/mnt/lustre/zhangming/data/list_for_manyuan'
    delete_workstations = []
    partitions = ['VI_SP_Y_V100_A', 'VI_SP_Y_V100_B', 'VI_SP_VA_V100', 'VI_ID_1080TI', 'VI_Face_1080TI']
    info = get_cls_info()
    total_list_num = 0
    for k, v in info.items():
        if k in partitions:
            total_list_num += len(v)
    replays = open(todolist, 'r').readlines()
    print('total: {}'.format(len(replays)))
    replays = [x.strip() for x in replays]
    print('todo: {}'.format(len(replays)))
    len_per_workstation = len(replays) // total_list_num + 1
    print('split into {} lists, each list has {} items'.format(total_list_num, len_per_workstation))
    replays_index = 0

    for partition in partitions:
        replay_list_path = os.path.join(list_dir, partition)
        if not os.path.isdir(replay_list_path):
            os.mkdir(replay_list_path)
        workstations = []
        if partition != 'VI_SP_Y_V100_A':
            workers_num_std_spe = workers_num_std
            workstations = info[partition]
        else:
            workers_num_std_spe = workers_num_my
            workstations = info[partition]
        for workstation_state in workstations:
            if replays_index < total_list_num:
                workstation = workstation_state[0]
                state = workstation_state[0]
                if workstation not in delete_workstations:
                    replay_list_path_now = os.path.join(replay_list_path, workstation)
                    with open(replay_list_path_now, 'w') as f:
                        for replay in replays[replays_index * len_per_workstation:(replays_index + 1) *
                                              len_per_workstation]:
                            f.write(replay + '\n')
                    print(
                        'srun -p {} -w {} python -u test.py clean_for_manyuan {} >> /mnt/lustre/zhangming/data/clean_for_manyuan.log.2 2>&1 &'
                        .format(partition, workstation, replay_list_path_now)
                    )
                    replays_index += 1


def process_0412():
    p = '/mnt/lustre/zhangming/data/0412.list'
    t = '/mnt/lustre/zhangming/data/listtemp410/replays.raw.list.5200'
    save_path = '/mnt/lustre/zhangming/data/0412.todo.list'
    prefix = '/mnt/lustre/zhangmanyuan/data_t1/nature-agi/replay_4.10.0/'
    lines = open(p, 'r').readlines()
    templates = open(t, 'r').readlines()
    templates = [x.strip().split('/')[-1].split('.')[0] for x in templates]
    templates_dict = {}
    for template in templates:
        templates_dict[template] = 1
    print(len(templates_dict))

    p_dict = {}
    for line in lines:
        n = line.strip().split('.')[0].split('_')[-1]
        if n not in p_dict:
            p_dict[n] = 0
        p_dict[n] += 1
    print(len(p_dict))

    p_dict_uniq = {}
    for k, v in p_dict.items():
        if v == 8:
            p_dict_uniq[k] = v
    print(len(p_dict_uniq))

    with open(save_path, 'w') as f:
        for k, v in templates_dict.items():
            if k not in p_dict_uniq:
                f.write(prefix + k.strip() + ".SC2Replay\n")


def process_0412_2():
    p = '/mnt/lustre/zhangming/data/0412.list'
    save_path = '/mnt/lustre/zhangming/data/0412.finish.list'
    prefix = '/mnt/lustre/zhangming/data/replays_decode_410/'
    lines = open(p, 'r').readlines()

    p_dict = {}
    for line in lines:
        n = line.strip().split('.')[0]
        if n not in p_dict:
            p_dict[n] = 0
        p_dict[n] += 1
    print(len(p_dict))

    p_dict_uniq = {}
    for k, v in p_dict.items():
        if v == 4:
            p_dict_uniq[k] = v
    print(len(p_dict_uniq))

    with open(save_path, 'w') as f:
        for k, v in p_dict_uniq.items():
            f.write(prefix + k.strip() + "\n")


def process_0412_3():
    p = '/mnt/lustre/zhangmanyuan/nature-agi/sl_data/Protoss_1708.txt'
    save_path = '/mnt/lustre/zhangming/data/Protoss_1708.result'
    prefix = '/mnt/lustre/zhangming/data/replay_decode_410_clean_manyuan/'
    lines = open(p, 'r').readlines()
    f = open(save_path, 'w')

    for index, line in enumerate(lines):
        print(index)
        x = line.strip().split('/')[-1]
        meta = torch.load(prefix + x + '.meta')
        win = 1 if meta['home_result'] == 1 else 0
        f.write(line.strip() + " " + str(win) + "\n")


def process_0415():
    meta = 's3://replay_decode_48/Protoss_Zerg_0_00175666484452f92efe9eba42be4ca278faaace895241e9a13532b0a4b6317a.meta'
    from sc2learner.utils import read_file_ceph
    # meta = torch.load(read_file_ceph(meta))
    meta = read_file_ceph(meta, read_type='pickle')
    print(meta)


class A:
    def __init__(self, func1):
        self.func1 = func1

    def func(self):
        self.func1('aaab')


class B:
    def __init__(self):
        self.password = '123456'

        def f(x, y=1):
            if x == 'aaa':
                x = x + ' - ' + self.password
            else:
                x = x + ' - not aaa'
            print(x, y)

        self.a = A(f)

    def go(self):
        self.a.func()


def process_0419():
    info = get_cls_info()
    count = 0
    for k, v in info.items():
        if k == 'VI_SP_Y_V100_A':
            for workstation, state in v:
                if count < 12:
                    print('sh run_coordinator.sh {} {} &'.format(k, workstation))


if __name__ == '__main__':
    # get_dict()
    # filter()
    # process()
    func_name = sys.argv[1]
    if func_name == 'process':
        process()
    elif func_name == 'auto':
        auto()
    elif func_name == 'auto_clean_for_manyuan':
        auto_clean_for_manyuan()
    elif func_name == 'decompress':
        todolist = sys.argv[2]
        decompress(todolist)
    elif func_name == 'clean':
        todolist = sys.argv[2]
        clean(todolist)
    elif func_name == 'clean_for_manyuan':
        todolist = sys.argv[2]
        clean_for_manyuan(todolist)
    elif func_name == 'process_0412':
        process_0412_3()
    elif func_name == 'process_0415':
        process_0415()
    elif func_name == 'testb':
        b = B()
        b.go()
    elif func_name == 'process_0419':
        process_0419()
