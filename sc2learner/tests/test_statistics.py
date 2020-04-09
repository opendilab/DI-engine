from sc2learner.envs.statistics import Statistics
import pytest
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import base64
import zlib


@pytest.fixture
def stat_processed():
    # produced by base64.b64encode(zlib.compress(pickle.dumps(torch.load(
    # r'Zerg_Terran_2479_0005e0d00cf4bca92d0432ecf23bd227337e39f7f8870a560ac3abfe7f89abc1.stat_processed'))))
    example_stat_processed = b'eJzt3ctO20AUBmA7CYS6pdBLSim90HvoJaXmLiEhoZZNRC/OxhtkBWdIXDk2x54gqISUjavSZ+gD9Ikq9RF4hEpddByDSoAUCYWQwP8tEskcPOOZc05WmVTjmySl9bgkSeWyR7LJXc8sZYwKt2xfMTy2XLHsgsGZ47uesaYqFEvvxPjiLV9kimG7+YKx4rllY3mDM1+h+HxKlqRq7Jti//m+8Hv4048Pg5lqbHE7Ll5r4/WL8VY9l7umaxtrzPMt1yFZBOi94i+2xbnNDOYUrLxDsa+6Ii7yjVVm+NZn5pOYcyKtd4mLfsn1OHVlY7UnsBxO3dmEngjv4TpFSmYTlYoYM60nw+BowiRFD6AsiInz3M5TkFwbWh2bGZtQJyfHJ6ZmVIpua65WKJ5NvuOU+ChutkTSIaFyPhOOUa86t/8KJQLq0sQspWwyEPOTA+rZMsUq2MzkYhF85b1XYB4rvLFMrtCFEY0UThc1uqSnxL8vs6LlOJZTNKJ9ccNg6i3F0qX4/M+lTl/08emxukVfnBpouOq1WLHqUwMHlh0ATszBrgYA0HzoNQDQKdCvAOAo6BMA0AroNQDQCug1zYO1BDjbwhrfW+f/q/nd2N2Ydu8P7T4/gPNmbw/Z33v2xjRjHAAAAGhP+JwGgFZArwGAVkCvAYBWQK8BADgIvREAWgG9BgBaAb0G4PxC/QNAq6HvAMBh0BsAoBOhdwGcfajz46DLAfVp1J+VsqnFIfkLXRGvWfF+dasUHhF/jdN1jVJ6n4g1K+WKnefWGjN8nud0Y5MG0rXD3CuOxaNz4+lmdGL8r1innxivTqmjdcf0rzc8ML4WKucz68fagqO+mN9s7VYozf5S8Ek/X3WOBgO6pdGQKJr1gG6HP7VwJyqXu5zuaTSsd4s4trLCTE73o4J42/kFMaFO1xVErnFBhKGiIHInuxPnEj0I6KFGj0Ty5QJ6HCbfkyj5nnJKazSi94goj/ksL7aUnkXpt93xP5uiTqrjdek32zj9wlCRfrOntEWn41/fo+cBvdDopUiR2YAyYYq8ilJklNNrjVSxB38BIhnvSw=='  # noqa
    example_stat_processed = pickle.loads(zlib.decompress(base64.b64decode(example_stat_processed)))
    return example_stat_processed


def compare(a, b, ignore_list=['mmr'], trace='ROOT'):
    print('Testing {}'.format(trace))
    if not type(a) == type(b):
        print(trace)
        print(a, b)
        return 1
    if isinstance(a, dict):
        for k, v in a.items():
            if k in ignore_list:
                continue
            if compare(a[k], b[k], trace=trace + ':' + str(k)):
                return 1
    elif isinstance(a, list) or isinstance(a, tuple):
        for n in range(len(a)):
            if compare(a[n], b[n], trace=trace + ':' + str(n)) and n not in ignore_list:
                return 1
    elif isinstance(a, torch.Tensor):
        if torch.abs(F.l1_loss(a.float(), b.float())) > 1e-7:
            print(trace)
            print('Tensor mismatch {} {}'.format(a, b))
            return 1
    elif isinstance(a, np.ndarray):
        if not (a == b).all():
            print(trace)
            print(a, b)
            print(np.argwhere(a != b))
            return 1
    else:
        if not a == b:
            print(trace)
            print(a, b)
            return 1
    return 0


def test_transformed_load_export(stat_processed):
    stat = Statistics(player_num=2)
    stat.load_from_transformed_stat(stat_processed, 0)
    assert stat.cached_transformed_stat[0] is not None
    assert stat.cached_transformed_stat[1] is None
    assert compare(stat_processed, stat.get_transformed_stat(player=0)) == 0
    stat.cached_transformed_stat[0] = None
    dumped = stat.get_transformed_stat(player=0)
    assert compare(stat_processed, dumped) == 0
    print(stat.build_order_statistics)
    print(stat.get_z(0))
