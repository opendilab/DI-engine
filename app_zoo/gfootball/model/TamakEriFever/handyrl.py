# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import sys
import yaml


if __name__ == '__main__':
    with open('config.yaml') as f:
        args = yaml.safe_load(f)
    print(args)

    if len(sys.argv) < 2:
        print('Please set mode of handyrl.')
        exit(1)

    mode = sys.argv[1]

    if mode == '--train' or mode == '-t':
        from handyrl_core.train import train_main as main
        main(args)
    if mode == '--train-server' or mode == '-ts':
        from handyrl_core.train import train_server_main as main
        main(args)
    elif mode == '--worker' or mode == '-w':
        from handyrl_core.worker import worker_main as main
        main(args)
    elif mode == '--eval' or mode == '-e':
        from handyrl_core.evaluation import eval_main as main
        main(args, sys.argv[2:])
    elif mode == '--eval-server' or mode == '-es':
        from handyrl_core.evaluation import eval_server_main as main
        main(args, sys.argv[2:])
    elif mode == '--eval-client' or mode == '-ec':
        from handyrl_core.evaluation import eval_client_main as main
        main(args, sys.argv[2:])
    else:
        print('Not found mode %s.' % mode)
