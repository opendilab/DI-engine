from typing import List, Union
import os
import copy
import click
from click.core import Context, Option
import numpy as np

from ding import __TITLE__, __VERSION__, __AUTHOR__, __AUTHOR_EMAIL__
from ding.config import read_config
from .predefined_config import get_predefined_config


def print_version(ctx: Context, param: Option, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    click.echo('{title}, version {version}.'.format(title=__TITLE__, version=__VERSION__))
    click.echo('Developed by {author}, {email}.'.format(author=__AUTHOR__, email=__AUTHOR_EMAIL__))
    ctx.exit()


def print_registry(ctx: Context, param: Option, value: str):
    if value is None:
        return
    from ding.utils import registries  # noqa
    if value not in registries:
        click.echo('[ERROR]: not support registry name: {}'.format(value))
    else:
        registered_info = registries[value].query_details()
        click.echo('Available {}: [{}]'.format(value, '|'.join(registered_info.keys())))
        for alias, info in registered_info.items():
            click.echo('\t{}: registered at {}#{}'.format(alias, info[0], info[1]))
    ctx.exit()


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    '-v',
    '--version',
    is_flag=True,
    callback=print_version,
    expose_value=False,
    is_eager=True,
    help="Show package's version information."
)
@click.option(
    '-q',
    '--query-registry',
    type=str,
    callback=print_registry,
    expose_value=False,
    is_eager=True,
    help='query registered module or function, show name and path'
)
@click.option(
    '-m',
    '--mode',
    type=click.Choice(
        [
            'serial',
            'serial_onpolicy',
            'serial_sqil',
            'serial_dqfd',
            'serial_trex',
            'serial_trex_onpolicy',
            'parallel',
            'dist',
            'eval',
            'serial_reward_model',
            'serial_gail',
            'serial_offline',
            'serial_ngu',
        ]
    ),
    help='serial-train or parallel-train or dist-train or eval'
)
@click.option('-c', '--config', type=str, help='Path to DRL experiment config')
@click.option(
    '-s',
    '--seed',
    type=int,
    default=[0],
    multiple=True,
    help='random generator seed(for all the possible package: random, numpy, torch and user env)'
)
@click.option('-e', '--env', type=str, help='RL env name')
@click.option('-p', '--policy', type=str, help='DRL policy name')
@click.option('--exp-name', type=str, help='experiment directory name')
@click.option('--train-iter', type=str, default='1e8', help='Maximum policy update iterations in training')
@click.option('--env-step', type=str, default='1e8', help='Maximum collected environment steps for training')
@click.option('--load-path', type=str, default=None, help='Path to load ckpt')
@click.option('--replay-path', type=str, default=None, help='Path to save replay')
# the following arguments are only applied to dist mode
@click.option('--enable-total-log', type=bool, help='whether enable the total DI-engine system log', default=False)
@click.option('--disable-flask-log', type=bool, help='whether disable flask log', default=True)
@click.option(
    '-P', '--platform', type=click.Choice(['local', 'slurm', 'k8s']), help='local or slurm or k8s', default='local'
)
@click.option(
    '-M',
    '--module',
    type=click.Choice(['config', 'collector', 'learner', 'coordinator', 'learner_aggregator', 'spawn_learner']),
    help='dist module type'
)
@click.option('--module-name', type=str, help='dist module name')
@click.option('-cdh', '--coordinator-host', type=str, help='coordinator host', default='0.0.0.0')
@click.option('-cdp', '--coordinator-port', type=int, help='coordinator port')
@click.option('-lh', '--learner-host', type=str, help='learner host', default='0.0.0.0')
@click.option('-lp', '--learner-port', type=int, help='learner port')
@click.option('-clh', '--collector-host', type=str, help='collector host', default='0.0.0.0')
@click.option('-clp', '--collector-port', type=int, help='collector port')
@click.option('-agh', '--aggregator-host', type=str, help='aggregator slave host', default='0.0.0.0')
@click.option('-agp', '--aggregator-port', type=int, help='aggregator slave port')
@click.option('--add', type=click.Choice(['collector', 'learner']), help='add replicas type')
@click.option('--delete', type=click.Choice(['collector', 'learner']), help='delete replicas type')
@click.option('--restart', type=click.Choice(['collector', 'learner']), help='restart replicas type')
@click.option('--kubeconfig', type=str, default=None, help='the path of Kubernetes configuration file')
@click.option('-cdn', '--coordinator-name', type=str, default=None, help='coordinator name')
@click.option('-ns', '--namespace', type=str, default=None, help='job namespace')
@click.option('-rs', '--replicas', type=int, default=1, help='number of replicas to add/delete/restart')
@click.option('-rpn', '--restart-pod-name', type=str, default=None, help='restart pod name')
@click.option('--cpus', type=int, default=0, help='The requested CPU, read the value from DIJob yaml by default')
@click.option('--gpus', type=int, default=0, help='The requested GPU, read the value from DIJob yaml by default')
@click.option(
    '--memory', type=str, default=None, help='The requested Memory, read the value from DIJob yaml by default'
)
@click.option(
    '--profile',
    type=str,
    default=None,
    help='profile Time cost by cProfile, and save the files into the specified folder path'
)
def cli(
    # serial/eval
    mode: str,
    config: str,
    seed: Union[int, List],
    exp_name: str,
    env: str,
    policy: str,
    train_iter: str,  # transform into int
    env_step: str,  # transform into int
    load_path: str,
    replay_path: str,
    # parallel/dist
    platform: str,
    coordinator_host: str,
    coordinator_port: int,
    learner_host: str,
    learner_port: int,
    collector_host: str,
    collector_port: int,
    aggregator_host: str,
    aggregator_port: int,
    enable_total_log: bool,
    disable_flask_log: bool,
    module: str,
    module_name: str,
    # add/delete/restart
    add: str,
    delete: str,
    restart: str,
    kubeconfig: str,
    coordinator_name: str,
    namespace: str,
    replicas: int,
    cpus: int,
    gpus: int,
    memory: str,
    restart_pod_name: str,
    profile: str,
):
    if profile is not None:
        from ..utils.profiler_helper import Profiler
        profiler = Profiler()
        profiler.profile(profile)

    train_iter = int(float(train_iter))
    env_step = int(float(env_step))

    def run_single_pipeline(seed, config):
        if config is None:
            config = get_predefined_config(env, policy)
        else:
            config = read_config(config)
        if exp_name is not None:
            config[0].exp_name = exp_name

        if mode == 'serial':
            from .serial_entry import serial_pipeline
            serial_pipeline(config, seed, max_train_iter=train_iter, max_env_step=env_step)
        elif mode == 'serial_onpolicy':
            from .serial_entry_onpolicy import serial_pipeline_onpolicy
            serial_pipeline_onpolicy(config, seed, max_train_iter=train_iter, max_env_step=env_step)
        elif mode == 'serial_sqil':
            from .serial_entry_sqil import serial_pipeline_sqil
            expert_config = input("Enter the name of the config you used to generate your expert model: ")
            serial_pipeline_sqil(config, expert_config, seed, max_train_iter=train_iter, max_env_step=env_step)
        elif mode == 'serial_reward_model':
            from .serial_entry_reward_model_offpolicy import serial_pipeline_reward_model_offpolicy
            serial_pipeline_reward_model_offpolicy(config, seed, max_train_iter=train_iter, max_env_step=env_step)
        elif mode == 'serial_gail':
            from .serial_entry_gail import serial_pipeline_gail
            expert_config = input("Enter the name of the config you used to generate your expert model: ")
            serial_pipeline_gail(
                config, expert_config, seed, max_train_iter=train_iter, max_env_step=env_step, collect_data=True
            )
        elif mode == 'serial_dqfd':
            from .serial_entry_dqfd import serial_pipeline_dqfd
            expert_config = input("Enter the name of the config you used to generate your expert model: ")
            assert (expert_config == config[:config.find('_dqfd')] + '_dqfd_config.py'), "DQFD only supports "\
            + "the models used in q learning now; However, one should still type the DQFD config in this "\
            + "place, i.e., {}{}".format(config[:config.find('_dqfd')], '_dqfd_config.py')
            serial_pipeline_dqfd(config, expert_config, seed, max_train_iter=train_iter, max_env_step=env_step)
        elif mode == 'serial_trex':
            from .serial_entry_trex import serial_pipeline_trex
            serial_pipeline_trex(config, seed, max_train_iter=train_iter, max_env_step=env_step)
        elif mode == 'serial_trex_onpolicy':
            from .serial_entry_trex_onpolicy import serial_pipeline_trex_onpolicy
            serial_pipeline_trex_onpolicy(config, seed, max_train_iter=train_iter, max_env_step=env_step)
        elif mode == 'serial_offline':
            from .serial_entry_offline import serial_pipeline_offline
            serial_pipeline_offline(config, seed, max_train_iter=train_iter)
        elif mode == 'serial_ngu':
            from .serial_entry_ngu import serial_pipeline_ngu
            serial_pipeline_ngu(config, seed, max_train_iter=train_iter)
        elif mode == 'parallel':
            from .parallel_entry import parallel_pipeline
            parallel_pipeline(config, seed, enable_total_log, disable_flask_log)
        elif mode == 'dist':
            from .dist_entry import dist_launch_coordinator, dist_launch_collector, dist_launch_learner, \
                dist_prepare_config, dist_launch_learner_aggregator, dist_launch_spawn_learner, \
                dist_add_replicas, dist_delete_replicas, dist_restart_replicas
            if module == 'config':
                dist_prepare_config(
                    config, seed, platform, coordinator_host, learner_host, collector_host, coordinator_port,
                    learner_port, collector_port
                )
            elif module == 'coordinator':
                dist_launch_coordinator(config, seed, coordinator_port, disable_flask_log)
            elif module == 'learner_aggregator':
                dist_launch_learner_aggregator(
                    config, seed, aggregator_host, aggregator_port, module_name, disable_flask_log
                )

            elif module == 'collector':
                dist_launch_collector(config, seed, collector_port, module_name, disable_flask_log)
            elif module == 'learner':
                dist_launch_learner(config, seed, learner_port, module_name, disable_flask_log)
            elif module == 'spawn_learner':
                dist_launch_spawn_learner(config, seed, learner_port, module_name, disable_flask_log)
            elif add in ['collector', 'learner']:
                dist_add_replicas(add, kubeconfig, replicas, coordinator_name, namespace, cpus, gpus, memory)
            elif delete in ['collector', 'learner']:
                dist_delete_replicas(delete, kubeconfig, replicas, coordinator_name, namespace)
            elif restart in ['collector', 'learner']:
                dist_restart_replicas(restart, kubeconfig, coordinator_name, namespace, restart_pod_name)
            else:
                raise Exception
        elif mode == 'eval':
            from .application_entry import eval
            eval(config, seed, load_path=load_path, replay_path=replay_path)

    if isinstance(seed, (list, tuple)):
        assert len(seed) > 0, "Please input at least 1 seed"
        if len(seed) == 1:  # necessary
            run_single_pipeline(seed[0], config)
        else:
            if exp_name is None:
                multi_exp_root = os.path.basename(config).split('.')[0] + '_result'
            else:
                multi_exp_root = exp_name
            if not os.path.exists(multi_exp_root):
                os.mkdir(multi_exp_root)
            abs_config_path = os.path.abspath(config)
            origin_root = os.getcwd()
            for s in seed:
                seed_exp_root = os.path.join(multi_exp_root, 'seed{}'.format(s))
                if not os.path.exists(seed_exp_root):
                    os.mkdir(seed_exp_root)
                os.chdir(seed_exp_root)
                run_single_pipeline(s, abs_config_path)
                os.chdir(origin_root)
    else:
        raise TypeError("invalid seed type: {}".format(type(seed)))
