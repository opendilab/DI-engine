import click
from click.core import Context, Option

from ding import __TITLE__, __VERSION__, __AUTHOR__, __AUTHOR_EMAIL__
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
    type=click.Choice(['serial', 'parallel', 'dist', 'eval']),
    help='serial-train or parallel-train or dist-train or eval'
)
@click.option('-c', '--config', type=str, help='Path to DRL experiment config')
@click.option(
    '-s',
    '--seed',
    type=int,
    default=0,
    help='random generator seed(for all the possible package: random, numpy, torch and user env)'
)
@click.option('-e', '--env', type=str, help='RL env name')
@click.option('-p', '--policy', type=str, help='DRL policy name')
@click.option('--train-iter', type=int, default=int(1e8), help='Policy training iterations')
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
def cli(
    # serial/eval
    mode: str,
    config: str,
    seed: int,
    env: str,
    policy: str,
    train_iter: int,
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
):
    if mode == 'serial':
        from .serial_entry import serial_pipeline
        if config is None:
            config = get_predefined_config(env, policy)
        serial_pipeline(config, seed, max_iterations=train_iter)
    elif mode == 'parallel':
        from .parallel_entry import parallel_pipeline
        parallel_pipeline(config, seed, enable_total_log, disable_flask_log)
    elif mode == 'dist':
        from .dist_entry import dist_launch_coordinator, dist_launch_collector, dist_launch_learner, \
            dist_prepare_config, dist_launch_learner_aggregator, dist_launch_spawn_learner
        if module == 'config':
            dist_prepare_config(
                config, seed, platform, coordinator_host, learner_host, collector_host, coordinator_port, learner_port,
                collector_port
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
        else:
            raise Exception
    elif mode == 'eval':
        from .application_entry import eval
        if config is None:
            config = get_predefined_config(env, policy)
        eval(config, seed, load_path=load_path, replay_path=replay_path)
