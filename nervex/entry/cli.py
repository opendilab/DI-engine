import click
from click.core import Context, Option
from nervex import __TITLE__, __VERSION__, __AUTHOR__, __AUTHOR_EMAIL__
from .serial_entry import serial_pipeline
from .parallel_entry import parallel_pipeline
from .dist_entry import dist_launch_coordinator, dist_launch_collector, dist_launch_learner, dist_prepare_config
from .application_entry import eval


def print_version(ctx: Context, param: Option, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    click.echo('{title}, version {version}.'.format(title=__TITLE__, version=__VERSION__))
    click.echo('Developed by {author}, {email}.'.format(author=__AUTHOR__, email=__AUTHOR_EMAIL__))
    ctx.exit()


def print_registry(ctx: Context, param: Option, value: str):
    if value is None:
        return
    from nervex.utils import registries  # noqa
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
    '--query_registry',
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
@click.option('--enable_total_log', type=bool, help='whether enable the total nervex system log', default=False)
@click.option('--disable_flask_log', type=bool, help='whether disable flask log', default=True)
# the following arguments are only applied to dist mode
@click.option(
    '-p', '--platform', type=click.Choice(['local', 'slurm', 'k8s']), help='local or slurm or k8s', default='local'
)
@click.option('--module', type=click.Choice(['config', 'collector', 'learner', 'coordinator']), help='dist module type')
@click.option('--module-name', type=str, help='dist module name')
@click.option('-cdh', '--coordinator_host', type=str, help='coordinator host', default='0.0.0.0')
@click.option('-lh', '--learner_host', type=str, help='learner host', default='0.0.0.0')
@click.option('-clh', '--collector_host', type=str, help='collector host', default='0.0.0.0')
def cli(
    mode: str,
    config: str,
    seed: int,
    platform: str,
    coordinator_host: str,
    learner_host: str,
    collector_host: str,
    enable_total_log: bool,
    disable_flask_log: bool,
    module: str,
    module_name: str,
):
    if mode == 'serial':
        serial_pipeline(config, seed, enable_total_log=enable_total_log)
    elif mode == 'parallel':
        parallel_pipeline(config, seed, enable_total_log, disable_flask_log)
    elif mode == 'dist':
        if module == 'config':
            dist_prepare_config(config, seed, platform, coordinator_host, learner_host, collector_host)
        elif module == 'coordinator':
            dist_launch_coordinator(config, seed, disable_flask_log)
        elif module == 'collector':
            dist_launch_collector(config, seed, module_name, disable_flask_log)
        elif module == 'learner':
            dist_launch_learner(config, seed, module_name, disable_flask_log)
        else:
            raise Exception
    elif mode == 'eval':
        eval(config, seed)
