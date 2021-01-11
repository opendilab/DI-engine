import click
from click.core import Context, Option
from .serial_entry import serial_pipeline
from .parallel_entry import parallel_pipeline


def print_version(ctx: Context, param: Option, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    click.echo('{title}, version {version}.'.format(title='nerveX', version='v0.0.2rc0'))
    click.echo('Developed by {author}, {email}.'.format(author='niuyazhe', email='niuyazhe@sensetime.com'))
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
@click.option('-m', '--mode', type=str, help='serial or parallel')
@click.option('-c', '--config', type=str, help='Path to DRL experiment config')
@click.option(
    '-s',
    '--seed',
    type=int,
    default=0,
    help='random generator seed(for all the possible package: random, numpy, torch and user env)'
)
@click.option('-p', '--platform', type=str, help='local or slurm or k8s', default='local')
# @click.option('-ch', '--coordinator_host', type=str, help='coordinator host')
def cli(mode: str, config: str, seed: int):
    assert mode in ['serial', 'parallel'], "nervex pipeline mode must in [serial, parallel]"
    if mode == 'serial':
        serial_pipeline(config, seed)
    elif mode == 'parallel':
        parallel_pipeline(config, seed)
