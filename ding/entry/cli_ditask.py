import click
import os
import sys
import importlib
import importlib.util
import json
from click.core import Context, Option

from ding import __TITLE__, __VERSION__, __AUTHOR__, __AUTHOR_EMAIL__
from ding.framework import Parallel
from ding.entry.cli_parsers import PLATFORM_PARSERS


def print_version(ctx: Context, param: Option, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    click.echo('{title}, version {version}.'.format(title=__TITLE__, version=__VERSION__))
    click.echo('Developed by {author}, {email}.'.format(author=__AUTHOR__, email=__AUTHOR_EMAIL__))
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
@click.option('-p', '--package', type=str, help="Your code package path, could be a directory or a zip file.")
@click.option('--parallel-workers', type=int, default=1, help="Parallel worker number, default: 1")
@click.option(
    '--protocol',
    type=click.Choice(["tcp", "ipc"]),
    default="tcp",
    help="Network protocol in parallel mode, default: tcp"
)
@click.option(
    "--ports",
    type=str,
    default="50515",
    help="The port addresses that the tasks listen to, e.g. 50515,50516, default: 50515"
)
@click.option("--attach-to", type=str, help="The addresses to connect to.")
@click.option("--address", type=str, help="The address to listen to (without port).")
@click.option("--labels", type=str, help="Labels.")
@click.option("--node-ids", type=str, help="Candidate node ids.")
@click.option(
    "--topology",
    type=click.Choice(["alone", "mesh", "star"]),
    default="alone",
    help="Network topology, default: alone."
)
@click.option("--platform-spec", type=str, help="Platform specific configure.")
@click.option("--platform", type=str, help="Platform type: slurm, k8s.")
@click.option("--mq-type", type=str, default="nng", help="Class type of message queue, i.e. nng, redis.")
@click.option("--redis-host", type=str, help="Redis host.")
@click.option("--redis-port", type=int, help="Redis port.")
@click.option("-m", "--main", type=str, help="Main function of entry module.")
def cli_ditask(*args, **kwargs):
    return _cli_ditask(*args, **kwargs)


def _parse_platform_args(platform: str, platform_spec: str, all_args: dict):
    if platform_spec:
        try:
            if os.path.splitext(platform_spec) == "json":
                with open(platform_spec) as f:
                    platform_spec = json.load(f)
            else:
                platform_spec = json.loads(platform_spec)
        except:
            click.echo("platform_spec is not a valid json!")
            exit(1)
    if platform not in PLATFORM_PARSERS:
        click.echo("platform type is invalid! type: {}".format(platform))
        exit(1)
    all_args.pop("platform")
    all_args.pop("platform_spec")
    try:
        parsed_args = PLATFORM_PARSERS[platform](platform_spec, **all_args)
    except Exception as e:
        click.echo("error when parse platform spec configure: {}".format(e))
        exit(1)

    return parsed_args


def _cli_ditask(
    package: str,
    main: str,
    parallel_workers: int,
    protocol: str,
    ports: str,
    attach_to: str,
    address: str,
    labels: str,
    node_ids: str,
    topology: str,
    mq_type: str,
    redis_host: str,
    redis_port: int,
    platform: str = None,
    platform_spec: str = None,
):
    # Parse entry point
    all_args = locals()
    if platform:
        parsed_args = _parse_platform_args(platform, platform_spec, all_args)
        return _cli_ditask(**parsed_args)

    if not package:
        package = os.getcwd()
    sys.path.append(package)
    if main is None:
        mod_name = os.path.basename(package)
        mod_name, _ = os.path.splitext(mod_name)
        func_name = "main"
    else:
        mod_name, func_name = main.rsplit(".", 1)
    root_mod_name = mod_name.split(".", 1)[0]
    sys.path.append(os.path.join(package, root_mod_name))
    mod = importlib.import_module(mod_name)
    main_func = getattr(mod, func_name)
    # Parse arguments
    if not isinstance(ports, int):
        ports = ports.split(",")
        ports = list(map(lambda i: int(i), ports))
        ports = ports[0] if len(ports) == 1 else ports
    if attach_to:
        attach_to = attach_to.split(",")
        attach_to = list(map(lambda s: s.strip(), attach_to))
    if labels:
        labels = labels.split(",")
        labels = set(map(lambda s: s.strip(), labels))
    if node_ids and not isinstance(node_ids, int):
        node_ids = node_ids.split(",")
        node_ids = list(map(lambda i: int(i), node_ids))
    Parallel.runner(
        n_parallel_workers=parallel_workers,
        ports=ports,
        protocol=protocol,
        topology=topology,
        attach_to=attach_to,
        address=address,
        labels=labels,
        node_ids=node_ids,
        mq_type=mq_type,
        redis_host=redis_host,
        redis_port=redis_port
    )(main_func)
