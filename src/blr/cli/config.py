"""
Update configuration file. If no --set option is given the current settings are printed.
"""
from collections.abc import MutableMapping
from importlib_resources import path as resource_path
import logging
from pathlib import Path
from shutil import get_terminal_size
import sys
from typing import List, Tuple, Union, Optional

from ruamel.yaml import YAML
from ruamel.yaml.parser import ParserError

from snakemake.utils import validate as validate_yaml

logger = logging.getLogger(__name__)
DEFAULT_PATH = Path("blr.yaml")
SCHEMA_FILE = "config.schema.yaml"


# Script is based on repos NBISSweden/IgDisover config script.
# Link https://github.com/NBISweden/IgDiscover/blob/master/src/igdiscover/cli/config.py

def main(args):
    run(yaml_file=args.file, changes_set=args.set, update_from=args.update_from, prompt=args.i)


def run(
    yaml_file: Optional[Path] = DEFAULT_PATH,
    changes_set: Optional[List[Tuple[str, str]]] = None,
    update_from: Optional[Path] = None,
    prompt: bool = False,
):
    changes_set = [] if changes_set is None else changes_set
    if update_from is not None:
        configs, _ = load_yaml(update_from)
        changes_set = update_changes_set(changes_set, configs)

    if changes_set:
        change_config(yaml_file, changes_set, prompt)
    else:
        print_config(yaml_file)


def print_config(filename: Union[Path, str]):
    """
    Print out current configs to terminal.
    """
    configs, yaml = load_yaml(filename)
    width, _ = get_terminal_size()
    header = f" CONFIGS IN: {filename} "
    padding = int((width - len(header)) / 2) * "="

    # Print out current setting
    print(f"{padding}{header}{padding}")
    yaml.dump(configs, stream=sys.stdout)
    print(f"{'=' * width}")


def change_config(
        filename: Union[Path, str],
        changes_set: List[Tuple[str, str]],
        prompt: bool = False,
        validate: bool = True
):
    """
    Change config YAML file at filename using the changes_set key-value pairs.
    :param filename: Path to YAML config file to change.
    :param changes_set: changes to incorporate.
    :param prompt: prompt before changing configs parameter
    """
    # Get configs from file.
    configs, yaml = load_yaml(filename)

    # Update configs
    for key, value in changes_set:
        # Convert relative paths to absolute
        value = make_paths_absolute(value, workdir=filename.parent)
        try:
            value = YAML(typ='safe').load(value)
        except ParserError:
            logger.warning(f"ParserError raised for value {value} in key {key} .")
        item = configs

        # allow nested keys
        keys = key.split('.')
        for i in keys[:-1]:
            item = item[i]

        prev_value = item[keys[-1]] if keys[-1] in item else "NOT SET"
        if prev_value != value:
            if prompt and input(f"Change value of '{key}': {repr(prev_value)} --> {repr(value)} (y/n)?") != "y":
                continue
            item[keys[-1]] = value
            if not prompt:
                logger.info(f"Changing value of '{key}': {repr(prev_value)} --> {repr(value)}")

    # Confirm that configs is valid.
    if validate:
        with resource_path('blr', SCHEMA_FILE) as schema_path:
            validate_yaml(configs, str(schema_path))

    # Write first to temporary file then overwrite filename.
    tmpfile = Path(str(filename) + ".tmp")
    with open(tmpfile, "w") as file:
        yaml.dump(configs, stream=file)
    tmpfile.rename(filename)


def load_yaml(filename: Union[Path, str]) -> Tuple[MutableMapping, YAML]:
    """
    Load YAML file and return the yaml object and data.
    :param filename: Path to YAML file
    :return: (data, yaml).
    """
    with open(filename) as file:
        yaml = YAML()
        yaml.allow_duplicate_keys = True
        data = yaml.load(file)
    return data, yaml


def make_paths_absolute(value: str, workdir: Union[Path, str] = Path.cwd()) -> str:
    """
    Detect if value is a relative path and make it absolut if so.
    :param value: Parameter value from arguments
    :param workdir: Path to workdir. Default: CWD
    :return:
    """
    if "../" in value and (workdir / value).exists():
        if (workdir / value).is_symlink():
            return str((workdir / value).absolute())
        return str((workdir / value).resolve(True))
    return value


def update_changes_set(changes_set: List[Tuple[str, str]], configs: MutableMapping) -> List[Tuple[str, str]]:
    """Update changes_set list of tuples with configs in dict configs"""
    configs = flatten(configs)
    configs = {k: str(v) if v is not None else "null" for k, v in configs.items()}

    # Merge configs from yaml with those in changes_set
    configs_primary = dict(changes_set)
    configs_primary = {**configs, **configs_primary}
    return list(configs_primary.items())


def flatten(d: MutableMapping, parent_key: str = '', sep: str = '.') -> MutableMapping:
    """Flatten nested dict into dict where keys are nested as KEY.SUBKEY[.SUBSUBKEY...]."""
    # Adapted from https://stackoverflow.com/a/6027615
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_arguments(parser):
    parser.add_argument(
        "-s", "--set", nargs=2, metavar=("KEY", "VALUE"), action="append",
        help="Set KEY to VALUE. Use KEY.SUBKEY[.SUBSUBKEY...] for nested keys. For empty values write 'null'. Can be "
             "given multiple times."
    )
    parser.add_argument(
        "-f", "--file", default=DEFAULT_PATH, type=Path, metavar="YAML",
        help="Configuration file to modify. Default: %(default)s in current directory."
    )
    parser.add_argument(
        "-u", "--update-from", type=Path, metavar="YAML",
        help="Update configuration using other configuration file."
    )
    parser.add_argument(
        "-i", action="store_true", default=False, help="Prompt before every change"
    )
