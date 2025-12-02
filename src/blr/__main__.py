"""
BLR is a pipeline for processing barcoded long reads
"""
import sys
import logging
import pkgutil
import importlib
from argparse import ArgumentParser, RawDescriptionHelpFormatter

import blr.cli as cli_package
from blr import __version__

logger = logging.getLogger(__name__)


def main(commandline_arguments=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(module)s - %(levelname)s: %(message)s")
    parser = ArgumentParser(description=__doc__, prog="blr")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--debug", action="store_true", default=False, help="Print debug messages")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="Save profile info to blr_<subcommand>.prof")
    subparsers = parser.add_subparsers()

    # Import each module that implements a subcommand and add a subparser for it.
    # Each subcommand is implemented as a module in the cli subpackage.
    # It needs to implement an add_arguments() and a main() function.
    modules = pkgutil.iter_modules(cli_package.__path__)
    for _, module_name, _ in modules:
        # Skip private/helper modules (underscore-prefixed) or known helpers
        # to avoid importing heavy dependencies that are not CLI subcommands.
        if module_name.startswith("_") or module_name in ("barcode_db", "tagfastq_mpi"):
            logger.debug("Skipping helper module: %s", module_name)
            continue
        try:
            module = importlib.import_module("." + module_name, cli_package.__name__)
        except Exception as e:
            logger.debug("Skipping module '%s' due to import error: %s", module_name, e)
            continue

        # Some modules may not define a module-level docstring; guard against None
        module_doc = module.__doc__ or ""
        help = module_doc.strip().split("\n", maxsplit=1)[0] if module_doc.strip() else ""

        # Only register modules that implement the CLI contract (add_arguments and main)
        if not (hasattr(module, "add_arguments") and hasattr(module, "main")):
            logger.debug("Skipping non-command module: %s", module_name)
            continue

        subparser = subparsers.add_parser(module_name, help=help, description=module_doc,
                                          formatter_class=RawDescriptionHelpFormatter)
        subparser.set_defaults(module=module)
        module.add_arguments(subparser)

    # Module 'run' needs to accept addition arguments
    args, extra_args = parser.parse_known_args(commandline_arguments)

    # For module 'run' extra_args are added to existing snakemake_args in namespace
    if hasattr(args, "snakemake_args"):
        args.snakemake_args += extra_args

    if args.debug:
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

    if not hasattr(args, "module"):
        parser.error("Please provide the name of a subcommand to run")
    else:
        module = args.module
        subcommand = module.main
        del args.module
        del args.debug
        profile = args.profile
        del args.profile

        module_name = module.__name__.split('.')[-1]

        # Re-parse extra arguments if module is not "run" to raise the expected error
        if module_name != "run" and extra_args:
            parser.parse_args(extra_args)

        # Print settings for module
        sys.stderr.write(f"SETTINGS FOR: {module_name} (version: {__version__})\n")
        for object_variable, value in vars(args).items():
            sys.stderr.write(f" {object_variable}: {value}\n")

        if profile:
            import cProfile
            profile_file = f'blr_{module_name}.prof'
            cProfile.runctx("subcommand(args)", globals(), dict(subcommand=subcommand, args=args),
                            filename=profile_file)
            logger.info(f"Writing profiling stats to '{profile_file}'.")
        else:
            subcommand(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
