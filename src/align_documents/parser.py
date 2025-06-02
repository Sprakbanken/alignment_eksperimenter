import sys
from argparse import ArgumentParser
from pathlib import Path

# subcommands
from align_documents.commands.align_all import main as align_all
from align_documents.commands.info import main as info

def set_align_all_parser(subparsers):
    parser = subparsers.add_parser(
        "align_all",
        help="Align all documents in data directory"
    )

    parser.set_defaults(func=align_all) # TODO: don't import this until required

    return parser

def set_info_parser(subparsers):
    parser = subparsers.add_parser(
        "info",
        help="Calculate dataset stats, and save to json files"
    )

    # TODO: Move/remove --data_dir?
    parser.add_argument(
        "-d", "--data_dir",
        type=Path,
        help="Path to the directory containing the documents.",
    )

    parser.add_argument(
        "-p", "--print-overview",
        action="store_true",
        help="Print dataset overview/aggregate stats to console",
    )

    parser.add_argument(
        "-P", "--print-full",
        action="store_true",
        help="Print full dataset stats to console",
    )

    parser.set_defaults(func=info) # TODO: don't import this until required

    return parser

def parse_args_():
    parser = ArgumentParser(
        prog="align_documents",
        description=(
            "Create and align document embeddings."
            "Command defaults to align_all if no command is given."
        )
    )

    subparsers = parser.add_subparsers(dest="command", required=False)

    parser.add_argument(
        "-c",
        "--config_file",
        help="Path to the config file",
        type=Path,
        default=Path("alignment_config.toml"),
    )
    parser.add_argument("-l", "--log_level", help="Log level", default="INFO")

    parser.set_defaults(func=None)

    set_info_parser(subparsers)
    default_parser = set_align_all_parser(subparsers)

    args = parser.parse_args()

    # Set default subparser when command is omitted, unless -h/--help is supplied.
    # NOTE: Modifies argv.
    if args.command is None and "help" not in args:
        sys.argv.insert(1, default_parser.prog.split()[-1]) # prog ~= "align_documents <subcommand>"
        args = parser.parse_args()

    return args
