from argparse import ArgumentParser
from pathlib import Path

# subcommands
from align_documents.commands.align_all import main as align_all
from align_documents.commands.print_overview import main as print_overview

def set_align_all_parser(subparsers):
    parser = subparsers.add_parser(
        "align_all",
        help="Align all documents in data directory"
    )

    parser.set_defaults(func=align_all) # TODO: don't import this until required

    return parser

def set_print_overview_parser(subparsers):
    parser = subparsers.add_parser(
        "info",
        help="Print dataset info"
    )

    # TODO: Move/remove --data_dir?
    parser.add_argument("-d", "--data_dir", help="Path to the directory containing the documents.", type=Path)

    parser.set_defaults(func=print_overview) # TODO: don't import this until required

    return parser

def get_parser():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser.add_argument(
        "-c",
        "--config_file",
        help="Path to the config file",
        type=Path,
        default=Path("alignment_config.toml"),
    )
    parser.add_argument("-l", "--log_level", help="Log level", default="INFO")

    parser.set_defaults(func=None)
    set_align_all_parser(subparsers)
    set_print_overview_parser(subparsers)

    return parser
