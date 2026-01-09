import logging
import argparse
import os
import shutil
from pathlib import Path

from align_documents.utils.config import get_config, CONFIG_PATH
from align_documents.utils.dataframe import get_file_info, get_websites_with_both_langs

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(
        description=f"Create a subset of the given .jsonl dataset, based on {CONFIG_PATH}"
    )

    parser.add_argument(
        "dest_dir",
        help="Path to the destination directory.",
        type=Path,
    )

    parser.add_argument(
        "--src_dir",
        help="Override the directory from which to copy files from.",
        type=Path,
    )

    parser.add_argument(
        "-l",
        "--both_langs",
        help=(
            "Copy only files that exist for both languages listed in"
            f" {CONFIG_PATH}, based on filename. Both languages will be"
            " copied, and will count as *one* file when passing '-n'."
        ),
        action="store_true",
    )

    parser.add_argument(
        "-s",
        "--sort_size",
        help=("Sort by size.\nasc  => smaller files first\ndesc => larger files first"),
        choices=["asc", "desc"],
        default=None,
    )

    parser.add_argument(
        "-c",
        "-n",
        "--count",
        help="Number of files to copy. Defaults to 200 files. Pass -1 to copy all files.",
        type=int,
        default=200,
    )

    parser.add_argument(
        "-O", "--overwrite", help="Overwrite destination directory", action="store_true"
    )

    parser.add_argument(
        "-M",
        "--merge",
        help=(
            "Copy files into destination directory, even if it already exists"
            ", and is not empty."
        ),
        action="store_true",
    )

    return parser.parse_args()


def prepare_directories(
    src: Path,
    dst: Path,
    overwrite: bool = False,
    merge: bool = False,
):
    if src == dst:
        raise ValueError("src_dir and dst_dir are the same.")

    # Prepare source dir
    if not src.exists() or not next(os.scandir(src), None):
        raise FileNotFoundError("Source directory empty or not found.")

    # Prepare dest dir
    if overwrite:
        assert src != dst  # Just to be extra safe...
        remove = input(f"Will remove '{dst.absolute()}'. Continue? [y/N] ")
        if remove.lower() in ["y", "yes"]:
            shutil.rmtree(dst)
            dst.mkdir(parents=True)
        else:
            logger.info("Aborting.")
            exit()
    elif merge:
        dst.mkdir(parents=True, exist_ok=True)
    else:
        try:
            dst.mkdir(parents=True, exist_ok=False)
        except FileExistsError as e:
            logger.critical(
                f"{dst} already exists. Consider using the merge or overwrite options."
            )
            raise e


def main(args: argparse.Namespace):
    config = get_config()

    src_dir: Path = args.src_dir if args.src_dir is not None else config.data_dir
    dst_dir: Path = args.dest_dir

    prepare_directories(src_dir, dst_dir, args.overwrite, args.merge)
    file_info = get_file_info(src_dir)

    if args.both_langs:
        file_info = get_websites_with_both_langs(file_info, config.languages)

    if args.sort_size is not None:
        file_info["file_size"] = [
            path.stat().st_size for path in file_info["file_path"]
        ]
        ascending = args.sort_size == "asc"
        file_info.sort_values(
            "file_size", inplace=True, ignore_index=True, ascending=ascending
        )

    for file in file_info["file_path"][: args.count]:
        shutil.copy(file, dst_dir / file.name)


if __name__ == "__main__":
    main(get_args())
