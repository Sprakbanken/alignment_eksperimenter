import tarfile
import subprocess
import logging
import argparse
import gzip
import shutil
from pathlib import Path
from align_documents.utils import setup_logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def decompress_gz_files(directory: Path) -> None:
    """Decompress all .gz files in the given directory and its subdirectories."""
    gz_files = list(directory.rglob("*.gz"))
    if not gz_files:
        logger.info("No .gz files found to decompress")
        return

    logger.info("Found %d .gz files to decompress", len(gz_files))

    for gz_file in tqdm(gz_files, desc="Decompressing"):
        # Create output filename by removing .gz extension
        output_file = gz_file.with_suffix("")

        logger.debug("Decompressing %s -> %s", gz_file.name, output_file.name)
        try:
            with gzip.open(gz_file, "rb") as f_in:
                with open(output_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            # Remove the .gz file after successful decompression
            gz_file.unlink()

        except Exception:
            logger.exception("Couldn't decompress %s", gz_file)


def download_malfrid_data(målfrid_url: str, tar_file_path: Path):
    logger.info(
        "Downloading Målfrid data from %s (saving to %s)", målfrid_url, tar_file_path
    )

    subprocess.run(
        ["wget", "-O", str(tar_file_path), målfrid_url],
        check=True,
    )
    logger.info("Download completed: %s", tar_file_path)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and extract the Målfrid dataset"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="https://www.nb.no/sbfil/tekst/maalfrid_2025/maalfrid_2025.tar",
        help="URL to download the Målfrid dataset from (default: %(default)s)",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("data/"),
        help="Path to extract the dataset to (default: %(default)s)",
    )
    parser.add_argument(
        "--tar_file_path", type=Path, help="Tar file path", required=False
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: %(default)s)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    setup_logging("download_malfrid", args.log_level)

    # Create data directory if it doesn't exist
    args.data_path.mkdir(parents=True, exist_ok=True)

    if args.tar_file_path is None:
        args.tar_file_path = args.data_path / "målfrid_out.tar"

    if not args.tar_file_path.exists():
        download_malfrid_data(målfrid_url=args.url, tar_file_path=args.tar_file_path)
    else:
        logger.info("Reading from %s (skipping download)", args.tar_file_path)

    # Extract the tar file
    logger.info("Extracting tar file...")
    with tarfile.open(args.tar_file_path, "r") as tar:
        tar.extractall(path=args.data_path)
    logger.info("Extraction completed to: %s", args.data_path)

    # Decompress all .gz files in the extracted content
    logger.info("Decompressing .gz files...")
    decompress_gz_files(args.data_path)
    logger.info("Decompression completed.")

    logger.info("Removing tar file...")
    args.tar_file_path.unlink(missing_ok=True)
    logger.info("Cleanup completed.")
