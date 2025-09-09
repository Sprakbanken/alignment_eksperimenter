import tarfile
import urllib.request
import logging
import argparse
import gzip
import shutil
from pathlib import Path
from align_documents.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def decompress_gz_files(directory: Path):
    """Decompress all .gz files in the given directory and its subdirectories."""
    gz_files = list(directory.rglob("*.gz"))
    if not gz_files:
        logger.info("No .gz files found to decompress")
        return

    logger.info(f"Found {len(gz_files)} .gz files to decompress")

    for gz_file in gz_files:
        # Create output filename by removing .gz extension
        output_file = gz_file.with_suffix("")

        logger.info(f"Decompressing {gz_file.name} -> {output_file.name}")

        with gzip.open(gz_file, "rb") as f_in:
            with open(output_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Remove the .gz file after successful decompression
        gz_file.unlink()


def download_and_extract_malfrid(
    målfrid_url: str,
    data_path: Path,
):
    """Download and extract the Målfrid project data from the resource catalouge"""

    # Create data directory if it doesn't exist
    data_path.mkdir(parents=True, exist_ok=True)

    tar_file_path = data_path / "målfrid_out.tar"

    logger.info(f"Downloading Målfrid data from {målfrid_url}...")
    try:
        # Download the tar file
        urllib.request.urlretrieve(målfrid_url, tar_file_path)
        logger.info(f"Download completed: {tar_file_path}")

        # Extract the tar file
        logger.info("Extracting tar file...")
        with tarfile.open(tar_file_path, "r") as tar:
            tar.extractall(path=data_path)
        logger.info(f"Extraction completed to: {data_path}")

        # Decompress all .gz files in the extracted content
        logger.info("Decompressing .gz files...")
        decompress_gz_files(data_path)
        logger.info("Decompression completed.")

        logger.info("Removing tar file...")
        tar_file_path.unlink()
        logger.info("Cleanup completed.")

    except Exception as e:
        logger.error(f"Error downloading or extracting data: {e}")
        return False

    return True


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

    success = download_and_extract_malfrid(
        målfrid_url=args.url, data_path=args.data_path
    )
    if success:
        logger.info("Målfrid data successfully downloaded and extracted!")
    else:
        logger.error("Failed to download or extract Målfrid data.")
