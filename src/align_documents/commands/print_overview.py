from pathlib import Path
import os

def main(args, config):
    data_dir = Path(args.data_dir or os.environ.get("MALFRID", None) or config["data_dir"])

    if not data_dir.exists():
        raise FileNotFoundError("Provided data directory does not exist")
    
    print(f"Data directory: {data_dir}")

    for i, e in enumerate(data_dir.iterdir()):
        print(f"{i}\t{e.name}")
