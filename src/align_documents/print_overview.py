from pathlib import Path
import os

def print_overview(args, config):
    # TODO: Check alignment_config.toml?
    if args.data_dir:
        data_dir = args.data_dir
        if not data_dir.exists():
            raise ValueError("Data directory does not exist.")
    else:
        data_dir = os.environ.get("MALFRID", None)
        if not data_dir:
            raise ValueError("No data directory provided.")
        data_dir = Path(data_dir)
    
    print(f"Data directory: {data_dir}")
    for i, e in enumerate(data_dir.iterdir()):
        print(f"{i}\t{e.name}")
