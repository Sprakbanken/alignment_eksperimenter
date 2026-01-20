import argparse
import glob
import json
import os
import sys
import traceback

from tqdm import tqdm


def load_jsonl_files_streaming(
    folder_paths: list,
    output_dataset: str,
    pattern="*.jsonl",
    max_files: int | None = None,
) -> dict:
    """Stream-process JSONL files, filtering duplicates without loading everything into memory."""
    file_paths = []
    try:
        for p in folder_paths:
            if os.path.isdir(p):
                file_paths.extend(glob.glob(os.path.join(p, pattern), recursive=False))
            elif os.path.isfile(p):
                file_paths.append(p)
            else:
                print(f"Skipping {p}: not a file or directory")
    except Exception as e:
        print(f"Error while collecting file paths: {e}")

    if max_files is not None:
        file_paths = file_paths[:max_files]

    if not file_paths:
        raise ValueError(
            "No JSONL files found to process. Check input paths and pattern."
        )

    print("Pass 1: Scanning files to identify duplicates...")
    records_index = []

    iterator = tqdm(file_paths, desc="Scanning files", leave=True)
    for file_path in iterator:
        iterator.set_postfix({"file": os.path.basename(file_path)[:30]})
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        doc_hash = record.get("doc_hash")
                        url = record.get("url")
                        records_index.append((file_path, line_idx, doc_hash, url))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")

    print(f"Total records scanned: {len(records_index)}")

    print("Determining records to keep...")
    keep_set = set()
    seen_hashes = set()
    seen_urls = set()

    for file_path, line_idx, doc_hash, url in reversed(records_index):
        if doc_hash in seen_hashes or url in seen_urls:
            continue

        keep_set.add((file_path, line_idx))
        if doc_hash is not None:
            seen_hashes.add(doc_hash)
        if url is not None:
            seen_urls.add(url)

    print(f"Records to keep after deduplication: {len(keep_set)}")

    print("Pass 2: Writing deduplicated records...")
    records_written = 0

    with open(output_dataset, "w", encoding="utf-8") as out_f:
        iterator = tqdm(file_paths, desc="Writing records", leave=True)
        for file_path in iterator:
            iterator.set_postfix({"file": os.path.basename(file_path)[:30]})
            try:
                with open(file_path, "r", encoding="utf-8") as in_f:
                    for line_idx, line in enumerate(in_f):
                        if (file_path, line_idx) in keep_set:
                            try:
                                record = json.loads(line)
                                record["source_file"] = os.path.basename(file_path)
                                out_f.write(json.dumps(record) + "\n")
                                records_written += 1
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    print(f"Total records written: {records_written}")
    return {"total_scanned": len(records_index), "total_written": records_written}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a Maalfrid superset from multiple datasets."
    )
    parser.add_argument(
        "--input_datasets",
        nargs="+",
        required=True,
        help="Paths to input datasets to be merged into a superset.",
    )
    parser.add_argument(
        "--output_dataset",
        required=True,
        help="Path to save the resulting Maalfrid superset dataset.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional limit on the number of files to process (useful for debugging).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Merging datasets: {args.input_datasets}")
    print(f"Saving merged dataset to: {args.output_dataset}")
    try:
        load_jsonl_files_streaming(
            args.input_datasets, args.output_dataset, max_files=args.max_files
        )
    except Exception as e:
        print(f"Error during merge: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
