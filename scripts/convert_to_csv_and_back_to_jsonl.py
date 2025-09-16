import argparse
import csv
import json
import re
from pathlib import Path


def rename_lang_keys_and_write_csv(input_file: Path, output_file: Path):
    """takes a jsonl file and renames the language keys to lang_1 and lang_2, then writes to a csv file"""
    lang_pattern = re.compile(r"_(\w{3})$")  # matches _nno, _nob, _eng
    rows = []
    headers = set()

    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            obj = json.loads(line)
            lang_codes = []
            for k in obj.keys():
                m = lang_pattern.search(k)
                if m:
                    code = m.group(1)
                    if code not in lang_codes:
                        lang_codes.append(code)
            lang_map = {}
            if len(lang_codes) >= 2:
                lang_map = {lang_codes[0]: "lang_1", lang_codes[1]: "lang_2"}
            elif len(lang_codes) == 1:
                lang_map = {lang_codes[0]: "lang_1"}
            new_obj = {}
            for k, v in obj.items():
                m = lang_pattern.search(k)
                if m and m.group(1) in lang_map:
                    new_k = lang_pattern.sub(f"_{lang_map[m.group(1)]}", k)
                else:
                    new_k = k
                if isinstance(v, list):
                    v = json.dumps(v, ensure_ascii=False)
                new_obj[new_k] = v
                headers.add(new_k)
            rows.append(new_obj)

    headers = sorted(headers)

    with open(output_file, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def convert_back_to_jsonl(input_file: Path, output_file: Path):
    """takes a csv file and converts it back to jsonl, renaming lang_1 and lang_2 to the actual language codes"""
    csv.field_size_limit(10**7)
    lang_pattern = re.compile(r"_(lang_1|lang_2)$")
    rows = []

    with open(input_file, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            lang_1_code = row.get("lang_lang_1", "lang_1")
            lang_2_code = row.get("lang_lang_2", "lang_2")
            new_obj = {}
            for k, v in row.items():
                m = lang_pattern.search(k)
                if m:
                    lang_code = lang_1_code if m.group(1) == "lang_1" else lang_2_code
                    new_k = lang_pattern.sub(f"_{lang_code}", k)
                else:
                    new_k = k
                try:
                    v_parsed = json.loads(v)
                    if isinstance(v_parsed, list):
                        v = v_parsed
                except (json.JSONDecodeError, TypeError):
                    pass
                new_obj[new_k] = v
            rows.append(new_obj)

    with open(output_file, "w", encoding="utf-8") as fout:
        for obj in rows:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def get_args():
    parser = argparse.ArgumentParser(
        description="Rename language keys in JSONL and output CSV"
    )
    parser.add_argument(
        "--input_file", type=Path, required=True, help="Input JSONL file"
    )
    parser.add_argument(
        "--output_file", type=Path, required=True, help="Output CSV file"
    )

    parser.add_argument(
        "--conversion_type",
        type=str,
        required=True,
        help="Type of conversion (csv_to_jsonl or jsonl_to_csv)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.conversion_type == "jsonl_to_csv":
        rename_lang_keys_and_write_csv(args.input_file, args.output_file)
    elif args.conversion_type == "csv_to_jsonl":
        convert_back_to_jsonl(args.input_file, args.output_file)
