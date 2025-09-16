import json
import logging
from pathlib import Path
import argparse
from itertools import combinations
from align_documents.utils.logging import setup_logging

logger = logging.getLogger(__name__)

TRUE_POSITIVES = [
    "positive_pairs_nno_nob.jsonl",
    "positive_pairs_eng_nob.jsonl",
    "positive_pairs_eng_nno.jsonl",
]

TRUE_NEGATIVES = [
    "negative_pairs_nob_nno.jsonl",
    "negative_pairs_nob_eng.jsonl",
    "negative_pairs_nno_eng.jsonl",
]


def load_domains(file_path: Path, langpair: str) -> dict:
    """loads domains from file, returns a mapping between domains and examples"""
    domain_key = f"domain_{langpair.split('_')[-1]}"
    domain_to_example = {}

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            domain = example.get(domain_key)
            if domain and domain not in domain_to_example:
                domain_to_example[domain] = example

    return domain_to_example

def get_domains_for_file(file:str, pos_domains:dict, neg_domains:dict) -> set:
    """Returns set of the existing domains in the positive and negative files"""
    if file in pos_domains:
        return set(pos_domains[file].keys())
    elif file in neg_domains:
        return set(neg_domains[file].keys())
    else:
        return set()


def domains_in_n_way_overlap(all_files:list, n:int, pos_domains:dict, neg_domains:dict)-> set:
    """Return all domains that appear in at least one n-way file, so either 6 files, 5 files, 3 files etc"""
    overlap_domains = set()
    for combo in combinations(all_files, n):
        intersection = get_domains_for_file(combo[0], pos_domains, neg_domains)
        for f in combo[1:]: 
            intersection &= get_domains_for_file(f, pos_domains, neg_domains)
        overlap_domains.update(intersection)
    return overlap_domains  


def find_overlapping_domains_progressively(pos_domains:dict, neg_domains:dict, max_examples=20):
    """Find up to max_examples domains present in both pos and neg groups, and progressively
    add domains that are present in n-way overlaps (starting from 6-way, then 5-way, etc).
    """
    selected_domains = set()
    contribution_log = []

    all_files = list(pos_domains.keys()) + list(neg_domains.keys())

    for n in range(len(all_files), 1, -1):
        logger.info(f"Checking {n}-way overlaps...")

        overlap_domains = domains_in_n_way_overlap(
            all_files, n, pos_domains, neg_domains
        )

        filtered_domains = set()
        for d in overlap_domains:
            found_in_pos = False
            for m in pos_domains.values():
                if d in m:
                    found_in_pos = True
                    break
            found_in_neg = False
            for m in neg_domains.values():
                if d in m:
                    found_in_neg = True
                    break 
            if found_in_pos and found_in_neg:
                filtered_domains.add(d)
        overlap_domains = filtered_domains


        new_domains = []
        for d in overlap_domains:
            if d not in selected_domains:
                new_domains.append(d)
        needed = max_examples - len(selected_domains)
        take = new_domains[:needed]

        selected_domains.update(take)

        contribution_log.append((n, take))
        logger.info(
            f"Added {len(take)} domains from {n}-way overlaps "
            f"(total so far: {len(selected_domains)}/{max_examples})"
        )
        if take:
            logger.info(f"→ Domains added at {n}-way: {', '.join(take)}")

        if len(selected_domains) >= max_examples:
            break

    logger.info(f"Final number of selected domains: {len(selected_domains)}")
    for k, domains in contribution_log:
        logger.info(
            f"{len(domains)} domains came from {k}-way overlaps: {', '.join(domains)}"
        )

    return list(selected_domains), contribution_log


def get_args():
    parser = argparse.ArgumentParser(prog="Create balanced dataset")

    parser.add_argument(
        "-input_dir",
        help="Path to the input directory",
        type=Path,
        default=Path("data/output"),
    )
    parser.add_argument(
        "-l",
        "--log_level",
        help="Log level",
        default="INFO",
        choices=["INFO", "DEBUG", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main(args):
    output_dir = args.input_dir / "20_pairs"
    output_dir.mkdir(exist_ok=True)
    log_file = output_dir / "dataset_creation.log"
    setup_logging("Create balanced dataset", log_level=args.log_level)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Starting process")

    pos_domains = {}
    for filename in TRUE_POSITIVES:
        path = args.input_dir / filename
        langpair = filename.replace(".jsonl", "").split("_")[-2:]
        langpair = "_".join(langpair)
        logger.info(f"Loading {filename}")
        pos_domains[filename] = load_domains(path, langpair)
        logger.info(f"{filename}: {len(pos_domains[filename])} unique domains")

    neg_domains = {}
    for filename in TRUE_NEGATIVES:
        path = args.input_dir / filename
        langpair = filename.replace(".jsonl", "").split("_")[-2:]
        langpair = "_".join(langpair)
        logger.info(f"Loading {filename}")
        neg_domains[filename] = load_domains(path, langpair)
        logger.info(f"{filename}: {len(neg_domains[filename])} unique domains")

    selected_domains, contribution_log = find_overlapping_domains_progressively(
        pos_domains, neg_domains, max_examples=20
    )
    logger.info(f"Final number of selected domains: {len(selected_domains)}")
    for k, count in contribution_log:
        logger.info(f"{count} domains came from {k}-way overlaps")

    for filename, domain_map in {**pos_domains, **neg_domains}.items():
        outname = "20_" + filename
        outpath = output_dir / outname
        with outpath.open("w", encoding="utf-8") as f:
            for d in selected_domains:
                if d in domain_map:
                    f.write(json.dumps(domain_map[d], ensure_ascii=False) + "\n")
        logger.info(f"Wrote {outpath} with up to {len(selected_domains)} examples")

    logger.info("Done.")


if __name__ == "__main__":
    args = get_args()
    main(args)
