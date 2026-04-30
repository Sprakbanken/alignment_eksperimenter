from collections import defaultdict
from pathlib import Path
import random
import pandas as pd
from align_documents.utils import setup_logging
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)


TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42


def get_domain_counts(superset_path: Path) -> defaultdict[str, dict[str, int]]:
    domain_counts: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for lang_pair_dir in sorted(superset_path.iterdir()):
        lang_pair = lang_pair_dir.name
        for jsonl_file in sorted(lang_pair_dir.glob("*.jsonl")):
            domain = jsonl_file.stem.removesuffix(f"_{lang_pair}")
            num_rows = sum(1 for _ in jsonl_file.open())
            domain_counts[domain][lang_pair] = num_rows
    return domain_counts


def log_domain_counts(domain_counts: defaultdict[str, dict[str, int]]) -> None:
    for domain in sorted(domain_counts):
        for lang_pair, count in sorted(domain_counts[domain].items()):
            logger.info(f"{domain} | {lang_pair}: {count} rows")
        logger.info("")


def assign_domains_to_splits(
    domain_counts: defaultdict[str, dict[str, int]],
) -> dict[str, str]:
    """Assign each domain globally to train/validation/test.

    Uses a greedy approach: sort domains by total row count (descending),
    then assign each domain to the split that is furthest below its target
    ratio, considering all language pairs simultaneously.
    """
    all_lang_pairs = sorted(
        {lang_pair for counts in domain_counts.values() for lang_pair in counts}
    )

    # Total rows per lang pair across all domains
    totals_per_lang_pair: dict[str, int] = defaultdict(int)
    for counts in domain_counts.values():
        for lang_pair, row_count in counts.items():
            totals_per_lang_pair[lang_pair] += row_count

    # Current rows assigned to each split, per lang pair
    split_rows: dict[str, dict[str, int]] = {
        "train": defaultdict(int),
        "validation": defaultdict(int),
        "test": defaultdict(int),
    }
    target_ratios = {
        "train": TRAIN_RATIO,
        "validation": VALIDATION_RATIO,
        "test": TEST_RATIO,
    }

    # Sort domains by total rows descending (largest first for better balance)
    domains_sorted = sorted(
        domain_counts.keys(),
        key=lambda domain: sum(domain_counts[domain].values()),
        reverse=True,
    )

    # Shuffle domains of equal size for reproducibility
    rng = random.Random(SEED)
    rng.shuffle(domains_sorted)
    domains_sorted.sort(
        key=lambda domain: sum(domain_counts[domain].values()), reverse=True
    )

    assignments: dict[str, str] = {}

    for domain in domains_sorted:
        best_split = None
        best_score = float("inf")

        for split in ("test", "validation", "train"):
            # Compute total squared deviation across ALL splits and lang pairs
            # if we assigned this domain to this split
            score = 0.0
            for lang_pair in all_lang_pairs:
                total = totals_per_lang_pair[lang_pair]
                if total == 0:
                    continue
                domain_rows = domain_counts[domain].get(lang_pair, 0)
                for other_split in ("train", "validation", "test"):
                    current = split_rows[other_split][lang_pair]
                    if other_split == split:
                        current += domain_rows
                    deviation = (current / total) - target_ratios[other_split]
                    score += deviation**2
            if score < best_score:
                best_score = score
                best_split = split

        assignments[domain] = best_split
        for lang_pair, row_count in domain_counts[domain].items():
            split_rows[best_split][lang_pair] += row_count

    return assignments


def log_split_statistics(
    domain_counts: defaultdict[str, dict[str, int]],
    assignments: dict[str, str],
) -> None:
    all_lang_pairs = sorted(
        {lang_pair for counts in domain_counts.values() for lang_pair in counts}
    )

    for lang_pair in all_lang_pairs:
        total = sum(counts.get(lang_pair, 0) for counts in domain_counts.values())
        logger.info(f"=== {lang_pair} (total: {total}) ===")
        for split in ("train", "validation", "test"):
            split_total = sum(
                domain_counts[domain].get(lang_pair, 0)
                for domain, assigned_split in assignments.items()
                if assigned_split == split
            )
            pct = (split_total / total * 100) if total else 0
            n_domains = sum(
                1
                for domain, assigned_split in assignments.items()
                if assigned_split == split and lang_pair in domain_counts[domain]
            )
            logger.info(
                f"  {split}: {split_total} rows ({pct:.1f}%) from {n_domains} domains"
            )
        logger.info("")

    # Domain lists per lang pair per split
    split_domains: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for domain, split in assignments.items():
        for lang_pair in domain_counts[domain]:
            split_domains[lang_pair][split].add(domain)

    for lang_pair in all_lang_pairs:
        for split in ("train", "validation", "test"):
            domains = sorted(split_domains[lang_pair].get(split, set()))
            logger.info(f"{lang_pair} {split} domains ({len(domains)}): {domains}")
        logger.info("")

    # Cross-lang-pair split intersections
    logger.info("=== Split domain intersections across lang pairs ===")
    for split in ("train", "validation", "test"):
        for idx, lang_pair_1 in enumerate(all_lang_pairs):
            for lang_pair_2 in all_lang_pairs[idx + 1 :]:
                domains_1 = split_domains[lang_pair_1].get(split, set())
                domains_2 = split_domains[lang_pair_2].get(split, set())
                overlap = sorted(domains_1 & domains_2)
                logger.info(
                    f"  {split} {lang_pair_1} ∩ {lang_pair_2}: {len(overlap)} domains"
                )
    logger.info("")

    # Cross-split intersections (should be empty by design)
    logger.info("=== Cross-split intersections (should be empty) ===")
    for split_a, split_b in [
        ("train", "test"),
        ("train", "validation"),
        ("validation", "test"),
    ]:
        domains_a = {
            domain for domain, assigned in assignments.items() if assigned == split_a
        }
        domains_b = {
            domain for domain, assigned in assignments.items() if assigned == split_b
        }
        overlap = sorted(domains_a & domains_b)
        logger.info(
            f"  {split_a} ∩ {split_b}: {len(overlap)} domains {overlap if overlap else ''}"
        )


def drop_columns_and_fix_date_format(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        col for col in df.columns if col.startswith("fulltext_joined")
    ] + ["key_0"]
    df = df.drop(columns=columns_to_drop)

    date_cols = [col for col in df.columns if col.startswith("date")]
    for date_col in date_cols:
        df[date_col] = (
            pd.to_datetime(df[date_col], unit="ms", utc=True)
            .dt.tz_convert("Europe/Oslo")
            .dt.strftime("%Y-%m-%d")
        )

    df = df.fillna("")
    return df


if __name__ == "__main__":
    log_level = "INFO"
    setup_logging(source_script="restructure_aligned_superset", log_level=log_level)

    superset_path = Path("data/output/målfrid_superset_aligned_raw")
    dataset_output_path = Path("data/målfrid_parallel")
    dataset_repo_id = "NbAiLab/maalfrid_parallel"
    private = True
    domain_counts = get_domain_counts(superset_path)

    if log_level == "DEBUG":
        log_domain_counts(domain_counts)

    assignments = assign_domains_to_splits(domain_counts)

    if log_level == "DEBUG":
        log_split_statistics(domain_counts, assignments)

    logger.debug("assignments: %s", assignments)
    split_to_domains = defaultdict(list)
    for domain, split in assignments.items():
        split_to_domains[split].append(domain)

    for lang_pair_dir in superset_path.iterdir():
        for split_name, domain_list in split_to_domains.items():
            output_file = (
                dataset_output_path / lang_pair_dir.name / f"{split_name}.jsonl"
            )
            if output_file.exists():
                continue

            jsonl_files = [
                lang_pair_dir / f"{domain}_{lang_pair_dir.name}.jsonl"
                for domain in domain_list
                if (lang_pair_dir / f"{domain}_{lang_pair_dir.name}.jsonl").exists()
            ]

            df = pd.concat(
                [pd.read_json(jsonl_file, lines=True) for jsonl_file in jsonl_files]
            )
            df = drop_columns_and_fix_date_format(df)

            output_file.parent.mkdir(exist_ok=True, parents=True)
            df.to_json(output_file, lines=True, orient="records", index=False)

        lang_pair_dataset_dir = dataset_output_path / lang_pair_dir.name
        ds = load_dataset(str(lang_pair_dataset_dir))
        ds.push_to_hub(
            repo_id=dataset_repo_id, config_name=lang_pair_dir.name, private=private
        )
        ds = load_dataset(dataset_repo_id, name=lang_pair_dir.name)
        logger.info("Successfully pushed dataset to huggingface! %s", ds)
