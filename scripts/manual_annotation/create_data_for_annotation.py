import jsonlines
import logging
from pathlib import Path
import argparse
from align_documents.utils import setup_logging
from collections import defaultdict
import pandas as pd

logger = logging.getLogger(__name__)


def main(args):
    args.output_dir.mkdir(exist_ok=True)
    setup_logging(
        "dataset_for_manual_annotation",
        log_level=args.log_level,
        log_dir=args.output_dir,
    )

    logger.info(
        "Finding all domains with aligned document pairs for each language pair "
    )
    pos_domains = defaultdict(set)
    for aligned_docs_file in args.positive_pairs_dir.glob("*.jsonl"):
        domain, lang1, lang2 = aligned_docs_file.stem.rsplit("_", maxsplit=2)
        lang1, lang2 = sorted([lang1, lang2])
        pos_domains[f"{lang1}_{lang2}"].add(domain)

    logger.info("Aligned docs language pairs %s", pos_domains.keys())
    logger.debug("Full aligned doc pairs dict %s", pos_domains)

    logger.info(
        "Finding all domains with unaligned document pairs for each language pair "
    )
    neg_domains = defaultdict(set)
    for unaligned_docs_file in args.negative_pairs_dir.glob("*.jsonl"):
        domain, lang1, lang2 = unaligned_docs_file.stem.rsplit("_", maxsplit=2)
        lang1, lang2 = sorted([lang1, lang2])
        neg_domains[f"{lang1}_{lang2}"].add(domain)

    logger.info("Unaligned docs language pairs %s", neg_domains.keys())
    logger.debug("Full unaligned doc pairs dict %s", neg_domains)

    if not (neg_domains and pos_domains):
        logger.warning("Couldn't find both positive and negative domains. Exiting.")
        exit(1)

    pos_domains, neg_domains = find_overlapping_domains(
        pos_domains, neg_domains, target_domains=args.num_target_domains
    )
    logger.debug("Selected pos domains: %s", pos_domains)
    logger.debug("Selected neg domains: %s", neg_domains)

    outfile = args.output_dir / "data_to_annotate.jsonl"
    logger.info("Read 1 line from each document pair and write to %s", outfile)

    doc_hashes_to_skip = set()
    if args.doc_hashes:
        logger.info("Reading doc hashes to skip from file")
        doc_hashes_to_skip = set(args.doc_hashes.read_text().splitlines())

    with jsonlines.open(outfile, "w") as f:
        for lang_pair, domain_set in sorted(pos_domains.items()):
            logger.debug(
                "Positive pairs: Lang pair: %s, domain set: %s", lang_pair, domain_set
            )
            lang_1, lang_2 = lang_pair.split("_")
            for domain in sorted(domain_set):
                line = get_line(
                    input_dir=args.positive_pairs_dir,
                    lang_1=lang_1,
                    lang_2=lang_2,
                    domain=domain,
                    skip_doc_hashes=doc_hashes_to_skip,
                )
                if not line:
                    continue
                # Add doc hashes from chosen document pair to doc_hashes_to_skip, to avoid duplicates in data for annotation
                doc_hashes_to_skip.add(line["doc_hash_lang_1"])
                doc_hashes_to_skip.add(line["doc_hash_lang_2"])

                f.write({"assumed_aligned": True, **line})

        for lang_pair, domain_set in sorted(neg_domains.items()):
            logger.debug(
                "Negative pairs: Lang pair: %s, domain set: %s", lang_pair, domain_set
            )

            lang_1, lang_2 = lang_pair.split("_")

            for domain in sorted(domain_set):
                line = get_line(
                    input_dir=args.negative_pairs_dir,
                    lang_1=lang_1,
                    lang_2=lang_2,
                    domain=domain,
                    skip_doc_hashes=doc_hashes_to_skip,
                )
                if not line:
                    continue

                # Add doc hashes from chosen document pair to doc_hashes_to_skip, to avoid duplicates in data for annotation
                doc_hashes_to_skip.add(line["doc_hash_lang_1"])
                doc_hashes_to_skip.add(line["doc_hash_lang_2"])

                f.write({"assumed_aligned": False, **line})

    logger.info(
        "Split file into %s files for annotation and save as .csv files",
        args.num_outfiles,
    )
    df = pd.read_json(outfile, lines=True).drop(columns=["assumed_aligned"])
    # shuffle df (so every annotator gets a variety of languages and pos/neg document pairs)
    df = df.sample(frac=1, random_state=42)

    # split data to annotate into args.num_outfiles parts
    num_lines_per_file = len(df) // args.num_outfiles
    logger.debug("Length of each file: %s", num_lines_per_file)
    for i in range(args.num_outfiles):
        sub_df = df[
            i * num_lines_per_file : i * num_lines_per_file + num_lines_per_file
        ]
        sub_df.to_csv(args.output_dir / f"data_to_annotate_part_{i}.csv", index=False)

    logger.info("Done. See output at %s", args.output_dir)


def get_args():
    parser = argparse.ArgumentParser(prog="Create balanced dataset")

    parser.add_argument(
        "--positive_pairs_dir",
        help="Path to the directory containing assumed positive doc pairs",
        type=Path,
        default=Path("data/output/maalfrid_2025/aligned"),
    )
    parser.add_argument(
        "--negative_pairs_dir",
        help="Path to the directory containing assumed negative doc pairs",
        type=Path,
        default=Path("data/output/maalfrid_2025/negative_pairs"),
    )
    parser.add_argument(
        "--output_dir",
        help="Place to store dataset for manual annotation",
        type=Path,
        default=Path("data/output/data_for_manual_annotation"),
    )
    parser.add_argument(
        "--num_target_domains",
        type=int,
        help="Number of domains to find document pairs from",
        default=20,
    )
    parser.add_argument(
        "--num_outfiles",
        type=int,
        default=3,
        help="Number of .csv-files to split the dataset into for annotation",
    )
    parser.add_argument(
        "--doc_hashes",
        type=Path,
        default=None,
        help="Newline separates file with doc hashes to skip when creating dataset to annotate",
    )
    parser.add_argument(
        "-l",
        "--log_level",
        help="Log level",
        default="INFO",
        choices=["INFO", "DEBUG", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def find_overlapping_domains(
    pos_domains: dict[str, set[str]],
    neg_domains: dict[str, set[str]],
    target_domains: int,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Find target_domains domains that are present in all language pairs, both assumed positive (aligned) and assumed negative document pairs.
    If not enough domains fulfill those criteria, supplement with domains that are present in both aligned and negative document pairs for each language pair.
    If not enough domains fulfill those criteria, supplement with domains that are present in either postive or negative document pairs for each language.
    """

    overlapping_lang_pairs = set(pos_domains) & set(neg_domains)
    if len(overlapping_lang_pairs) < len(pos_domains) or len(
        overlapping_lang_pairs
    ) < len(neg_domains):
        logger.info("Overlapping lang pairs: %s", overlapping_lang_pairs)

    intersecting_domains = None
    for lang_pair in overlapping_lang_pairs:
        if intersecting_domains is None:
            intersecting_domains = pos_domains[lang_pair]
        intersecting_domains &= pos_domains[lang_pair]
        intersecting_domains &= neg_domains[lang_pair]

    if len(intersecting_domains) >= target_domains:
        logger.info(
            "Found %s overlapping domains across all language pairs and pos/neg alignment",
            target_domains,
        )
        selected_domains = sorted(intersecting_domains)[:target_domains]
        logger.debug("Selected domains: %s", selected_domains)
        pos_domains = neg_domains = {
            lang_pair: set(selected_domains) for lang_pair in overlapping_lang_pairs
        }
        return (pos_domains, neg_domains)

    logger.info(
        "Not enough overlapping domains. Found %s domains that overlap across all assumed positive and assumed negative document pairs.",
        len(intersecting_domains),
    )

    num_missing_domains = target_domains - len(intersecting_domains)

    for lang_pair in overlapping_lang_pairs:
        positive_domains = pos_domains[lang_pair] - intersecting_domains
        negative_domains = neg_domains[lang_pair] - intersecting_domains

        lang_pair_intersecting_domains = positive_domains & negative_domains

        if len(lang_pair_intersecting_domains) >= num_missing_domains:
            selected_domains = sorted(lang_pair_intersecting_domains)[
                :num_missing_domains
            ]
            pos_domains[lang_pair] = intersecting_domains.union(selected_domains)
            neg_domains[lang_pair] = intersecting_domains.union(selected_domains)
        else:
            logger.info(
                "Could not find enough overlapping domains between positive and negative alignment for %s",
                lang_pair,
            )
            lang_pair_num_missing = num_missing_domains - len(
                lang_pair_intersecting_domains
            )
            pos_domains_to_add = sorted(
                positive_domains - lang_pair_intersecting_domains
            )[:lang_pair_num_missing]

            if len(pos_domains_to_add) < lang_pair_num_missing:
                logger.warning("Not enough domains in %s aligned documents", lang_pair)

            neg_domains_to_add = sorted(
                negative_domains - lang_pair_intersecting_domains
            )[:lang_pair_num_missing]

            if len(neg_domains_to_add) < lang_pair_num_missing:
                logger.warning("Not enough domains in %s negative pairs", lang_pair)

            pos_domains[lang_pair] = intersecting_domains.union(
                lang_pair_intersecting_domains
            ).union(pos_domains_to_add)

            neg_domains[lang_pair] = intersecting_domains.union(
                lang_pair_intersecting_domains
            ).union(neg_domains_to_add)

    return (pos_domains, neg_domains)


def get_line(
    input_dir: Path, lang_1: str, lang_2: str, domain: str, skip_doc_hashes: set[str]
) -> dict[str, str]:
    """Get a line of a document pair json file from input_dir, skipping lines where any document doc hash is in skip_doc_hashes"""

    columns_to_keep = [
        "doc_hash",
        "lang",
        "url",
        "domain",
        "date",
        "mimetype",
        "fulltext",
        "fulltext_joined",
    ]

    filename = input_dir / f"{domain}_{lang_1}_{lang_2}.jsonl"
    # Language pair order is not consistently sorted
    if not filename.exists():
        filename = input_dir / f"{domain}_{lang_2}_{lang_1}.jsonl"

    for line in jsonlines.open(filename):
        if (
            line[f"doc_hash_{lang_1}"] in skip_doc_hashes
            or line[f"doc_hash_{lang_2}"] in skip_doc_hashes
        ):
            logger.debug(
                "Skipping line %s %s",
                line[f"doc_hash_{lang_1}"],
                line[f"doc_hash_{lang_2}"],
            )
            continue

        return {
            **{
                f"{column}_lang_1": line[f"{column}_{lang_1}"]
                for column in columns_to_keep
            },
            **{
                f"{column}_lang_2": line[f"{column}_{lang_2}"]
                for column in columns_to_keep
            },
        }
    logger.warning("Couldn't find a line in %s", filename)
    return {}


if __name__ == "__main__":
    args = get_args()
    main(args)
