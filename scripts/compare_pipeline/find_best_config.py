from align_documents.utils import setup_logging
from pathlib import Path
import argparse
import json
import logging
import re

logger = logging.getLogger(__name__)

GRID_SEARCH_DIR = Path("data/output/grid_search")
FILENAME_RE = re.compile(
    r"^scores_(?P<agg>.+)_(?P<thr>\d\.\d{2})_(?P<lang1>[a-z]{3})_(?P<lang2>[a-z]{3})_(?P<mode>strict|lenient)\.json$"
)


def parse_score_file(path: Path):
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    embedding_model = str(path.parent.relative_to(GRID_SEARCH_DIR))
    scores = json.loads(path.read_text())
    return {
        "embedding_model": embedding_model,
        "aggregation_strategy": m["agg"],
        "threshold": float(m["thr"]),
        "lang1": m["lang1"],
        "lang2": m["lang2"],
        "mode": m["mode"],
        **scores,
    }


def get_args():
    parser = argparse.ArgumentParser(
        description="Find best alignment config per (language pair, mode) from grid search outputs"
    )
    parser.add_argument(
        "-m",
        "--metric",
        default="f1",
        choices=["f1", "precision", "recall", "accuracy"],
        help="Metric to optimize",
    )
    parser.add_argument(
        "-d",
        "--grid_search_dir",
        type=Path,
        default=GRID_SEARCH_DIR,
        help="Directory containing grid search score files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional path to write best configs as JSON",
    )
    parser.add_argument(
        "-l",
        "--log_level",
        default="INFO",
        choices=["INFO", "DEBUG", "WARNING", "ERROR"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    setup_logging("find_best_config", log_level=args.log_level)

    results = []
    for path in args.grid_search_dir.rglob("scores_*.json"):
        entry = parse_score_file(path)
        if entry is None:
            logger.warning("Skipping unrecognized file: %s", path)
            continue
        results.append(entry)
    logger.info("Loaded %d score files", len(results))

    groups: dict[tuple[str, str, str], list[dict]] = {}
    for r in results:
        key = (r["lang1"], r["lang2"], r["mode"])
        groups.setdefault(key, []).append(r)

    best_per_group = {}
    rows = []
    for key in sorted(groups):
        lang1, lang2, mode = key
        best = max(
            groups[key],
            key=lambda r: (r[args.metric], r["f1"], r["precision"]),
        )
        best_per_group[f"{lang1}_{lang2}_{mode}"] = best
        rows.append(
            (
                f"{lang1}-{lang2}",
                mode,
                best["embedding_model"],
                best["aggregation_strategy"],
                f"{best['threshold']:.2f}",
                f"{best['f1']:.3f}",
                f"{best['precision']:.3f}",
                f"{best['recall']:.3f}",
                f"{best['accuracy']:.3f}",
            )
        )

    headers = (
        "langs",
        "mode",
        "embedding_model",
        "agg",
        "thr",
        "f1",
        "precision",
        "recall",
        "accuracy",
    )
    widths = [
        max(len(str(row[i])) for row in (rows + [headers])) for i in range(len(headers))
    ]
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    print(f"\nBest config per (language pair, mode), optimizing {args.metric}:\n")
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(best_per_group, indent=2))
        logger.info("Wrote best configs to %s", args.output)
