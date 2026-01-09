from datasets import load_dataset
from align_documents.compare_alignment import compare_alignment, calculate_scores
from align_documents.utils.config import get_config
from align_documents.utils import setup_logging
import json
from pathlib import Path
import argparse
import logging

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(
        description="Compare alignment pipeline to NPK dataset"
    )
    parser.add_argument(
        "-c",
        "--config_file",
        help="Path to the config file for alignment",
        type=Path,
        default=Path("alignment_config.toml"),
    )
    parser.add_argument(
        "-l",
        "--log_level",
        help="Log level",
        default="INFO",
        choices=["INFO", "DEBUG", "WARNING", "ERROR"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    setup_logging("compare_pipeline_to_npk", log_level=args.log_level)
    logger.info(args)

    config = get_config(config_file=args.config_file)
    logger.info(config)

    dataset_name = "NbAiLab/ndla_npk_conversational_nb_to_nn"

    output_dir = Path("data/output/comparison") / dataset_name
    output_dir.mkdir(exist_ok=True, parents=True)

    dataset = load_dataset(dataset_name, split="validation")
    logger.info("Dataset %s loaded with %s examples", dataset, len(dataset))

    positive_pairs = [(e["nn"], e["nb"]) for e in dataset]

    logger.info("Starting alignment comparison...")
    conf_matrix = compare_alignment(
        config=config,
        positive_pairs=positive_pairs,
        filename_prefix="-".join(dataset_name.split("/")),
    )
    logger.debug("Confusion matrix: %s", conf_matrix)

    scores = calculate_scores(confusion_matrix=conf_matrix)
    logger.info("Calculated scores: %s", scores)

    outfile = output_dir / "scores.json"
    with outfile.open("w+") as f:
        f.write(json.dumps(scores, indent=4))
    logger.info("Scores saved to: %s", outfile)
