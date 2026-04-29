from pathlib import Path
from align_documents.compare_alignment import compare_alignment, calculate_scores
from align_documents.utils.config import Config
from align_documents.utils import setup_logging
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/output/grid_search")
# todo: add more models
EMBEDDING_MODEL_LIST = [
    # "NbAiLab/nb-sbert-v2-base",
    "microsoft/harrier-oss-v1-0.6b",
    # "codefuse-ai/F2LLM-v2-1.7B",
]
AGGREGATION_STRATEGY = ["mean", "cut-off"]
THRESHOLDS = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
MODES = ["strict", "lenient"]

if __name__ == "__main__":
    setup_logging("grid_search", log_level="INFO")

    gold_data_path = Path("manual_annotation/annotated_data/gold_data.csv")
    gd = pd.read_csv(gold_data_path)

    for (lang1, lang2), group in gd.groupby(["lang_lang_1", "lang_lang_2"]):
        for mode in MODES:
            logger.info("Comparing %s and %s in %s mode", lang1, lang2, mode)
            logger.debug("Unique annotations: %s", group.annotation.unique())

            positive_pairs = [
                (tup.fulltext_joined_lang_1, tup.fulltext_joined_lang_2)
                for tup in group[group.annotation == "Parallell"].itertuples(
                    index=False
                )
            ]
            negative_pairs = [
                (tup.fulltext_joined_lang_1, tup.fulltext_joined_lang_2)
                for tup in group[group.annotation == "Not parallell"].itertuples(
                    index=False
                )
            ]
            almost_parallel_pairs = [
                (tup.fulltext_joined_lang_1, tup.fulltext_joined_lang_2)
                for tup in group[group.annotation == "Almost parallel"].itertuples(
                    index=False
                )
            ]
            if mode == "strict":
                negative_pairs = negative_pairs + almost_parallel_pairs
            else:
                positive_pairs = positive_pairs + almost_parallel_pairs

            logger.info(
                "Group size: %d, positive pairs: %d, negative pairs: %d",
                len(group),
                len(positive_pairs),
                len(negative_pairs),
            )

            for embedding_model in EMBEDDING_MODEL_LIST:
                model_output_dir = OUTPUT_DIR / embedding_model
                model_output_dir.mkdir(parents=True, exist_ok=True)
                for aggregation_strategy in AGGREGATION_STRATEGY:
                    embedding_dir = Path("data/embeddings") / embedding_model
                    embedding_dir.mkdir(parents=True, exist_ok=True)
                    for threshold in THRESHOLDS:
                        scores_path = (
                            model_output_dir
                            / f"scores_{aggregation_strategy}_{threshold:.2f}_{lang1}_{lang2}_{mode}.json"
                        )
                        if scores_path.exists():
                            logger.info("%s already exists, skipping", scores_path)
                            continue

                        config = Config(
                            embedding_model=embedding_model,
                            aggregation_strategy=aggregation_strategy,
                            match_threshold=threshold,
                            embedding_dir=embedding_dir,
                            languages=(lang1, lang2),
                            data_dir=Path(""),
                            output_dir=model_output_dir,
                            batch_size=4,
                            number_to_letter_ratio=0,
                            min_document_length=0,
                        )
                        confusion_matrix = compare_alignment(
                            config,
                            positive_pairs=positive_pairs,
                            negative_pairs=negative_pairs,
                            filename_prefix=f"{mode}_{lang1}_{lang2}",
                        )
                        scores = calculate_scores(confusion_matrix)

                        scores_path.write_text(json.dumps(scores))
                        logger.info(
                            "Embedding model: %s, Aggregation strategy: %s, Threshold: %.2f, Scores: %s",
                            embedding_model,
                            aggregation_strategy,
                            threshold,
                            scores,
                        )
