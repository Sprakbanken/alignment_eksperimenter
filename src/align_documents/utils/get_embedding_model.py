from sentence_transformers import SentenceTransformer
import torch
import logging


logger = logging.getLogger(__name__)


def get_embedding_model(embedding_model_id: str) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    embedding_model = SentenceTransformer(embedding_model_id, device=device)

    basemodel_max_len = embedding_model[0].auto_model.config.max_position_embeddings
    if basemodel_max_len != embedding_model.get_max_seq_length():
        logger.info(
            "Setting max_seq_length to %s (was %s)",
            basemodel_max_len,
            embedding_model.get_max_seq_length(),
        )
        embedding_model.max_seq_length = basemodel_max_len

    return embedding_model