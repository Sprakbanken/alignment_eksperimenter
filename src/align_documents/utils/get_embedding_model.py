from sentence_transformers import SentenceTransformer
import torch
import logging


logger = logging.getLogger(__name__)


def get_embedding_model(embedding_model_id: str) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    embedding_model = SentenceTransformer(embedding_model_id, device=device)
    return embedding_model
