import transformers
from semchunk import semchunk
import logging

logger = logging.getLogger(__name__)

def chunk_texts(
    texts: list[str],
    tokenizer: transformers.PreTrainedTokenizer,
    chunk_max_len: int,
) -> list[list[str]]:
    """Returns sentences as lists of model_max_len sentence chunks
    """
    logger.debug("Chunking...") # chunker's internal tqdm has no description
    chunker = semchunk.chunkerify(tokenizer, chunk_size=chunk_max_len)
    chunked_texts = chunker(texts, progress=True)

    return chunked_texts

