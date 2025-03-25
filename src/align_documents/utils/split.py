import transformers
from semchunk import semchunk

def chunk_texts(
    texts: list[str],
    tokenizer: str | transformers.PreTrainedTokenizer | tokenizers.Tokenizer,
    chunk_max_len: int,
) -> list[list[str]]:
    """Returns sentences as lists of model_max_len sentence chunks
    """
    chunker = semchunk.chunkerify(tokenizer, chunk_size=chunk_max_len)
    chunked_texts = chunker(texts)

    return chunked_texts

