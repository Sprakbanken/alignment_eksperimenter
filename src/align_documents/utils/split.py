import transformers
from sentence_transformers import SentenceTransformer
import tokenizers
from semchunk import semchunk

def tokenize_sentences(
    sentences: list[str],
    model: SentenceTransformer | None = None,
    tokenizer: str | transformers.PreTrainedTokenizer | tokenizers.Tokenizer | None = None,
    chunk_max_len: int | None = None,
) -> list[list[str]]:
    """Returns sentences as lists of model_max_len sentence chunks
    """
    if model is not None:
        if chunk_max_len is None:
            chunk_max_len = model.get_max_seq_length()
        if tokenizer is None:
            tokenizer = model.tokenizer

    chunker = semchunk.chunkerify(tokenizer, chunk_size=chunk_max_len)
    chunked_sentences = chunker(sentences)

    return chunked_sentences

