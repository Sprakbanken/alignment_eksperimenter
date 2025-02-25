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

def tokenize_and_split_text(text:str, tokenizer: transformers.BertTokenizerFast, model_max_len: int, separators = ["?", "!", "."]) -> list[str]:
    ids = tokenizer(text)["input_ids"]
    tokenized_text = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
    if len(tokenized_text) <= model_max_len:
        return [text] 
    
    split_start_index = 0
    split_indices = []
    while split_start_index != len(tokenized_text):
        start, end = split_on_separator_tokens(tokenized_text, split_start_index, max_len=model_max_len, separator_tokens=separators)
        split_indices.append((start, end))

        if end + model_max_len >= len(tokenized_text):
            start = end 
            end = len(tokenized_text)
            split_indices.append((start, end))

        split_start_index = end

    texts = [tokenizer.convert_tokens_to_string(tokenized_text[start:end]) for (start, end) in split_indices]
    texts = [t for t in texts if t]
    return texts

def split_on_separator_tokens(tokens: list[str], split_start_index: int, max_len:int, separator_tokens: list[str]) -> tuple[int, int]:
    max_split_index = split_start_index + max_len

    for i, token in enumerate(reversed(tokens[split_start_index:max_split_index])):
        current_split_index = max_split_index - i 
        if token in separator_tokens and len(tokens[split_start_index:current_split_index]) <= max_len:
            return (split_start_index, current_split_index)

    # could not find a separator that split the text into suitable length, split on word boundary instead
    return split_on_word_boundary(tokens, split_start_index, max_len)

def split_on_word_boundary(tokens:list[str], split_start_index:int, max_len:int) -> tuple[int, int]:
    max_split_index = split_start_index + max_len

    current_split_index = max_split_index
    token = tokens[current_split_index]

    while token.startswith("##") and current_split_index > split_start_index:
        current_split_index -= 1
        token = tokens[current_split_index]

    if current_split_index == split_start_index:
        # all tokens in span are part of the same word, return max split index
        return (split_start_index, max_split_index+1)
    return (split_start_index, current_split_index+1)
