import transformers

def print_matches(matches, query_texts: list[str], corpus_texts: list[str], stop_printing_at=-1) -> None:
    print_i = 0
    for query_i, e in matches:
        if print_i == stop_printing_at:
            return
        
        query_text = query_texts[query_i]

        match_i = e["corpus_id"]
        match_text = corpus_texts[match_i]

        score = e["score"]

        print(f"""
Søketekst: 
        {query_text[:300]}
Match:
        {match_text[:300]}
Likhet:
        {score}
        
Indekser:
            Nynorsk/søketekst:  {query_i}
            Bokmål/treff:       {match_i}
_______________________________________________________________""")
        print_i += 1

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