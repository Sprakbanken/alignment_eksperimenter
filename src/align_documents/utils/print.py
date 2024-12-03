from align_documents.types import Match
import numpy as np
from sentence_transformers import util


def print_matches(matches: list[tuple[int, Match]], query_texts: list[str], corpus_texts: list[str], stop_printing_at=-1) -> None:
    print_i = 0
    for query_i, e in matches:
        if print_i == stop_printing_at:
            return
        
        query_text = query_texts[query_i]

        match_i = e["corpus_id"]
        match_text = corpus_texts[match_i]

        score = e["score"]

        print_s = (
            f"Søketekst: {query_text[:300]}\n"
            f"Match: {match_text[:300]}\n"
            f"Likhet: {score}\n"
            f"Indekser: Nynorsk/søketekst: {query_i}, Bokmål/treff: {match_i}\n"
            f"_______________________________________________________________"
        )

        print(print_s)
        print_i += 1

def print_misses(misses: set[tuple[int, int]], search_result: list[list[Match]], bokmål_texts: list[str], nynorsk_texts: list[str], bokmål_embeddings: np.array, nynorsk_embeddings: np.array, threshold:float) -> None:
    for nn_i, bm_i in misses:
        match_i = search_result[nn_i][0]["corpus_id"]
        match_score = search_result[nn_i][0]["score"]
        
        if match_i != bm_i:
            print("Likeste søketreff er et annet dokument enn fasit\n")
            print(f"Fasit indeks: {bm_i}\nTreff indeks: {match_i}")
            print(f"Søketekst og fasit likhet: {float(util.cos_sim(nynorsk_embeddings[nn_i], bokmål_embeddings[bm_i]))}")
            print(f"Søketekst og match likhet: {match_score}")
            print(f"Treff og fasit likhet {float(util.cos_sim(bokmål_embeddings[bm_i], bokmål_embeddings[match_i]))}\n\n")

            print(f"Nynorsk søketekst:\n\t{nynorsk_texts[nn_i][:300]}\n_____")
            print(f"Bokmål søketreff:\n\t{bokmål_texts[match_i][:300]}\n_____")
            print(f"Bokmål fasit:\n\t{bokmål_texts[bm_i][:300]}\n_____")

        else:
            print("Likeste søketreff er det samme som fasiten, men similarity score var under terskelen\n")
            assert match_score <= threshold
