# Alignment experiments
Some experiments with bitext mining based on data from the Målfrid project.  
The goal is to get more high quality Norwegian Nynorsk-Bokmål and Norwegian Nynorsk-English parallel data. 

## Install/setup
You can easily install this project with tools like pdm or uv. 
```bash
uv sync         # OR
pdm install
```
Alternatively, manually create a virtual environment and install with pip: 
```
python3 -m venv .venv
source .venv/bin/activate
pip install . 
```

All examples on how to run the code in this README show how to run with uv, modify accordingly to your installation.

# Running the repo code 
All scripts and modules support --help flag for cli arguments explanation

## Download målfrid data
The script `scripts/download_målfrid.py` downloads målfrid data from the [resource catalogue](https://www.nb.no/sprakbanken/ressurskatalog/?_search=m%C3%A5lfrid). 

Input to the script is the specific målfrid resource url (e.g. https://www.nb.no/sbfil/tekst/maalfrid_2025/maalfrid_2025.tar for 2025 data).  
Running the script will create a `maalfrid_<year>` directory in data_dir with files following this format: `<domain-name>_<language>_<mimetype>.jsonl`. 

Example run: 
```bash
uv run scripts/download_målfrid.py --url https://www.nb.no/sbfil/tekst/maalfrid_2022/maalfrid_2022.tar.gz --data_dir data/
``` 

This will create `data/maalfrid_2022` with files such as `1700-tallet.no_nob_html.jsonl`, `yrkesfisker.no_eng_doc.jsonl`, or `velgekte.no_nob_pdf.jsonl`.  

## Run alignment pipeline on målfrid data

These are equivalent:

```bash
align_all                               # OR
uv run -m align_documents               # OR
uv run -m align_documents.align_all
```

The alignment pipeline expects a config file. See our sample config:   

```
data_dir = "data/maalfrid_2025"                     # Path to the directory containing documents to align 
output_dir = "data/output/maalfrid_2025"            # Path to the output directory where script outputs are stored
embedding_model = "BAAI/bge-m3"                     # Sentence embedding model to use (local path or huggingface hub repo id)
embedding_dir = "data/maalfrid_2025_embeddings"     # Directory to read/write document embeddings
batch_size = 8                                      # Batch size when encoding documents
aggregation_strategy = "mean"                       # "cut-off" or "mean". Aggregation strategy for document embeddings (when input is longer than model max_len)
match_threshold = 0.95                              # Cosine similarity threshold for matching documents
languages = ["nno", "nob"]                          # Languages to align (must be a list of length 2)
number_to_letter_ratio = 0.3                        # Ratio of numbers to letters in the document (discard if greater)
min_document_length = 100                           # Minimum number of characters in a document (discard if less)
```
(also at [alignment_config.toml](alignment_config.toml))

The aligned documents will be stored in `<output_dir>/aligned/` with filenames on this format: `<domain-name>_<language1>_<language_2>.jsonl`.  
Example: `data/output/maalfrid_2023/aligned/valg.no_nno_nob.jsonl` or `data/output/maalfrid_2025/aligned/skatteetaten.no_eng_nob.jsonl` 


## Run info script
This script will read the alignment config file and calculate info about the source data for alignment (i.e the content of `data_dir`)

These are equivalent:
```bash
info                            # OR
uv run -m align_documents.info
```

## Find negative document pairs
The script `scripts/find_negative_documents.py` finds pairs of documents that are pretty similar, but expected to not be actually parallell (i.e above min_threshold, but less than the match threshold in `alignment_config.yaml`). We use this to find assumed negative document pairs for our manually annotated documents.

The output files are the same format as the alignment pipeline, but stored in `<output_dir>/negative_pairs/` instead of `<output_dir>/aligned/`

Example run
```bash
uv run scripts/find_negative_documents.py --config_file alignment_config.toml --min_threshold 0.5  --pairs_per_website 5 --total_pairs 50
```

# Dev setup

## Pre-commit

Run `uv run pre-commit install` (or `python -m pre-commit install`) to set up pre-commit first time.
Then, the pre-commit hooks will run each time you create a commit.

