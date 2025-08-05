# Alignment experiments
Some experiments with bitext mining based on data from the Målfrid project.  
The goal is to get more high quality Norwegian Nynorsk-Bokmål and Norwegian Nynorsk-English parallel data. 

## Install/setup
You can easily install this project with tools like pdm or uv. 
```bash
uv sync
# OR
pdm install
```
Alternatively, manually create a virtual environment and install with pip: 
```
python3 -m venv .venv
source .venv/bin/activate
pip install . 
```

## Run alignment pipeline 

These are equivalent (prefix with `uv run` or `pdm run`)
```bash
align_all
python -m align_documents
python -m align_documents.align_all
```

The alignment pipeline expects a config file. See our sample config:   

```
data_dir = "data/maalfrid_2025"                     # Path to the directory containing documents to align 
output_dir = "data/output/maalfrid_2025_aligned"    # Path to the output directory where aligned docs will be stored
embedding_model = "BAAI/bge-m3"                     # Sentence embedding model to use (local path or huggingface hub repo id)
embedding_dir = "data/maalfrid_2025_embeddings"     # Directory to store/read embeddings
batch_size = 8                                      # Batch size for encoding documents
aggregation_strategy = "mean"                       # Aggregation strategy for document embeddings (when input is longer than models max_len)
match_threshold = 0.95                              # Threshold for matching documents
languages = ["nno", "nob"]                          # Languages to align (must be a list of length 2)
number_to_letter_ratio = 0.3                        # Ratio of numbers to letters in the document (discard if greater)
min_document_length = 100                           # Minimum number of characters in a document (discard if less)
```
(also at [alignment_config.toml](alignment_config.toml))

## Run info script
This script will read the alignment config file and calculate info about the source data for alignment

These are equivalent (prefix with `uv run` or `pdm run`)
```bash
info
python -m align_documents.info
```


## Other scripts
`scripts/scripts/dowload_målfrid.py` downloads målfrid data from the [resource catalogue](https://www.nb.no/sprakbanken/ressurskatalog/?_search=m%C3%A5lfrid). Input to the script is the specific målfrid resource url (e.g. https://www.nb.no/sbfil/tekst/maalfrid_2025/maalfrid_2025.tar for 2025 data or https://www.nb.no/sbfil/tekst/maalfrid_2022/maalfrid_2022.tar.gz for 2022 data) and a path to the directory store the data

The script `scripts/find_negative_documents.py` finds pairs of documents that are pretty similar, but expected to not be actually parallell (i.e above min_threshold, but less than the match threshold in `alignment_config.yaml`). We use this to find assumed negative document pairs for our manually anntoated documents.




## Dev setup

### Pre-commit

Run `pre-commit install` (or `python -m pre-commit install`) to set up pre-commit first time.
Then, the pre-commit hooks will run each time you create a commit.

