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

The alignment pipeline expects a config file. See our sample config at [alignment_config.toml](alignment_config.toml)  
**TODO**: explain config file parameters

## Run info script
This script will read the alignment config file and calculate info about the source data for alignment

These are equivalent (prefix with `uv run` or `pdm run`)
```bash
info
python -m align_documents.info
```


## Other scripts
The script `scripts/find_negative_documents.py` finds pairs of documents that are pretty similar, but expected to not be actually parallell (i.e above min_threshold, but less than the match threshold in `alignment_config.yaml`). We use this to find assumed negative document pairs for our manually anntoated documents.



## Dev setup

### Pre-commit

Kjør `pdm run pre-commit install` eller `pre-commit install` for å sette opp pre-commit første gang. Deretter vil pre-commit hooks kjøre hver gang du skriver git commit, og evt hindre deg i å commite hvis ikke hooksene passer. (Per idag har vi en ruff-hook som vil gjøre koden compliant med PEP 8)


## Data

### Nynorsk-bokmål alignment

- 2021-data
    - Se mappa `eksperimenter_2021_data` 

- 2023-data
    - TBA

### Nynorsk-engelsk alignment
- TBA
