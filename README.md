

# Hvordan kjøre filene her:
Du trenger python3.11 og pip

Lag et virtuelt pythonmiljø med f.eks venv:  
```bash
python3 -m venv <navn-på-miljø>     # lag miljø
. <navn-på-miljø>/bin/activate      # aktiver miljø
pip install .                       # installer pakker og moduler
```

eller bruk f.eks pdm:  

```bash
pdm install
```

Da kan du kjøre alignment-pipelinen slik:
```bash
python3 -m align_documents.align_all
```

eller  
```bash
pdm run python -m align_documents.align_all
```

## Scripts
The script `scripts/find_negative_documents.py` finds pairs of documents that are pretty similar, but expected to not be actually parallell (i.e above min_threshold, but less than the match threshold in `alignment_config.yaml`). We use this to find assumed negative document pairs for our manually anntoated documents.



## Dev setup

### Pre-commit

Kjør `pdm run pre-commit install` eller `pre-commit install` for å sette opp pre-commit første gang. Deretter vil pre-commit hooks kjøre hver gang du skriver git commit, og evt hindre deg i å commite hvis ikke hooksene passer. (Per idag har vi en ruff-hook som vil gjøre koden compliant med PEP 8)