# Alignment eksperimenter

Forskjellige eksperimenter med text alignment med data fra Målfrid.

Formålet er å få mer nynorsk-bokmål og nynorsk-engelsk parallelldata av høy kvalitet.

## Installering

Du trenger python3.11 og pip.

Lag et virtuelt pythonmiljø med f.eks venv eller pdm.

venv:
```bash
python3 -m venv <navn-på-miljø>     # lag miljø
. <navn-på-miljø>/bin/activate      # aktiver miljø
pip install .                       # installer pakker og moduler
```

pdm:
```bash
pdm install
```


## Bruk

### Kjør alignment pipelinen
venv:
```bash
python3 -m align_documents align_all
```

pdm:
```bash
pdm run python -m align_documents align_all
```

#### Shorthand
`align_documents` uten å spesifiesere sub-kommando blir automatisk tolket som `align_documents align_all`:
```bash
# Disse to linjene har samme effekt
python -m align_documents
python -m align_documents align_all
```
NB: Hvis du vil se dokumentasjon av `align_all` sub-kommandoen ved bruk av `--help`-argumentet, må du fullstendig spesifisere `align_documents align_all --help`. Se [Mer informasjon](#mer-informasjon).

&nbsp;
### Mer informasjon

For mer informasjon om bruk og tilgjengelige sub-kommandoer:
```bash
python -m align_documents --help
```
Eller for en konkret sub-kommando, f.eks. for `align_all`:
```bash
python -m align_documents align_all --help
```


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