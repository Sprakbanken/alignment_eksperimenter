Forskjellige eksperimenter med text alignment med data fra Målfrid.

Formålet er å få mer nynorsk-bokmål og nynorsk-engelsk parallelldata av høy kvalitet.

Installering
------------

Du trenger python3.11 og pip.

Lag et virtuelt pythonmiljø med f.eks venv:  
```bash
python3 -m venv <navn-på-miljø>     # lag miljø
. <navn-på-miljø>/bin/activate      # aktiver miljø
pip install .                       # installer pakker og moduler
```

Med pdm:
```bash
pdm install
```

Bruk
----

Da kan du kjøre alignment-pipelinen slik:
```bash
python3 -m align_documents align_all
```

eller med pdm:
```bash
pdm run python -m align_documents align_all
```

`align_documents` uten å spesifiesere sub-kommando blir automatisk tolket som `align_documents align_all`:
```bash
python -m align_documents # => python -m align_documents align_all
```

Mer informasjon
---------------

For mer informasjon om bruk og tilgjengelige sub-kommandoer:
```bash
python -m align_documents --help
```
Eller for en konkret sub-kommando, f.eks. for `align_all`:
```bash
python -m align_documents align_all --help
```

Data
----

### Nynorsk-bokmål alignment

- 2021-data
    - Se mappa `eksperimenter_2021_data` 

- 2023-data
    - TBA

### Nynorsk-engelsk alignment
------------------------------
- TBA
