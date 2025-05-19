

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
python3 -m align_documents align_all
```

eller med pdm:
```bash
pdm run python -m align_documents align_all
```

`align_documents` uten å spesifiesere sub-kommando blir automatisk tolket som `align_documents align_all`:
```bash
python -m align_documents [optional_argument, ...] # => python -m align_documents align_all [optional_argument, ...]
```


For mer informasjon:
```bash
pdm run python -m align_documents -h
```
eller for en konkret sub-kommando:
```bash
pdm run python -m align_documents align_all -h
```

# Nynorsk-bokmål alignment

## 2021-data
Se mappa `eksperimenter_2021_data` 

## 2023-data
TBA

# Nynorsk-engelsk alignment
