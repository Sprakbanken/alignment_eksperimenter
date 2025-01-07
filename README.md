

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

# Nynorsk-bokmål alignment

## 2021-data
Se mappa `eksperimenter_2021_data` 

## 2023-data
TBA

# Nynorsk-engelsk alignment