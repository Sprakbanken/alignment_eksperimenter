# Målfrid parallell

Ressursen inneholder paralleldata for engelsk-bokmål, engelsk-nynorsk og bokmål-nynorsk fra 238 statelige domener.

Datagrunnlaget er hentet fra det såkalte Målfrid-prosjektet, der Nasjonalbiblioteket i samarbeid med Språkrådet høster statlige nettsider i forbindelse med språktilsyn.

Vi kombinerte følgende datasett:
- Målfrid 2021 [ressurskatalog](https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-69/)
- Målfrid 2022 [ressurskatalog](https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-97/)
- Målfrid 2023 [ressurskatalog](https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-98/)
- Målfrid 2024 [ressurskatalog](https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-99/)
- Målfrid 2025 [ressurskatalog](https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-102/)

## Dataformat og bruk

Dataen er på jsonlformat med en jsonlfil per språkpar per datasplitt. Hver rad inneholder følgende unike kolonner: 

- url_{språk}: url-en dokumentet ble høstet fra
- domain_{språk}: navnet på nettsiden dokumentet ble høstet fra.
- mimetype_{språk}: type dokument teksten ble høtet fra (pdf, docx, html)
- fulltext_{språk}: tekstinnholdet i dokumentet (liste av strenger)
- date_{lang}: tidspunkt for høsting

Du kan laste inn datasettet slik: 

```
from datasets import load_dataset

ds = load_dataset("maalfrid_parallel/nob_eng")

print(ds)
```

Output vil da se slik ut: 


```
DatasetDict({
    train: Dataset({
        features: ['doc_hash_nob', 'lang_nob', 'url_nob', 'domain_nob', 'date_nob', 'mimetype_nob', 'fulltext_nob', 'doc_hash_eng', 'lang_eng', 'url_eng', 'domain_eng', 'date_eng', 'mimetype_eng', 'fulltext_eng'],
        num_rows: 157044
    })
    validation: Dataset({
        features: ['doc_hash_nob', 'lang_nob', 'url_nob', 'domain_nob', 'date_nob', 'mimetype_nob', 'fulltext_nob', 'doc_hash_eng', 'lang_eng', 'url_eng', 'domain_eng', 'date_eng', 'mimetype_eng', 'fulltext_eng'],
        num_rows: 19720
    })
    test: Dataset({
        features: ['doc_hash_nob', 'lang_nob', 'url_nob', 'domain_nob', 'date_nob', 'mimetype_nob', 'fulltext_nob', 'doc_hash_eng', 'lang_eng', 'url_eng', 'domain_eng', 'date_eng', 'mimetype_eng', 'fulltext_eng'],
        num_rows: 19720
    })
})
```

## Metode

Dokumentparene ble sammenstilt per nettside med NbAiLab/nb-sbert-v2-base og sentence-transformers-biblioteket.
For engelsk-norsk parallelldata er minste cosinus-likhetsterskel 0,80, og for norsk parallelldata er den 0,95.
Vi brukte et manuelt annotert parallleldatasett for å finne de optimale terskelverdiene. Se kildekode https://github.com/Sprakbanken/alignment_eksperimenter. Det manuelt annoterte datasettet ligger på https://github.com/Sprakbanken/alignment_eksperimenter/blob/main/manual_annotation/annotated_data/gold_data.csv. 

## Statistikk

Antall dokumentpar per språkpar per split:

- engelsk-bokmål:
    - trening: 157 044
    - validering: 19 720
    - test: 19 720
- engelsk-nynorsk:
    - trening: 30 901 
    - validering: 3796
    - test: 3795
- bokmål-nynorsk: 
    - trening: 31 434
    - validering: 3928
    - test: 3929



