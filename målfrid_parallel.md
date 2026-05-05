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



---

# Målfrid Parallel

The resource contains parallel data for English–Bokmål, English–Nynorsk, and Bokmål–Nynorsk from 238 government domains.

The data originates from the so-called Målfrid project, in which the National Library of Norway, in collaboration with the Language Council of Norway, harvests government websites as part of language supervision.

We combined the following datasets:
- Målfrid 2021 [resource catalogue](https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-69/)
- Målfrid 2022 [resource catalogue](https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-97/)
- Målfrid 2023 [resource catalogue](https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-98/)
- Målfrid 2024 [resource catalogue](https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-99/)
- Målfrid 2025 [resource catalogue](https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-102/)

## Data format and usage

The data is in JSONL format with one JSONL file per language pair per data split. Each row contains the following unique columns:

- url_{language}: the URL the document was harvested from
- domain_{language}: the name of the website the document was harvested from
- mimetype_{language}: the type of document the text was harvested from (pdf, docx, html)
- fulltext_{language}: the text content of the document (list of strings)
- date_{lang}: the time of harvesting

You can load the dataset as follows:

```
from datasets import load_dataset

ds = load_dataset("maalfrid_parallel/nob_eng")

print(ds)
```

The output will look like this:


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

## Method

Document pairs were aligned per website using NbAiLab/nb-sbert-v2-base and the sentence-transformers library.
For English–Norwegian parallel data the minimum cosine similarity threshold is 0.80, and for Norwegian parallel data it is 0.95.
We used a manually annotated parallel dataset to find the optimal threshold values. See source code at https://github.com/Sprakbanken/alignment_eksperimenter. The manually annotated dataset is available at https://github.com/Sprakbanken/alignment_eksperimenter/blob/main/manual_annotation/annotated_data/gold_data.csv.

## Statistics

Number of document pairs per language pair per split:

- English–Bokmål:
    - train: 157 044
    - validation: 19 720
    - test: 19 720
- English–Nynorsk:
    - train: 30 901
    - validation: 3796
    - test: 3795
- Bokmål–Nynorsk:
    - train: 31 434
    - validation: 3928
    - test: 3929


