# Eksperimenter med text alignment for bokmål og nynorsk

## Dependencies
Se `requirements.txt` for python-pakker du trenger for å kjøre denne koden


# Lånekassen-eksperimenter

Lånekassen har mye parallelldata, så vi har utført flere eksperimenter på deres data.  
Det er 566 dokumenter, hvorav 256 er nynorske og 310 er på bokmål.  

## Dokument-alignment

### "Fasit" 
(se notebooken [lånekassen_fasit.ipynb](lånekassen_fasit.ipynb))  
Av de 566 dokumentene vi har scrapet fra lånekassen, er det 110 dokumenter hvor urlen er helt lik bortsett fra språkkoden ("nn-NO" og "nb-NO").  
Disse bruker vi som en "fasit" for parallelle dokumenter.  
MERK: Dette er bare 1/5 av alle dokumentene, og vi har ingen negative eksempler. Vi kan dermed ikke måle precision og recall, men dette subsettet av parallelldata kan gi en indikator på hvor bra eksperimentene går ved å se i hvor stor grad de treffer fasiten.

### Eksperimenter med sentence-transformers: NB sBERT og LaBSE
(se notebookene [lånekassen_sbert.ipynb](lånekassen_sbert.ipynb) og [lånekassen_labse.ipynb](lånekassen_labse.ipynb))

Vi fant parallelldokumenter ved å lage dokument-embeddings for alle dokumentene, og fant den likeste bokmålsembeddingen for hver nynorskembedding. Hvis bokmålsembeddingen var likere enn en viss terskel (i våre eksperimenter en cosinuslikhet over 0.95), kaller vi det en match, og et potensielt parallelldokument til det nynorske. 

Både NB sBERT og LaBSE har kortere makslengde for inputs enn de fleste dokumentene. NB sBERT sin makslengde er 75 tokens, mens LaBSE har 256. For å omgå den korte kontekstlengden delte vi dokumentene opp i biter kortere enn makslengden, og kjørte disse gjennom modellene, og aggregerte opp til en embedding med forskjellige aggregeringsstrategier.  

Eksperimentet som fikk flest matcher (altså fant flest potensielle parallelldokumenter på bokmål til nynorskdokumentene) var NB sBERT og mean pooling[^1]. Dette ga 236 par av dokumenter.  
Dette var også metoden som traff flest av dokumentene fra fasiten (traff 52 av 55).  

Nest best i treff på fasit deles mellom NB sBERT med max pooling[^2], LaBSE med mean pooling og LaBSE med naiv cut-off [^3]. Disse traff 48 av 55.  
De fant matcher for 221, 212 og 217 dokumenter, respektivt. Siden vi ikke har noen god måte å måle falske positiver (annet enn stikkprøvder), er det ikke så godt å si hvilke av disse metodene som egentlig er best.  


[^1]: å la hvert dokuments vektorrepresentasjon være gjennomsnittsvektoren av embeddingene av bitene i dokumentet  
[^2]: å la hvert dokuments vektorrepresentasjon være maksvektoren (altså maksverdien i hver dimensjon) av embeddingene av bitene i dokumentet  
[^3]: å godta at dokumentet blir kuttet på makslengden (kanskje starten av dokumentet er representativt nok?)  

## Avsnitts-alignment

Dokumentene er delt opp i lister av avsnitt/setninger, som er tekstinnholdet til p-tags på nettisdene.   
Hvor store disse bitene er avhenger fra dokument til dokument, fra nettside til nettside.     
I data fra lånekassen ser det ut som mange av disse avsnittene egentlig er setninger (og i noen tilfeller, enkeltord).  

Det er 4032 avsnitt på nynorsk, og 6452 på bokmål.

### Eksperimenter med sentence-transformers: NB sBERT og LaBSE
(se notebookene [lånekassen_sbert.ipynb](lånekassen_sbert.ipynb) og [lånekassen_labse.ipynb](lånekassen_labse.ipynb))

Ca 70% av avsnittene/setningene er innenfor NB sBERT sin makslengde.  
Ca 93% er innenfor LaBSE sin makslengde. 

Vi bruker tilnærming som ved dokumentalignmen for å finne parallellavsnitt, og de samme aggregeringsstrategiene for å ta høyde for de avsnittene som er for lange.

I tillegg sjekker vi bare enkel stringmatch, og finner at 341 av de nynorske avsnitt/setning/ordene har en nøyaktig match fra bokmålavsnitt/-setning/-ordene. 

Eksperimentet som fikk flest matcher var LaBSE embeddings og naiv cutoff på biter[^4], som ga 2389 paralellavsnitt på nynorsk og bokmål.  
For avsnitt/setningsalignment fikk alle LaBSE-eksperimentene flere matches enn NB sBERT-eksperimentet med flest matches. Med NB sBERT er det naiv cutoff som gir flest matches, med 2343 paralellavsnitt.   
Vi har ikke kvalitetssikret disse treffene, så det er ikke nødvendigvis slik at metoden som finner flest potensielle parallelltekster er den som er best.   


[^4]: vi deler dokumentet opp i biter som er kortere enn makslengden og sender lista med biter inn i modellen. Det som skjer er at modellen lager en embedding av de to første listene (hacky løsning, se notebooken [diverse/sbert_weirdness.ipynb](diverse/sbert_weirdness.ipynb) for detaljer)

