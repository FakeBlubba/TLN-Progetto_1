# Laboratorio 1

## Calcolare la similarità la similarità nelle definizioni dei 4 concetti del dataset.

Si è scelto di intendere come **similarità** la sovrapposizione lessicale fra le varie definizioni ossia il numero di parole usate nelle definizioni da tutti diviso la lunghezza media delle definizioni.

```python
from nltk import word_tokenize, SnowballStemmer
from nltk.corpus import stopwords
import string
import pandas as pd
import time
```

Sono state importate le seguenti libreria per svolgere il pre-processing. L'algoritmo di stemming che si è utilizzato è lo Snowball Stemmer.

```python
stop_words = set(stopwords.words('english'))
stop_words.update(set(string.punctuation))
```

In `stop_words` sarà contenuto l'insieme di tutte le parole da eliminare nelle definizioni, compresa la punteggiatura, così da ridurre il rumore.

```python
corpus = {}          # Dictionary che conterrà il set delle parole pre-processate per ogni concetto
intersection = {}    # Dictionary che conterrà la lista dei conteggi di ogni parola contenuta in corpus
media = {}           # Dictionary che conterrà la media aritmetica della lunghezza di ogni definizione per ogni concetto
int_count = {}       # Dictionary che conterrà l'associazione le parole dentro corpus e il suo conteggio divisi per ogni concetto
output = {}          # Dictionary contenente le chiavi con i nomi dei concetti e come valori la similarità delle definizioni
```

Sopra a questo testo sono espresse le principali variabili e strutture dati utilizzate.

```python
df = pd.read_csv('dataset/defs.csv', header=0)
df = df.dropna()
df.drop(['Partecipante'],axis=1,inplace=True)
data = df.to_dict()
```

Utilizzando la libreria importata `pandas` sono state salvate nel dictionary `data` i dati contenuti in `defs.csv`, ossia il foglio di partenza contente tutte le definizioni. Tramite `df.dropna()` sono stati eliminati i campi vuoti, ed è stata elminata anche la colonna `Partecipante`. Ne risulterà un dictionary con la seguente struttura: `data = {'concetto' : {0: prima, definizione}}`.

```python
for names in data.keys():
    for index in data[names]:
        data[names][index] = word_tokenize(data[names][index].lower())
        data[names][index] = list(set([snow_stemmer.stem(word) for word in data[names][index] if word not in stop_words]) - stop_words) 
```

Si effettuerò la tokenizzazione delle definizioni, verrà effettuato lo stemming delle definizioni che non sono presenti in `stop_words`. Oltre questo verrà aggiornato il dictionary`data` con le sole parole non appartenenti a `stop_words`, grazie alla differenza tra i due insiemi.

```python
for name in data.keys():
    # Lista contenente tutte le parole usate all'interno delle definizioni
    corpus[name] = list(set(word for index in data[name] for word in data[name][index]))
    total = 0
    word_count = []
    
    # Conteggio delle parole per trovare quella che si ripete di più
    intersection_count = [0] * len(corpus[name])
    
    for index in data[name]:
        word_count.append(len(data[name][index]))
        for word in data[name][index]:
            id_word = corpus[name].index(word)
            intersection_count[id_word] += 1
        total += 1
    intersection[name] = intersection_count
    count = sum(word_count)
    media[name] = round(count / total,2)    
```

Con questo loop si andranno a:

- Aggiornare il dictionary `corpus` con la lista delle parole di ogni concetto (senza ripetizioni perchè precedentemente convertite in un `set`;
- La lista `intersection_count` conterrà il conteggio di ogni parola contenuta in `corpus`.
- Aggiornare il dictionary `intersection` con i conteggi contenuti in `intersection_count` (il cui contenuto viene refreshato a ogni iterazione sui concetti).
- `word_count` servirà per calcolare la media, verranno prima sommati tra loro i singoli conteggi delle parole per ottenere il totale delle parole in tutte le definizioni.
- `media` conterrà, per ogni concetto, la divisione tra la somma di tutte le parole `count` contenute nelle definizioni e il `total`, il conteggio di tutte le definizioni. Quest'operazione verrà effettuata per ogni concetto in un loop.

```python
for name in corpus.keys():
    int_count[name] = [[corpus[name][i], intersection[name][i]] for i in range(0, len(intersection[name]))]
    int_count[name] = sorted(int_count[name], key=lambda x: x[1], reverse=True)   
    out_list = []
    for i in range(0, len(int_count)):
        out_list.append(int_count[name][i][1])
    output[name] = round(sum(out_list) / media[name],2)
```

`int_count` conterrà per ogni concetto, una lista di liste composte dalle coppie "parola, conteggio", come espresso di seguito: `[mela, 5]`. Verrà poi effettuato un'ordinamento delle liste nel dictionary di `int_count` disponendo gli elementi con il conteggio più alto in ordine decrescente.

L'`output` conterrà la similarità delle varie definizioni per ogni concetto. Esso verrà ottenuto tramite l'arrotondamento della divisione tra la somma dei conteggi delle parole in `out_list` e la media delle parole per ogni definizione.