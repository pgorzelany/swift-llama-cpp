# Czy naprawy zachowały poprawną generację?

**Mechanizm generacji i składania tekstu jest zgodny z natywnym llama.cpp
w sprawdzonych przypadkach. Nie każda odpowiedź modelu jest jednak dobra.**

01.10.2026 porównaliśmy 72 pełne odpowiedzi: dziewięć scenariuszy, dwa modele,
dwa samplery, stary i poprawiony wrapper. Następnie odtworzyliśmy wszystkie
36 odpowiedzi poprawionego wrappera przez niezależną pętlę C API.

## Wynik kontroli technicznej

| Kontrola | Stary wrapper | Poprawiony wrapper |
|---|---:|---:|
| Pełne odpowiedzi zakończone EOS | 36/36 | 36/36 |
| Stream zachowuje dokładny tekst reprezentowany przez sampled C tokens | 33/36 | **36/36** |
| Identyczne tokeny z niezależną generacją C | nie sprawdzano | **36/36** |

Żadna odpowiedź nie została ucięta limitem 256 tokens. Najdłuższa miała 83 tokens.
Nowe odpowiedzi nie zawierają replacement character ani surowych znaczników `<|`.

Stary wrapper w trzech próbach tracił rzeczywiście wygenerowane znaki:

- LFM, oba samplery, echo Unicode: C wygenerowało emoji 🌍, którego nie było
  w zwróconym streamie. Nowy wrapper je zachowuje.
- Llama, sampler produkcyjny, echo Unicode: C wygenerowało `Žāļ oldē jaiņa`,
  a stream zwrócił `Žā oldē jaia`. Nie jest to poprawne wykonanie instrukcji,
  ale niezależnie od jakości modelu wrapper dodatkowo gubił jego litery.

W nowej wersji wszystkie tokeny odpowiadają natywnej generacji C, a stream nie
zmienia reprezentowanego przez nie tekstu. To mocniejsza kontrola niż samo
stwierdzenie, że odpowiedź nie jest pusta albo zawiera oczekiwane słowo.

## Co mówią rzeczywiste odpowiedzi

| Zadanie | Obserwacja |
|---|---|
| Dokładne `READY` | Oba modele i oba samplery odpowiadają `READY` przed i po |
| 17 + 25 | Wszystkie odpowiedzi: `42` |
| Pamięć rozmowy | Wszystkie odpowiedzi zachowują hasło `cobalt-lantern` |
| Streszczenie podróży | LFM zachowuje podstawowe fakty; Llama po zmianie przy temp 0.5 dodaje nieuzasadnione sformułowanie o „intended departure time” |
| Krótkie opowiadanie | Modele tworzą płynny tekst o kocie na Marsie; fikcyjnej fabuły nie traktujemy jako testu faktów astronomicznych |
| Język polski | LFM odpowiada sensownie o chłodzeniu/przechowywaniu żywności; Llama 1B daje błędne lub bezsensowne odpowiedzi w obu wersjach |
| Tłumaczenie | LFM greedy po zmianie: `Dzień dobry, dziękuję za Twoją pomoc.`; sampler produkcyjny ma błąd fleksji. Llama miewa błędne tłumaczenia, w tym gorszy przykład po zmianie |
| Powtórzenie Unicode | LFM zachowuje polskie znaki i emoji po naprawie, ale sam model zmienia końcowy tekst hindi. Llama po zmianie odmawia wykonania nieszkodliwego polecenia z błędnym uzasadnieniem |
| JSON bez gramatyki | LFM zwraca wymagany obiekt; Llama nadal dodaje Markdown mimo zakazu. To ograniczenie instruction following w tej próbce |

Nie twierdzimy zatem, że wszystkie odpowiedzi są lepsze po poprawkach.
Naprawy BOS oraz kolejności i historii penalties zmieniają rozkład samplowania,
więc mogą zmienić konkretną odpowiedź przy tym samym seedzie. W niektórych
przykładach treść po zmianie wypada gorzej. Pełny zbiór odpowiedzi zachowujemy,
łącznie z błędami i odmowami — bez wybierania wyłącznie udanych przykładów.

Ważne rozróżnienie: niezależna generacja C z poprawioną konfiguracją odtwarza
również błędne odpowiedzi. Nie wynikają one z utraty bytes, przestawienia pozycji
decode ani rozbieżności samplera wrappera względem tej konfiguracji C.
Pozostaje ocena doboru modelu, parametrów i jakości odpowiedzi na szerszym corpusie.

## Metoda

- Mac M1 Max, Metal, Release, batch/microbatch 1024, context 4096, jeden host thread.
- Te same GGUF Llama 3.2 1B Q4_K_M i LFM 2.5 1.2B QAD Q4_0, llama.cpp b10964.
- Przed: `b7f9e68250aff79d11232e8905c7f2db41a4bc18`.
  Po: `bcb9e0518a9f62031d7093d8ec0330aa2b8bc7fe`.
- Osobne programy ze snapshotów niezmienionych źródeł obu commitów.
- Temperature 0 oraz 0.5; seed 42; top-p 0.95, bez top-k;
  domyślne penalties lastN 64, repeat 1.1, frequency/presence 0.
  **Greedy tutaj zachowuje penalties**, inaczej niż greedy w benchmarku performance.
- Zwykłe `initializeCompletion` i `generateNextToken`, reset i nowy sampler przed
  każdym zadaniem. Nowa wersja używa również normalnego `finishDecoding`.
- Native text: bezpośredni `llama_detokenize` na tokenach wygenerowanych przez
  każdy wrapper, bez używania jego metody konwersji tekstu.
- Niezależna generacja nowej wersji: osobny C context, własny `llama_batch`,
  własny łańcuch C penalties→greedy lub penalties→top-p→temp→dist,
  własna pętla sample/decode oraz synchronizacja. Współdzieli wyłącznie model C.
  Penalties przyjmują historię promptu zgodnie z poprawionym kontraktem.
- Reference otrzymuje tokeny promptu przygotowane przez wrapper. Kontrola izoluje
  generację i bytes, **nie jest niezależnym testem całego renderowania promptu**.
  Tokenizację/BOS/chat formatting sprawdzają wcześniejsze testy regresji.

## Testy ścieżek wyższego poziomu

Ponownie uruchomiono na Macu w Release:

```sh
swift test -c release --no-parallel \
  --filter 'LlamaBehaviorTests|LlamaGrammarRegressionTests|multilingualGeneration|testTypedStreamingPerson|testShortStorySemanticBaseline|realGGUF'
```

**12 testów / 6 suites passed**, 10.287 s. Obejmują m.in. rzeczywistą odpowiedź
przez publiczny executor, pamięć rozmowy, generację po anulowaniu, opowiadanie,
JSON z gramatyką i typed decoding oraz dokładne wielojęzyczne UTF-8.
To testy znaczenia i kontraktów na małych przypadkach; nie są pełnym benchmarkiem
jakości językowej ani oceną wszystkich aplikacyjnych przepływów.

## Odtworzenie i dowody

```sh
python3 Verification/run-generation-quality.py \
  --llama-model /absolute/path/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --lfm-model /absolute/path/LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf \
  --output /tmp/wrapper-generation-quality
python3 Verification/summarize-generation-quality.py /tmp/wrapper-generation-quality
```

- [Wszystkie odpowiedzi przed/po](generation-quality/answers.md).
- [JSONL: tekst, tokeny, reference i EOS](generation-quality/results.jsonl).
- [Automatyczne kontrole](generation-quality/checks.json).
- [Metadane i hashe](generation-quality/metadata.json).
- [Wybrane wyniki testów](generation-quality/test-results.txt).

To mała kontrola jakości i zgodności, z jednym seedem i dwoma małymi modelami.
Nie dowodzi braku każdej możliwej regresji jakości, poprawności każdej odpowiedzi
ani równoważnego zachowania wszystkich innych GGUF i ustawień.
