# GPU: wrapper przed i po naprawach

Pomiar z 01.10.2026 na Apple M1 Max 64 GB, macOS 27.0, Swift 6.4.
Porównujemy cały przypięty kod wrappera, a nie wcześniejszy izolowany koszt samplera.

**Greedy zyskało, ale domyślny sampler produkcyjny ma regresję w części prób.**
Nie ma podstaw do ogólnej deklaracji „aplikacja na GPU jest szybsza po naprawach”.
Nie zmieniamy domyślnych parametrów ani nie wyłączamy penalties na podstawie tego pomiaru.

## Wersje i ustawienia

- Przed: `b7f9e68250aff79d11232e8905c7f2db41a4bc18`, produkcyjny `main`
  wrappera w momencie pomiaru, przed audytem i naprawami.
- Po: `bcb9e0518a9f62031d7093d8ec0330aa2b8bc7fe`, naprawy na branchu
  `fix/llama-wrapper-correctness-performance`.
- Ten sam lokalny llama.cpp b10964 XCFramework, revision
  `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`, w obu programach.
- GPU / pełny Metal offload, jeden host thread w obu fazach, context 4096,
  batch i microbatch 1024. To batch domyślny macOS aplikacji,
  odczytany z `Enclave-MacOS/Screens/Chat/ChatScreenModel.swift`.
  iOS aplikacja ma batch 256; tych wyników nie traktujemy jako pomiaru iPhone'a.
- Llama 3.2 1B Q4_K_M i produkcyjny LFM 2.5 1.2B QAD Q4_0,
  z identycznymi GGUF w obu wariantach. Hashe są zapisane w metadanych.

## Metoda

Skrypt eksportuje **niezmienione** `Sources/SwiftLlama` dwóch konkretnych commitów
do osobnych katalogów tymczasowych. Ten sam harness kompiluje się wraz z każdym
z nich jako program Release. Nie przełączamy głównego checkoutu ani nie przenosimy
poprawek do wersji bazowej. Harness jest poza targetem produkcyjnej biblioteki.
Oba programy linkują ten sam framework. Kompilacje kończą się przed pomiarami.
Sprawdziliśmy bajtową zgodność wszystkich 24/25 plików źródłowych z commitami
oraz identyczny SHA-256 binarnego llama frameworka załadowanego przez oba programy.

Trzy warianty:

1. `greedy-controlled`: temperature 0, bez penalties, identyczne tokeny promptu
   w obu wersjach. Helper odtwarza tokeny przez normalny context/batch wrappera,
   a generacja wywołuje niezmienioną metodę `Llama.generateNextToken` danej wersji.
   To kontrola wpływu napraw na sample + decode + konwersję UTF-8.
   Prefill tego wariantu nie obejmuje formatowania i tokenizacji.
2. `greedy-native`: temperature 0, bez penalties, normalne
   `initializeCompletion(messages:)` i `generateNextToken()` obu wersji.
3. `production`: normalne metody obu wersji, temperature 0.5, top-p 0.95,
   top-k wyłączone, domyślne repetition penalties: lastN 64, repeat 1.1,
   frequency/presence 0. Seed 42 ustalony dla odtwarzalności. Temperatura 0.5
   odpowiada domyślnej wartości ustawień aplikacji.

Ten sam tekst użytkownika i te same dokładne fixture tokens w każdej parze.
Prompt obejmuje padding oraz prośbę o długie opowiadanie science fiction.
Docelowe długości promptu: 64, 512, 2048; output zawsze **32 tokeny**.
Pomiar przerywa się przy wcześniejszym EOS; krótsze odpowiedzi nie są mieszane
z dłuższymi w porównaniu tok/s. Próbne workloady z liczeniem do 10000 kończyły
niektóre odpowiedzi przed limitem (107/128, 12/32 lub 16/32), więc nie weszły
do końcowego zbioru danych. Nie wybieraliśmy pomiarów według szybkości.

W normalnej ścieżce brakujący BOS starego wrappera jest oczekiwaną różnicą
poprawności: helper akceptuje wyłącznie identyczny prompt albo pominięcie jego
jednego pierwszego tokenu. Pozostałe tokeny muszą zgadzać się dokładnie.
JSONL zapisuje pominięty token i rzeczywistą długość wejścia. Porównanie kontrolne
wymaga pełnej zgodności wejścia oraz wszystkich generowanych tokenów greedy.
Wyjście normalnej ścieżki może się zmienić po naprawie BOS, kolejności penalties
i zasilania ich historią promptu; nie jest to test niezmienności treści odpowiedzi.

Cztery bloki, w każdym rozgrzewka i cztery właściwe powtórzenia każdego workloadu.
Łącznie **16 właściwych pomiarów na wersję/workload**. Kolejność wersji jest
przeplatana before/after, after/before, before/after, after/before. Kolejność modeli,
samplerów i długości również odwraca się między blokami. GPU pracuje sekwencyjnie.
Nie uruchamiamy równolegle innych testów ani kompilacji. Stan termiczny jest
rejestrowany; podsumowanie wymaga `nominal` we wszystkich próbach.

Przed próbą czyścimy KV i odtwarzamy sampler w obu wersjach. Dzięki temu brak
resetu historii samplera w starym `resetCompletion` nie zanieczyszcza powtórzeń.
Reset i budowa samplera są poza timerem. Prompt cache nie przyspiesza prefillu.

TG = output tokens / pętla `generateNextToken` (sample, decode, piece/UTF-8).
PP = rzeczywiste prompt tokens / prefill. TTFT = prefill + pierwsze sample/decode.
TTFT wyklucza wczytanie modelu, sprawdzenie dostępności GPU i inicjalizację Metal.
Load zapisany w JSONL jest pomocniczym pomiarem po sprawdzeniu GPU, z rozgrzanym
systemowym page cache. Nie jest pomiarem całego zimnego startu aplikacji.
Tabela podaje mediany i zmianę TG osobno dla każdego bloku, bez usuwania odstających prób.

## Odtworzenie

Z katalogu wrappera, z lokalnym b10964 XCFramework:

```sh
python3 Verification/run-gpu-comparison.py \
  --llama-model /absolute/path/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --lfm-model /absolute/path/LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf \
  --output /tmp/gpu-wrapper-ab
python3 Verification/summarize-gpu-comparison.py /tmp/gpu-wrapper-ab
```

Output musi wskazywać nowy katalog. Skrypt zachowuje snapshoty i logi kompilacji
w katalogu tymczasowym, którego ścieżka znajduje się w `metadata.json`.
Nie pobiera modeli ani zależności z internetu.

## Wyniki i decyzja

Wszystkie 720 prób zakończyły się z dokładnie 32 output tokens: 144 warmupy
oraz 576 właściwych pomiarów. Thermal state nominal w każdej próbie. Kontrolowane
wejście i wyjście greedy zgadzają się w **96/96 parach**. Wyjście kontrolne nowej
wersji zgadza się również z jej zwykłą ścieżką greedy w 96/96 porównaniach.
W normalnej ścieżce stara wersja pomija BOS w **obu** modelach: 128000 dla Llama,
1 dla LFM. Długości przed/po wynoszą 63/64, 511/512, 2047/2048.

Poniżej docelowy prompt 512 i 32 output tokens. TG w tok/s, TTFT w milisekundach.

| Model | Sampler | TG przed | TG po | Zmiana TG | TTFT przed | TTFT po |
|---|---|---:|---:|---:|---:|---:|
| lfm-1.2b | greedy-controlled | 214.0 | 243.9 | +14.0% | 188.9 | 192.3 |
| lfm-1.2b | greedy-native | 210.1 | 242.1 | +15.2% | 196.7 | 203.6 |
| lfm-1.2b | production | 229.1 | 222.9 | -2.7% | 199.8 | 200.3 |
| llama-1b | greedy-controlled | 173.2 | 185.6 | +7.2% | 207.0 | 223.1 |
| llama-1b | greedy-native | 172.3 | 184.4 | +7.1% | 222.8 | 226.9 |
| llama-1b | production | 171.5 | 165.6 | -3.5% | 224.6 | 222.5 |

- **Greedy:** kontrola identycznych tokens potwierdza zysk generacji około
  7–14% w zależności od modelu/długości. Dla 512 tokens: Llama +7.2%, LFM +14.0%.
  Zwykła ścieżka także wygrywa: +7.1% oraz +15.2% dla 512 tokens.
  Greedy w tym pomiarze ma penalties wyłączone; nie jest to konfiguracja
  domyślnego czatu ani pełny pomiar greedy z domyślnymi penalties.
- **Default produkcji:** dla 512 tokens Llama -3.5%, LFM -2.7%.
  Spadek pojawia się w każdym z czterech bloków: Llama -3.1…-11.2%,
  LFM -1.7…-2.9%. To sygnał regresji w badanym workloadzie, który trzeba
  uwzględnić przed deklarowaniem poprawy produkcyjnego performance.
- **Inne długości defaultu:** Llama -2.1% dla 64 i -6.7% dla 2048;
  LFM -0.7% dla 64, ale **+9.3% dla 2048**. Nie ma jednego współczynnika
  zmiany dla wszystkich promptów. Przy krótkiej generacji część prób ma
  szerszy rozrzut; wszystkie pozostają w zbiorze i tabeli bloków.
- **Prefill i TTFT:** brak dużego, wspólnego przyspieszenia. W normalnej
  ścieżce PP zmienia się od -4.3% do +6.0%; TTFT zmienia się o pojedyncze
  milisekundy, czasem zyskując mimo wolniejszej dalszej generacji.
  Kontrola Llama/512 ma **PP -7.8%** i TTFT 207.0→223.1 ms.
  Jest to obserwacja wymagająca wyjaśnienia, nie dowód kosztu tokenizacji:
  kontrola omija tokenizację, a konfiguracja i decode C pozostają takie same.
  Małe różnice i odchylenia trzeba odróżnić od stabilnego zysku greedy.

Najbardziej wiarygodny trop dla wolniejszego samplera produkcyjnego to poprawiona
kolejność penalties. Stary łańcuch uruchamiał top-p przed penalties; nowy penalties
przed top-p i dodatkowo zasila je historią promptu. W przypiętym C++
`src/llama-sampler.cpp:2950` funkcja `llama_sampler_penalties_apply` iteruje po
całym `cur_p` i wykonuje lookup historii. Przed top-p oznacza to pełny słownik,
po top-p tylko przycięty zbiór. To **hipoteza przyczyny kosztu** na podstawie kodu,
nie izolowany profil CPU przypisujący cały zmierzony spadek tej jednej operacji.
W normalnej ścieżce różni się także BOS i wynikowe logits/output, więc nie wolno
traktować tego pomiaru jako identycznej jakościowo odpowiedzi z innym czasem.

Następny sensowny cel to profilowanie i przyspieszenie poprawnego penalties-before-filter
oraz ustalenie źródła odchylenia prefillu. Cofnięcie kolejności albo wyłączenie
penalties zmieniłoby semantykę i nie jest naprawą regresji. Ten commit zawiera
pomiary i narzędzia, **bez kolejnych zmian runtime lub parametrów produkcji**.

Pełne mediany, wyniki wszystkich czterech bloków i zgodność outputów:
[summary.md](gpu-before-after/summary.md). Wszystkie próby, w tym warmupy:
[results.jsonl](gpu-before-after/results.jsonl). Środowisko, SHA i hashe modeli:
[metadata.json](gpu-before-after/metadata.json). Fizyczne przypisanie Metal:
[backend assignments](gpu-before-after/backend-assignments.txt).

## Granice

Pomiar dotyczy tych dwóch modeli i tego M1 Max. Nie mierzy UI, pełnego streamingu
executora/serwisu, tool calling, typed JSON, LoRA, snapshot/restore ani cache reuse.
Nie jest pomiarem fizycznego iPhone'a, zużycia energii, długiej sesji termicznej
ani szczytowej pamięci. Kontrola nominalnego thermal state nie wyklucza zmiennego
taktowania i obciążenia przez system. Małych różnic kilku procent nie traktujemy
jako uniwersalnej gwarancji albo automatycznego progu CI.
