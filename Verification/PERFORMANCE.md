# CPU i Metal na Macu

Bezpośrednie porównanie całego wrappera przed/po, wykonane 01.10.2026:
[GPU-BEFORE-AFTER.md](GPU-BEFORE-AFTER.md). Greedy przyspieszyło; sampler
produkcyjny ma regresję w części prób. Poniższe CPU/Metal wyniki nie są
porównaniem starego i nowego wrappera.

Host: Apple M1 Max, 10 CPU cores (8P + 2E), 64 GB, macOS 27.0, Swift 6.4.
Runtime: przypięty b10964 Apple XCFramework. Pomiar obejmuje rzeczywisty wrapper
Swift: jego formatowanie i tokenizację promptu, batching, sampler i decode.

## Metoda

Build Release. GPU oznacza pełny domyślny Metal offload; CPU wyłącza layer,
KQV oraz operation offload. Logi potwierdzają przypisanie buforów.
Context 4096, identyczne dokładne prompty 64/512/2048 tokens, 32 output tokens.
Temperature=0, seed=42, penalties wyłączone. Każdy wariant ma jeden pomiar
rozgrzewki (repetition=0, pominięty w medianach) i trzy właściwe powtórzenia.
Model/context jest ponownie używany, ale pamięć i sampler są resetowane między
próbami. Prompt cache reuse nie skraca kolejnych prefillów.

PP = pełny prompt / czas initializeCompletion.
TG = 32 tokeny / czas pętli sample+decode.
TTFT = czas initializeCompletion + pierwsze sample+decode.
Wczytanie to pojedynczy osobny model/context initialization per profile,
po wcześniejszych testach na tym hoście. Nie jest to zimny dysk/page cache
ani powtarzany cold-start benchmark. TTFT nie obejmuje wczytania.

CPU 1/4/8 porównano przy 64 tokens; dalsze prompty na CPU używają 8 wątków.
GPU porównano z 1 i 4 host threads, batch/microbatch 256/256,
1024/1024 oraz 1024/256. Context, prompt i sampler w danej długości są takie same.
Warianty uruchamiane kolejno, bez innych model tests/buildów w trakcie pomiaru.
Fixed order i trzy powtórzenia są ograniczeniem — to lokalny benchmark,
nie gwarancja dla wszystkich urządzeń i modeli ani próg CI.
Output służy identycznemu workloadowi, nie ocenie jakości.

Początkowy próbny prompt kończył odpowiedź na 22 tokens, więc jego pomiar przerwano
i wyłączono z wyników. Zweryfikowany workload wymaga dokładnej liczby prompt
i output tokens; test sprawdza oba warunki. Preflight CPU/Metal również
nie wchodzi w poniższe mediany.

## Odtworzenie

Z katalogu pakietu:

    LLAMA_BENCHMARK=1 LLAMA_BENCH_OUTPUT=/tmp/llama-benchmark.jsonl \
      swift test -c release --no-parallel --filter LlamaPerformanceTests

Opcjonalnie:
- LLAMA_BENCH_MODEL=/absolute/path/to/model.gguf
- LLAMA_BENCH_PROFILES=cpu-8,metal-1
- LLAMA_BENCH_MAX_PROMPT=512

JSONL zawiera warmup i wszystkie właściwe próby. Skrypt generuje CSV oraz tabelę:

    python3 Verification/summarize-performance.py Verification/llama-1b.jsonl

Pomiar samego host samplera na ustalonych logits:

    LLAMA_SAMPLER_BENCHMARK=1 \
      swift test -c release --no-parallel --filter LlamaSamplerPerformanceTests

W nim stary łańcuch top-p → temp(0) → dist porównywany jest z greedy,
przy tych samych logits i wyłączonych penalties. To izolowany koszt wyboru tokenu,
nie end-to-end generation tok/s. Kolejność jest naprzemienna; 200 próbek na próbę,
jedna rozgrzewka i siedem właściwych powtórzeń. Wybrane tokeny muszą być identyczne.

## Wyniki


### Llama 3.2 1B Q4_K_M
| Profile | Prompt | Prefill tok/s | Generation tok/s | TTFT s | Load s | TG min–max |
|---|---:|---:|---:|---:|---:|---:|
| cpu-1 | 64 | 3.0 | 2.3 | 22.042 | 1.041 | 2.3–2.3 |
| cpu-4 | 64 | 11.2 | 8.6 | 5.827 | 1.001 | 8.6–8.6 |
| cpu-8 | 64 | 21.7 | 16.8 | 3.011 | 0.997 | 16.1–16.9 |
| cpu-8 | 512 | 21.8 | 15.9 | 23.556 | 0.997 | 15.8–16.0 |
| cpu-8 | 2048 | 21.7 | 15.3 | 94.570 | 0.997 | 14.5–16.2 |
| metal-1 | 64 | 2061.2 | 195.3 | 0.036 | 0.228 | 195.2–195.8 |
| metal-1 | 512 | 2550.9 | 192.0 | 0.206 | 0.228 | 191.8–192.9 |
| metal-1 | 2048 | 2370.8 | 183.7 | 0.870 | 0.228 | 183.5–185.9 |
| metal-4 | 64 | 2080.2 | 197.8 | 0.036 | 0.221 | 197.7–197.9 |
| metal-4 | 512 | 2553.4 | 194.0 | 0.206 | 0.221 | 193.7–199.0 |
| metal-b1024 | 512 | 2659.3 | 194.4 | 0.198 | 0.239 | 194.3–194.4 |
| metal-b1024 | 2048 | 2317.3 | 185.6 | 0.890 | 0.239 | 183.8–185.6 |
| metal-b1024-u256 | 512 | 2558.7 | 191.9 | 0.206 | 0.229 | 191.7–191.9 |
| metal-b1024-u256 | 2048 | 2191.3 | 183.1 | 0.941 | 0.229 | 177.2–183.5 |

### LFM 2.5 1.2B QAD Q4_0
| Profile | Prompt | Prefill tok/s | Generation tok/s | TTFT s | Load s | TG min–max |
|---|---:|---:|---:|---:|---:|---:|
| cpu-8 | 64 | 31.8 | 31.6 | 2.047 | 0.351 | 28.6–32.3 |
| cpu-8 | 512 | 31.5 | 28.6 | 16.303 | 0.351 | 26.6–31.8 |
| metal-1 | 64 | 2274.1 | 248.0 | 0.033 | 0.099 | 247.9–248.6 |
| metal-1 | 512 | 2767.5 | 245.5 | 0.190 | 0.099 | 245.1–245.9 |
| metal-4 | 64 | 2276.1 | 248.0 | 0.032 | 0.102 | 247.9–248.2 |
| metal-4 | 512 | 2768.9 | 248.7 | 0.189 | 0.102 | 245.2–250.2 |

### Greedy fast path

Stary sampler: 5.8636 ms / wybór tokenu; nowy: 0.0591 ms.
Około 99.3× mniej czasu w izolowanym host samplerze,
przy identycznych wybranych tokenach. Nie jest to 99×
szybsza pełna inferencja: decode pozostaje główną częścią generacji.
Dotyczy temperature=0; probabilistyczne generowanie nadal wymaga filtrów.

### Decyzje

- CPU 8 threads zamiast 1 daje przy krótkim promptcie Llama około 7,3× więcej
  generation tok/s oraz 7,3× krótsze TTFT. Domyślny CPU profile używa rdzeni P;
  obie fazy można ustawić niezależnie.
- Metal na tym Macu wygrał we wszystkich porównanych workloadach. Przy 512 tokens
  Llama generuje 192,0 vs 15,9 tok/s (około 12×); LFM 245,5 vs 28,6 (około 8,6×).
  TTFT odpowiednio 0,206 vs 23,556 s oraz 0,190 vs 16,303 s.
- 4 host threads dają małe różnice (około 0–1,3% względem 1) w tych próbach.
  W default GPU profile pozostaje 1; brak podstaw do globalnej zmiany na 4
  na wszystkich Apple devices na podstawie trzech prób i jednego hosta.
- Batch/microbatch 1024/1024 przy 512 tokens przyspiesza prefill Llama o około 4%,
  ale przy 2048 jest około 2% wolniejszy. 1024/256 także nie wygrywa z 256/256.
  Oddzielne parametry są udostępnione, default nie jest automatycznie powiększany.
- Zachowano synchronizację decode, FP16 KV i domyślne AUTO Flash Attention.
  W tej pracy nie ma dowodu uzasadniającego zmianę tych ustawień.
- Nie włączono backend-attached sampling. Eksperyment z wcześniejszego audytu
  był wolniejszy; obecna poprawa greedy jest zmianą host sampler chain.

CPU-only nie jest najlepszym wyborem performance na badanym M1 Max dla tych
dwóch modeli. Wynik nie dowodzi, że GPU zawsze wygrywa na dowolnym urządzeniu,
modelu, stopniu offloadu czy przy innej presji pamięci. Nie mierzyliśmy energii
ani fizycznego iPhone'a.

### Dane

- [Llama JSONL](llama-1b.jsonl), [CSV](llama-1b.csv): 56 prób, 14 workloadów,
  w tym 14 warmupów i 42 właściwe pomiary.
- [LFM JSONL](lfm-1.2b.jsonl), [CSV](lfm-1.2b.csv): 24 próby, 6 workloadów,
  w tym 6 warmupów i 18 właściwych pomiarów.
- [Sampler JSONL](sampler.jsonl): 16 prób, dwie implementacje, po 7 właściwych prób.
- [Backend assignments](backend-assignments.txt): CPU 0/17 i Metal 17/17 layers
  dla Llama; CPU i Metal buffers sprawdzone również dla LFM.

Hashe modeli z kwalifikacji:
Llama: 6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83;
LFM: bb741ebb106d543e9de114b843a3d3d73d51c74b5801e69da2abde821a0cb3e1.
Nie porównujemy tych wyników liczbowo z wcześniejszym C++ audytem jako
before/after: prompty, długość generacji i sampler nie były identyczne.
Kontrolowanym before/after jest osobny test greedy host sampler.
