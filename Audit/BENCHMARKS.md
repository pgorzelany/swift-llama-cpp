# Wyniki i reprodukcja

## Warunki

M1 Max (8P + 2E), 64 GB, macOS 27.0, Swift 6.4 / Xcode 27. Framework identyczny z używanym przez wrapper: `b10964`; nagłówki macOS, iOS device i iOS Simulator porównano przez `cmp` z przypiętym `include/llama.h` — zgodne.

C++ probes budowane są `clang++ -O3`, ale linkują istniejący framework; nie tworzą nowej, inaczej zoptymalizowanej kopii llama.cpp. Bieżący wrapper nie był modyfikowany.

Każda konfiguracja: jeden niemierzony warmup, potem trzy mierzone powtórzenia. Jeden model/context naraz, ciepłe wagi; przed próbą czyszczona pamięć sekwencji i resetowany sampler. Prompt jest sztucznym ciągiem rzeczywistych tokenów, 64 tokeny + 16 generation steps dla CPU oraz 2048 + 128 dla Metal. Generation steps nie kończą się na EOG, aby długość prób była porównywalna; to benchmark obliczeń, nie quality evaluation. Pomiar prefill kończy synchronizacja, generation obejmuje sampling i decode, bez Swift, UI, rendering Unicode i transportu transcript. C timings w głównym probe są aktywne przez `no_perf=false`.

`cpu` w CSV znaczy **n_gpu_layers=0**, z osobną kolumną offload_ops określającą offload_kqv/op_offload. Nie należy interpretować samego n_gpu_layers=0 jako gwarancji CPU-only. Wyniki obu wariantów potwierdzają, że w tej próbie podstawowym ograniczeniem był jeden wątek, nie tylko niejawny offload operacji.

Kolumna sampling: 0 = prosty greedy selector; 1 = kolejność obecnego wrappera (top-p .95 → repeat 1.1 / lastN 64 → temp .5 → dist seed 42); 2 = ta sama chain z dodanym top-k 40. Pozostałe porównania używają sampling=0, aby oddzielić ustawienia obliczeń od jakości samplerów. Dane sampling=1/2 mają inny losowy ciąg tokenów; nie dowodzą wpływu samego top-k na jakość ani identyczności wyników.

## Llama 3.2 — CPU

| Threads | Batch/ubatch | Sync | FA | KV | Sampling | Offload ops | PP tok/s median | TG tok/s median (range) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 256/256 | 1 | AUTO | f16 | 0 | 1 | 2.89 | 2.23 (2.19–2.24) |
| 4 | 256/256 | 1 | AUTO | f16 | 0 | 1 | 11.22 | 8.54 (8.53–8.55) |
| 8 | 256/256 | 1 | AUTO | f16 | 0 | 1 | 19.53 | 14.55 (13.03–14.70) |
| 1 | 256/256 | 1 | AUTO | f16 | 0 | 0 | 2.86 | 2.22 (2.22–2.23) |
| 8 | 256/256 | 1 | AUTO | f16 | 0 | 0 | 20.05 | 14.39 (14.30–14.59) |

Mediana 1→8 threads: około **6,8× prefill i 6,5× generation**. To dowód kosztu sztywnego `1/1` dla tej konfiguracji CPU na tym Macu. Nie jest to przewidywanie przyspieszenia zwykłej aplikacji z Metal ani symulatora względem iPhone’a. Prompt CPU jest krótki; osobno należy zbadać długi prefill i Accelerate/BLAS.

## Llama 3.2 — Metal

| Threads | Batch/ubatch | Sync | FA | KV | Sampling | Offload ops | PP tok/s median | TG tok/s median (range) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 256/256 | 1 | AUTO | f16 | 0 | 1 | 2336.21 | 139.65 (135.00–163.41) |
| 4 | 256/256 | 1 | AUTO | f16 | 0 | 1 | 2096.01 | 167.08 (157.56–167.87) |
| 1 | 1024/1024 | 1 | AUTO | f16 | 0 | 1 | 2030.60 | 161.79 (160.97–165.08) |
| 1 | 1024/256 | 1 | AUTO | f16 | 0 | 1 | 2030.52 | 152.16 (142.65–163.29) |
| 1 | 256/256 | 0 | AUTO | f16 | 0 | 1 | 1999.50 | 159.24 (157.85–161.72) |
| 1 | 256/256 | 1 | OFF | f16 | 0 | 1 | 1967.64 | 111.86 (111.60–119.58) |
| 1 | 256/256 | 1 | ON | q8_0 | 0 | 1 | 1978.90 | 146.39 (144.64–147.11) |
| 1 | 256/256 | 1 | AUTO | f16 | 1 | 1 | 1967.71 | 151.54 (150.12–151.69) |
| 1 | 256/256 | 1 | AUTO | f16 | 2 | 1 | 2011.08 | 144.74 (136.39–160.30) |

## LFM 2.5 QAD — Metal

| Threads | Batch/ubatch | Sync | FA | KV | Sampling | Offload ops | PP tok/s median | TG tok/s median (range) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 256/256 | 1 | AUTO | f16 | 0 | 1 | 2585.24 | 187.45 (168.83–188.12) |
| 4 | 256/256 | 1 | AUTO | f16 | 0 | 1 | 2601.24 | 166.57 (164.50–212.47) |
| 1 | 1024/1024 | 1 | AUTO | f16 | 0 | 1 | 2753.12 | 193.24 (168.23–236.63) |
| 1 | 1024/256 | 1 | AUTO | f16 | 0 | 1 | 2408.80 | 196.09 (172.93–230.71) |
| 1 | 256/256 | 0 | AUTO | f16 | 0 | 1 | 2545.07 | 178.49 (171.99–184.75) |
| 1 | 256/256 | 1 | OFF | f16 | 0 | 1 | 2299.64 | 146.74 (143.48–187.29) |
| 1 | 256/256 | 1 | ON | q8_0 | 0 | 1 | 2276.43 | 215.82 (191.34–218.21) |
| 1 | 256/256 | 1 | AUTO | f16 | 1 | 1 | 2346.29 | 189.36 (188.97–191.47) |
| 1 | 256/256 | 1 | AUTO | f16 | 2 | 1 | 2260.73 | 188.63 (168.69–222.96) |

Wnioski: zachować AUTO Flash Attention jako bazę; wyłączenie FA było w obu próbach wolniejsze. Cztery host threads nie były uniwersalnie lepsze (dla LFM generation spadła). Batch 1024/ubatch256 jest realną dostępną opcją, ale nie zapewniał jednolitego zysku. Q8 KV nie dał jednolitego przyspieszenia między modelami; wpływu na footprint i jakość nie zmierzono. Brak dodatkowego synchronize również nie wygrał w każdym przypadku. Nie ma pomiarowego uzasadnienia do zastosowania jednego „najlepszego” zestawu globalnie.

Kolejność konfiguracji jest stała. Krótkie serie mają zmienność i możliwy wpływ temperatury / systemowego obciążenia; widać to w zakresach. Małych różnic nie należy interpretować jako pewnej przewagi. Do strojenia release potrzebne są przeplatane / losowane serie, powtórzenie przy stabilnym thermal state i normalne prompty z aplikacji.

## Backend sampling na Metal

| Backend requested | Attach | Backend selected token | PP median tok/s | TG median tok/s |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 2363.92 | 166.00 |
| 1 | 1 | 1 | 2340.01 | 118.75 |

Attach zwrócił true, a `llama_get_sampled_token_ith` zwrócił faktyczny token (nie LLAMA_TOKEN_NULL). Mechanizm działa na tym frameworku i backendzie. Median generation **166,0 → 118,8 tok/s**, około **28,5% spadku**. Nie jest obecnie kwalifikowaną optymalizacją dla tego modelu / chain. Nie zmierzono wszystkich wariantów selectorów ani identyczności jakości/seedów między CPU i backend RNG. Dowód: [backend-sampling.csv](evidence/backend-sampling.csv).

## Koszt emitowania transcript

Syntetyczny engine, bieżący executor i prawdziwa Apple session na Macu; zero obliczeń modelu. Dodatkowe próby:

| Output tokens | Czas |
| --- | --- |
| 1000 | 0,0413 s |
| 4000 | 0,1630 s |
| 8000 | 0,3928 s |

Ta próba nie wskazuje obecnego streamingu jako głównego ograniczenia krótkich generacji. Narastające metadata/rawOutput pozostają kandydatem do zmniejszenia kopiowania, szczególnie dla dłuższego tekstu / innych urządzeń. Te liczby obejmują również Apple session; nie izolują samego emitera.

## Testy poprawności

- Istniejący `swift test --no-parallel`: zakończony sukcesem; runner raportuje 91 tests / 14 suites, 205,676 s. Pominięte deklaracje: Gemma4, dwa cache tests, shipped LFM tool integration. Dokładne skip messages w [baseline-summary.txt](evidence/baseline-summary.txt).
- Następnie testy z podanymi lokalnymi LFM/Gemma i audit probes: 8 tests / 4 suites, **zero skips**, 72,733 s. Cache test ma także dwa warianty batch size. Gemma zwróciła GEMMA4_OK, LFM tool test wykonał pięć sesyjnych zapytań z kontrolą testowych record values, unrelated question i recovery. Cache LFM używał bezpiecznego full reprocessing, gdy recurrent trim odmówił. [qualified-summary.txt](evidence/qualified-summary.txt).
- Próby obserwacyjne potwierdziły utratę emoji, 123 pieces >64 bytes, niepoprawny BOS i ciche pominięcie błędnej grammar. Identyczny prompt bez nowych tokenów zachował dokładnie te same logits. [observations.txt](evidence/observations.txt).
- Izolowane reprodukcje long-piece, long-tokenize, vocab-only-tokenize i borrowed-batch celowo kończą proces signal 5/5/5/6; błędny EOS zwracał niepoprawną wartość bez crasha. [failure-probes.txt](evidence/failure-probes.txt).
- Dodatkowy `controlTokenRendering`: 1 test, 0,489 s, sukces. Rzeczywiste Gemma delimitery kanału zachowują się przy obu ustawieniach `renderSpecial`. Próba używa poprawnego C tokenization sizing, ponieważ wrapper crashuje przy `vocab_only` / `n_ctx_train=0`. [gemma-protocol-summary.txt](evidence/gemma-protocol-summary.txt).
- iOS Simulator jest kontrolą kompilacji / poprawności, nie pomiarem fizycznego iPhone’a.

iPhone 17 Pro / iOS 27 Simulator: **TEST SUCCEEDED**, 32 tests / 3 suites, 18,512 s (Xcode session 28,778 s), zero skips. Zestawy: LlamaLanguageModelTests, LlamaToolCallingTests i cztery ówczesne audit probes. Potwierdzono również BOS / utratę emoji / długie pieces na symulatorze. [ios-summary.txt](evidence/ios-summary.txt).

## Polecenia

Z katalogu pakietu, po przygotowaniu obecnego frameworka i fixture:

```sh
swift test --no-parallel
LLAMA_AUDIT=1 swift test --no-parallel --filter LlamaAuditProbes

LLAMA_AUDIT=1 \
ENCLAVE_GGUF_TEST_MODEL="$PWD/../../SharedIntelligence/LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf" \
GEMMA4_GGUF_PATH="$PWD/Tests/Models/gemma-4-E2B-it-Q4_0.gguf" \
swift test --no-parallel --filter 'LlamaAuditProbes|LlamaCacheReuseTests|LlamaToolCallingIntegrationTests|GemmaCompatibilityTests'

bash Audit/run-cpp-probe.sh > /tmp/llama-audit.csv
bash Audit/run-cpp-probe.sh /absolute/path/to/LFM.gguf --metal-only > /tmp/lfm-audit.csv
bash Audit/run-backend-probe.sh > /tmp/llama-backend-audit.csv

python3 Audit/generate-method-index.py
```

Crash probes uruchamiać **po jednej, w osobnych procesach**, bez iOS runnera i bez równoległego benchmarku. Nie są expected-green tests. `LLAMA_AUDIT_FAILURE` dostępne wartości: long-piece, long-tokenize, vocab-only-tokenize, borrowed-batch, eos-pointer.

```sh
LLAMA_AUDIT=1 LLAMA_AUDIT_FAILURE=long-piece \
swift test --skip-build --no-parallel --filter 'LlamaAuditProbes/isolatedFailure'
```

Simulator command użyty w audycie (device ID dotyczy tej maszyny):

```sh
TEST_RUNNER_LLAMA_AUDIT=1 xcodebuild -scheme swift-llama-cpp \
  -destination 'platform=iOS Simulator,id=5761AFC8-92FF-48EA-990D-DEAA861739EE' \
  -derivedDataPath /tmp/EnclaveLlamaAudit-iOS20260930 \
  -only-testing:SwiftLlamaTests/LlamaLanguageModelTests \
  -only-testing:SwiftLlamaTests/LlamaToolCallingTests \
  -only-testing:SwiftLlamaTests/LlamaAuditProbes \
  -parallel-testing-enabled NO CODE_SIGNING_ALLOWED=NO test
```

Pomiary w CSV zapisują każdą próbę, nie tylko mediany. Odrzucono wstępne uruchomienia eksperymentalne, w tym jedną zbyt długą próbę CPU i run z błędem shell po zmianie skryptu podczas jego wykonania; przedstawione główne dane pochodzą z kolejnych kompletnych runów z exit 0. Nie uruchamiać innych model tests równolegle z pomiarem. Fizycznego iPhone’a, Intel Mac, battery/energy, peak footprint ani quality po KV quantization nie zmierzono.
