# Naprawy wrappera i pomiary CPU / Metal

Kontrola rzeczywistych odpowiedzi i zgodności z niezależną generacją C:
[GENERATION-QUALITY.md](GENERATION-QUALITY.md). Nowy wrapper zachowuje tekst
i tokeny w 36/36 próbach; raport pokazuje również błędy jakościowe modeli.

Praca na branchu fix/llama-wrapper-correctness-performance, po audycie z 30.09.2026.
Raport w Audit/ opisuje stan sprzed napraw. Runtime pozostaje przypięty do b10964,
b29c606e28a01b1bc8c1351026a0fa6e616bf6c4.

## Co naprawiono i jak to sprawdzono

| Audyt | Zmiana | Test / dowód |
|---|---|---|
| F01, F03 | Dwufazowe ustalenie rozmiaru buforów tokenizacji i token pieces; brak zależności od kontekstu treningowego | Tokenizacja vocab-only i >131k tokenów zgodna z C; wszystkie 123 pieces >64 B zgodne z C |
| F02 | Bajtowe pieces i stanowy decoder UTF-8 w serwisie i executorze; detokenize zachowuje NUL | Wszystkie granice bajtów i rzeczywistych tokenów; polski, emoji, arabski, chiński, hindi; wymuszona real generation |
| F04, F08 | BOS zgodny ze słownikiem bez powielania BOS z promptu; EOS dostaje vocab pointer | Bezpośrednie porównanie API C, kwalifikacja LFM/Gemma oraz cache reuse |
| F05, F10 | Batch posiada bufory; sprawdza tryb, pojemność, pozycje i szerokość embeddingów | 1000 cykli singleSequence; granice i stride; Address Sanitizer |
| F06 | Penalties przed filtrami; prompt zasila tylko penalties, nie gramatykę; greedy omija filtry | Kontrolowane logits dowodzą zmiany wyboru przy top-k=1; kolejność etapów; natywna GBNF generation |
| F09 | Usunięty sampler zwalniany, selector chroniony; model utrzymywany przez sampler/clone/adapter, adapter przez kontekst | Weak-reference lifetime; prawdziwe load/apply/decode/remove syntetycznego LoRA |
| F11 | Tokeny dopisywane po sukcesie decode; błąd czyści oba stany; legacy service czeka na zatrzymanie generacji | Abort CPU, puste oba cache po błędzie, następna generacja; cancellation/reuse tests |
| F12 | Błąd grammar rzucany; cytowane, unikalne i wymagane klucze JSON; stabilne nazwy GBNF; escapes | Natywny parser odrzuca brak/duplikaty/extra keys, akceptuje Unicode i quoted braces; typed test musi zdekodować JSON |
| F12 | Typed respond dekoduje całą odpowiedź i propaguje błąd; inferencja odrzuca nieobsługiwane enumy i ogranicza rekursję | Real object/array/scalar tests, unsupported enum i recursive schema |
| F14 | Context ograniczany przed alokacją; CPU wyłącza też KQV/op offload; niezależne threads i microbatch | Walidacja config, real CPU decode, osobne pomiary CPU/Metal i logi backend assignments |
| F15 | Backend procesowy, deferred shutdown do zwolnienia modeli | Memory handle utrzymuje context/model mimo shutdown; ostatni owner zwalnia zasoby |
| F16 | Engine snapshot odtwarza tokens, sampler, UTF-8 i logits poza pamięcią C | Osiem kolejnych losowanych tokens identycznych przed/po restore; obce i puste dane odrzucane |
| F16 | Embeddings mają n_embd_out; callback abort zwalniany po detach/synchronize; granice session capacity | Porównanie z pinem C; abort/lifetime i sanitizer |
| F18 | Dynamiczne bufory metadata/description/split; puste splits i invalid templates | Długie metadata i 1500 B split path; invalid/missing template tests |

Pierwsze cztery nowe testy odtworzyły pięć problemów starego kodu: EOS/BOS,
NUL, sampler order i ignorowaną gramatykę. Dalsze testy wykryły niepoprawne
nazwy reguł GBNF, escapes JSON oraz brak logits w serializacji upstream.
llama_context::state_write_data zapisuje architekturę i pamięć, nie bieżące
outputs. Samo wczytanie C bytes nie wystarcza do wznowienia generacji.

## Zmiany kontraktów API

- LlamaSampler(config:model:) rzuca błąd invalid config/grammar. Caller dodaje
  try; wymaga uwzględnienia przy wersjonowaniu publicznego pakietu.
- LlamaConfig zachowuje stare wywołania init i dodaje opcjonalne
  microBatchSize, nThreads, nThreadsBatch.
- useGPU=false wyłącza layer, KQV i operation offload. Symulator zawsze używa
  CPU. Jeśli backend nie udostępnia GPU offloadu, wybierany jest profil CPU.
  Domyślnie CPU wybiera rdzenie P z hw.perflevel0.physicalcpu; fallback
  to maksymalnie 4 aktywne rdzenie. GPU nadal domyślnie używa 1 host thread.
- Batch operations zwracają Bool; invalid operations nie zmieniają buforów.
  setEmbedding dodaje kompletny vector z następną pozycją.
- piece renderuje pojedynczy fragment z replacement dla niekompletnego UTF-8;
  streaming używa pieceBytes i zachowuje niekompletny suffix między tokenami.
- Typed inference nadal nie jest uniwersalnym JSON Schema. Obsługuje
  syntetyzowane struktury i primitive/array shapes; optional keys zawsze
  emitowane jako wartość lub null. Enumy nie są automatycznie enumerowane.
  Ostateczne ograniczenia typu weryfikuje JSONDecoder; błędy są propagowane.
- Surowe LlamaContext.saveState/loadState pozostają snapshotami pamięci C,
  nie portable transcript/sampler snapshots. Internal engine przyjmuje tylko
  ostatni własny snapshot i zachowuje dodatkowe składniki stanu.

## Granice tej pracy

F13 pozostaje otwarte dla dowolnych GGUF: C API nie wykonuje arbitralnego Jinja.
Kwalifikowane LFM i Gemma renderery pozostają; integracja common/chat wymaga
osobnego bridge i testów zgodności. Nie zmieniono KV precision, Flash Attention
policy, częściowego offloadu, memory budget ani synchronicznego decode.
Wysokie API jawnie odrzuca encoder/diffusion models.
Raw C params nadal wymagają zarządzania lifetime callbacków/pointer members
przez zaawansowanego callera; raw context/sampler nie są thread-safe.
Zbieranie C perf counters pozostaje domyślnie wyłączone: benchmark używa wall clock, a
performanceData nie jest dowodem realnego throughput przy wyłączonych counters.
Warmup, cached-token metrics i szerokie JSON Schema wymagają osobnej pracy.

Address Sanitizer obejmuje kod Swift i interceptory alloc/free; prebuilt
C/C++ XCFramework nie jest w całości instrumentowany. Nie zastępuje pełnego
ASan build upstream. Fizyczny iPhone, Intel Mac, energia, thermal throttling
i szczyt pamięci wymagają osobnych pomiarów.

## Weryfikacja

- Pełne Mac tests: 119 testów / 21 suites, także LFM i Gemma.
- Nowe regresje pod Address Sanitizer: 21 testów / 4 suites.
- iOS 27 Simulator: 35 testów / 5 suites.
- Aplikacje macOS i iOS Simulator: build exit 0.
- CPU/Metal Release benchmark: 56 prób Llama i 24 LFM; oba testy exit 0.
- Izolowany benchmark samplera: 16 prób; identyczne wybrane tokens; exit 0.

Skrócone wyniki: [test-results.txt](test-results.txt).
Dwa istniejące story quality tests skorygowano semantycznie: akceptacja słowa
feline jako cat oraz dopasowanie instrukcji do 320-tokenowego budżetu.
Nie zmieniono asercji wymagających poprawnego stanu, JSON lub tekstu Unicode.

## Benchmark

Instrukcje i wyniki: [PERFORMANCE.md](PERFORMANCE.md).

Bezpośrednie GPU before/after: [GPU-BEFORE-AFTER.md](GPU-BEFORE-AFTER.md).
Kontrolowane greedy zyskuje; domyślny sampler produkcyjny jest wolniejszy
w części workloadów. Raport zawiera pełne dane i zakres porównania.
