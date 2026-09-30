# Audyt SwiftLlama / llama.cpp — 30 września 2026

## Werdykt

Wrapper ma rozsądną architekturę podstawowej inferencji, ale **nie jest w pełni poprawny i nie ma podstaw, żeby uznać jego konfigurację za optymalną na wszystkich platformach**. Najpierw należy naprawić reprezentację tokenów i kontrakty C, później stroić wydajność. Sama aktualizacja llama.cpp nie naprawi znalezionych błędów Swift.

Najważniejsze problemy bieżącej ścieżki aplikacji: utrata fragmentów UTF-8, crash dla długiego tokenu, crash przy tokenizacji bardzo długiego wejścia, pomijanie BOS dla części tokenizerów, niewłaściwa kolejność kar względem filtrów samplerów i sztywne ustawienie jednego wątku. Publiczne pomocnicze API ma dodatkowo błędy własności pamięci i niepoprawny wskaźnik EOS.

Ten branch zawiera audyt i odtwarzalne próby. Nie zmienia produkcyjnego runtime, modeli ani ustawień użytkowników.

## Zakres i baza porównania

| Element | Audytowana baza |
| --- | --- |
| Swift wrapper | `b7f9e68` (`Stop logging private chat and tool prompts`) |
| Aplikacje i integracja | `731fd261575159b9e0c3aa033bf5b1ce1b09b409` |
| Framework | `b10964`, llama.cpp v0.4.1, `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4` |
| Aktualny upstream sprawdzony podczas audytu | `05af0d2b1398394cfa67e1918fee7feabccaa9bc`, commit z 30.09.2026 17:10 CEST |
| Host pomiarów | Apple M1 Max, 10 rdzeni CPU (8P + 2E), 64 GB, macOS 27.0 `26A428`, Swift 6.4 |
| Model bazowy | Llama-3.2-1B-Instruct-Q4_K_M, SHA-256 `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83` |
| Model aplikacji do kwalifikacji tools | LFM2.5-1.2B-Instruct-QAD-Q4_0, SHA-256 `bb741ebb106d543e9de114b843a3d3d73d51c74b5801e69da2abde821a0cb3e1` |

Przeczytano wszystkie pliki `Sources/SwiftLlama`, testy istotnych kontraktów, manifest, przygotowanie frameworka, fabrykę sesji, mapowanie opcji i konfigurację miejsc użycia w aplikacjach. [Indeks metod](METHODS.md) obejmuje każdą deklarację funkcji, initializer i deinitializer w produkcyjnym kodzie Swift, także pomocnicze typy generowania gramatyk. Zgrupowane poniżej oceny wyjaśniają kontrakty i odsyłają do konkretnych problemów.

Źródła pierwotne porównania:

- [Przypięte C API](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/include/llama.h).
- [Model i domyślne parametry](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/src/llama-model.cpp), [słownik i tokenizacja](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/src/llama-vocab.cpp).
- [Batch i własność pamięci](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/src/llama-batch.cpp), [context / decode / stan](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/src/llama-context.cpp).
- [Samplery C](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/src/llama-sampler.cpp), [referencyjna warstwa sampling](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/common/sampling.cpp), [domyślne ustawienia narzędzi](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/common/common.h).
- [Prosty przykład inferencji C++](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/examples/simple/simple.cpp).
- [Chat w C](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/src/llama-chat.cpp), [Jinja / tools w common](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/common/chat.h).
- [LoRA](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/src/llama-adapter.cpp), [recurrent memory](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/src/llama-memory-recurrent.cpp).
- [Budowa Apple XCFramework](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/build-xcframework.sh).
- [Aktualne C API](https://github.com/ggml-org/llama.cpp/blob/05af0d2b1398394cfa67e1918fee7feabccaa9bc/include/llama.h), [aktualne mechanizmy speculative decoding](https://github.com/ggml-org/llama.cpp/blob/05af0d2b1398394cfa67e1918fee7feabccaa9bc/docs/speculative.md).

## Rzeczywiste ścieżki aplikacji

`LocalChatSessionFactory` → `LlamaModelSessionFactory` → `LlamaLanguageModelExecutor` → `LlamaExecutorRuntime` → actor `Llama` → `LlamaModel`, `LlamaContext`, `LlamaBatch`, `LlamaSampler`.

`LlamaService` jest nadal publicznym API biblioteki, ale nie jest bieżącym silnikiem czatu i voice w Enclave. Osobny `AnyLanguageModel/Models/LlamaLanguageModel.swift` nie jest tym wrapperem i nie był przedmiotem audytu metoda po metodzie. Nie należy przenosić na niego automatycznie tego raportu.

| Użycie | Batch / ubatch | Context | Sampling |
| --- | --- | --- | --- |
| iOS text chat | 256 / 256 | ustawienie sesji; domyślnie 2048 przy RAM < 3,5 GB, inaczej 8192 | temperatura użytkownika (domyślnie 0,5), top-p 0,95, losowy lub stały seed |
| macOS text chat | 1024 / 1024 | jak wyżej | jak wyżej |
| iOS i macOS voice | 256 / 256 | ustawienie sesji | jak wyżej; tools wyłączone w fabryce voice |
| iOS AskEnclaveIntent | 256 / 256 | 2048 | odrębne wywołanie fabryki |
| benchmark EnclaveModelRuntime | domyślnie 512 / 512 | domyślnie 4096 | sesja i scenariusz benchmarku |

W każdym z tych przypadków `Llama.init` ustawia `n_threads = 1` i `n_threads_batch = 1`. Wypisuje jednak „Using 9 threads” na tym Macu: log nie odzwierciedla rzeczywistej konfiguracji. Symulator wymusza `n_gpu_layers = 0`; jego wyników nie można traktować jako wydajności iPhone’a z Metal.

## Ustalenia i priorytety

P1 oznacza problem do naprawy przed następnym strojeniem lub poszerzaniem wsparcia modeli. P2 oznacza istotną poprawkę kontraktu lub zmierzoną możliwość optymalizacji. P3 oznacza ograniczenie / higienę API. Priorytet zależy również od tego, czy metoda jest używana przez aplikację.

### F01 — P1: `piece` zakłada maksymalnie 64 bajty i nie obsługuje ujemnego wyniku C

`LlamaModel.piece` alokuje 64 bajty. `llama_token_to_piece` zwraca ujemną wymaganą długość, gdy bufor jest za mały. Swift bez sprawdzenia używa jej w `prefix(upTo:)`, co kończy proces.

Potwierdzenie na modelu testowym: 123 tokeny mają kawałek dłuższy niż 64 bajty. Token `5713` potrzebuje 72 bajtów i reprodukuje `Fatal error: Range requires lowerBound <= upperBound`, signal 5. Ten helper jest wywoływany dla każdego wygenerowanego tokenu przez bieżącą aplikację. Duże fragmenty whitespace również są tokenami; problem nie wymaga egzotycznego modelu.

Naprawa: dynamiczne ponowienie po ujemnym wyniku, sprawdzenie zakresów / overflow i zwracanie bajtów, nie samodzielnego `String` dla każdego tokenu.

### F02 — P1: dekodowanie UTF-8 token po tokenie gubi treść

Token nie musi kończyć się na granicy znaku Unicode. `String(cString:…, encoding: .utf8) ?? ""` zamienia niepełny fragment na pusty tekst. Strata zachodzi przed parserem reasoning, metadata raw output i transcript replay, więc „rawOutput” nie jest wtedy bezstratnym zapisem wyjścia modelu.

Potwierdzenie: tokenizacja `Zażółć gęślą jaźń 🎯🚀🔥` i połączenie `piece` daje `Zażółć gęślą jaźń`; pełne `llama_detokenize` zachowuje emoji. Testy sprawdzające brak `�` nie wykrywają cichego usunięcia znaków.

Naprawa: stanowy decoder UTF-8 przy generacji, z buforem reszty między tokenami, dynamicznym buforem `token_to_piece` i jasno zdefiniowanym flush przy EOG / cancellation. Liczba tokenów musi uwzględniać także tokeny, które nie dają jeszcze pełnego znaku. C++ wypisuje fragmenty bajtów do jednego strumienia; granice tokenów nie stają się granicami dekodowania Unicode.

### F03 — P1: `tokenize` podaje rozmiar treningowego kontekstu zamiast pojemności bufora

Rzeczywista alokacja to `utf8Count + specials + 1`, ale `n_tokens_max` wynosi `trainedContextSize()`. To inny kontrakt: limit kontekstu inferencji nie jest rozmiarem bufora tokenizatora. Wynik ujemny nie jest obsługiwany i znowu trafia do `prefix(upTo:)`.

Potwierdzenie: 131083 tokeny przy `n_ctx_train = 131072` reprodukują signal 5. Zwykły dużo mniejszy limit aplikacji może dać poprawny błąd context window; bardzo duże wklejone wejście powoduje crash jeszcze przed kontrolą limitu w `initializeCompletion`. `contextUsage` korzysta z tego samego tokenizatora.

Ten sam błąd dotyczy `vocab_only`: w tym trybie `n_ctx_train = 0`, więc nawet krótki tekst powoduje crash w publicznym `tokenize`. Osobna próba dla słownika Gemma potwierdziła pięć poprawnie tokenizowanych tokenów przez C API przy `n_ctx_train = 0`; sam tokenizator nie potrzebuje treningowego kontekstu.

Nie wykazano przepełnienia zaalokowanej pamięci dla tego GGUF: heurystyczny bufor jest większy niż wymagana liczba tokenów. Udowodnione są błędny argument API i crash na ujemnym wyniku, a nie dowolny zapis poza bufor.

Naprawa: jak `examples/simple/simple.cpp`, najpierw zapytanie o wymaganą liczbę tokenów, następnie dokładna alokacja i sprawdzenie wyniku; ewentualnie bezpieczny wstępny bufor i ponowienie. Limit wejścia i błąd context należy sprawdzać niezależnie. Rozważyć tanie ograniczenie bardzo dużego tekstu przed pełnym renderowaniem i tokenizacją.

### F04 — P1: `shouldAddBos` ignoruje wymaganie BOS poza SentencePiece

Jeśli `llama_vocab_get_add_bos` jest true, metoda zwraca true wyłącznie dla `LLAMA_VOCAB_TYPE_SPM`. Wymaganie słownika dotyczy także innych tokenizerów.

Potwierdzenie: Llama 3.2 GGUF ma `native=true`, wrapper zwraca `false`; BOS to `128000`, pierwszy token rzeczywistego sformatowanego promptu to `128006` (`start_header`). C chat template Llama 3 nie dodaje BOS. Prompt działa i podstawowy test odpowiada „Paris”, ale to nie jest prawidłowy format wejścia modelu.

Naprawa: respektować `add_special` i ustawienia vocab, jednocześnie zapobiegać podwójnemu BOS, gdy pełny renderer już go wstawił. Nie zamieniać obecnego wyjątku na bezwarunkowe `true`: Gemma/LFM mają osobne renderery i wymagają testów dokładnej sekwencji tokenów. Testować BOS/EOS/SEP dla BPE i SPM oraz rzeczywistego GGUF każdej wspieranej rodziny.

### F05 — P1 dla publicznego API: `singleSequence` miesza bufor pożyczony z posiadanym

`llama_batch_get_one` nie alokuje pamięci: zapisuje wskaźnik do tablicy podanej przez caller. Swift przekazuje lokalną tablicę `cTokens`, której żywotność nie obejmuje użycia zwróconego wrappera. Następnie zastępuje `rawBatch` wcześniej zaalokowany przez `llama_batch_init`, tracąc tę alokację. `deinit` wywołuje `llama_batch_free` na pamięci należącej do Swift.

Potwierdzenie: `LlamaBatch.singleSequence(tokens: [1, 2, 3])` i zwolnienie obiektu kończą proces signal 6. Są tu trzy niezależne problemy: lifetime, utrata starej alokacji i invalid free. Bieżący actor używa `init(initialSize:)`, nie `singleSequence`, więc nie przypisuję tego crasha zwykłej generacji w aplikacji.

Naprawa: użyć posiadanego batcha i skopiować tokeny / positions / flags albo wydzielić pożyczony batch dostępny tylko wewnątrz scope closure. Nie wystarczy samo pominięcie `free`.

### F06 — P2: sampler stosuje penalties dopiero po top-k/top-p

Rzeczywista kolejność to `grammar? → top-k? → top-p → penalties → temp → dist`. Referencyjna kolejność narzędzi upstream stosuje penalties przed filtrami. Po odrzuceniu tokenu przez top-k/top-p kara dla konkurencyjnego tokenu nie może go przywrócić. Kara zmienia więc inną dystrybucję niż sugeruje konfiguracja.

To istotne również w trybie „greedy”: temperatura 0 jest ustawiana dopiero po wcześniejszych filtrach. Najlepszy token po zastosowaniu kary mógł już zostać usunięty. Samo `temp=0` zapewnia wybór maksimum pozostałych kandydatów, a nie pełnych logits po karach.

Sampler jest odtwarzany po przygotowaniu promptu i nie dostaje historii promptu przez `accept`; kara dotyczy tylko dotychczas wygenerowanego fragmentu bieżącej odpowiedzi. Jeśli to zamierzone, trzeba ten zakres udokumentować. Upstream rozdziela akceptację tokenów promptu do historii kar i tokenów wyjścia do gramatyki. Obecny komentarz `accept` sugerujący karmienie całym promptem również grammar jest mylący.

Naprawa: oddzielić grammar od history penalties, ustalić politykę kar dla promptu/history i stosować kary przed odcinaniem kandydatów. Osobna prosta ścieżka greedy po wymaganych transformacjach zmniejszy koszt. Zweryfikować jakość i tokeny na stałym seedzie; nie zmieniać kolejności jako „czysto wydajnościowej” poprawki.

### F07 — P2, zmierzony: jeden wątek dla wszystkich urządzeń i obu faz

`n_threads` i `n_threads_batch` mają inne zastosowania. Ustawienie `1/1` jest możliwym dobrym wariantem przy pełnym Metal, ale nie uzasadnia polityki dla CPU, Intel Mac, fallbacku i symulatora. Log o `processorCount - 1` jest dodatkowo nieprawdziwy.

Benchmark tego samego frameworka potwierdził na M1 Max około czterokrotne przyspieszenie po zmianie `1/1 → 4/4` dla `n_gpu_layers=0`; pełne wyniki i ograniczenia są w [BENCHMARKS.md](BENCHMARKS.md). Nie oznacza to czterokrotnego przyspieszenia normalnego czatu z pełnym Metal.

Wariant `8/8` osiągnął 14,55 zamiast 2,23 tok/s generacji, około 6,5×. Prefill wzrósł z 2,89 do 19,53 tok/s. To krótka próba na jednym modelu i Macu; nie ustala uniwersalnej liczby wątków.

Naprawa: jawne, niezależne ustawienia obu faz, profile dla CPU / GPU i możliwość pomiaru na konkretnym urządzeniu. Dla CPU porównać liczbę rdzeni P, nie ślepo `processorCount - 1`. Dla GPU zacząć od 1–4 host threads i utrzymać najlepszy zmierzony wariant. Symulator służy poprawności; iPhone trzeba mierzyć fizycznie.

### F08 — P2: EOS otrzymuje wskaźnik modelu zamiast słownika

`LlamaModel.eosToken()` wywołuje `llama_vocab_eos(modelPointer)`, powinno `vocabPointer`. Swift widzi oba jako `OpaquePointer`, więc typy C nie chronią przed błędem.

Potwierdzenie: wrapper zwrócił `-256782208`, poprawne API `128009`. Wynik wrappera jest nieokreślony i zależy od layoutu / adresów. Generacja używa poprawnego `isEogToken(vocabPointer, …)`, więc błąd nie wyjaśnia zatrzymania bieżącej ścieżki aplikacji. Naprawa jest prosta; dodać test zgodności specjalnych tokenów z vocab.

### F09 — P2: własność usuwanego samplera i adapterów jest niespójna

`LlamaSampler.remove` ignoruje wskaźnik zwrócony przez `llama_sampler_chain_remove`. Upstream przekazuje własność usuniętego samplera callerowi; chain już go nie zwolni. Każde poprawne usunięcie przecieka pamięć. Usunięcie końcowego samplera wyboru pozostawia chain, którego `sample` może zakończyć się asercją. Naprawa: zwolnić lub zwrócić posiadany wrapper, pilnować końcowego selectora.

`LlamaLoraAdapter` nie trzyma referencji do modelu i nie ma `deinit`. W tej wersji model rejestruje adapter w `model.loras` i zwalnia go we własnym destruktorze. Nie jest to permanentny wyciek po zwolnieniu modelu, ale nie ma niezależnego bezpiecznego lifetime adaptera ani wcześniejszego uwolnienia nieużywanych adapterów. Adapter może przeżyć model i zawierać dangling pointer.

Naprawiając adapter, należy jednocześnie: zatrzymać model przez silną referencję, zwalniać `llama_adapter_lora_free` w wrapperze i zapewnić, że context utrzymuje zastosowane adaptery do odłączenia. Dodanie samego `deinit` utworzyłoby use-after-free dla contextu, który w C przechowuje nieposiadane wskaźniki adapterów.

### F10 — P2: brak walidacji trybu i granic batcha w publicznym API

`LlamaBatch` nie przechowuje capacity ani embedding dimension. `addToken` może pisać poza allocation, `setLastTokenLogits` na pustym batchu indeksuje -1, `addToken` na embedding batchu dereferencjonuje nil, `setEmbedding` na token batchu również. Offset embeddingu liczony jest z długości przekazanego vectora zamiast zaalokowanej szerokości; metoda nie inkrementuje token count, nie ustawia position/sequence/logits, a API nie oferuje poprawnej operacji „dodaj embedding”.

Bieżący actor zachowuje pojemność i ostatni niepusty batch prawidłowo, więc nie potwierdzono takiego przekroczenia w jego pętli. API biblioteki nie realizuje jednak obietnicy bezpiecznych Swift typów.

Naprawa: osobne tryby token/embedding, pojemność i szerokość przechowywane w obiekcie, walidowane operacje append z pełnymi polami batcha. Ten sam problem walidacji dotyczy ujemnych `capacity` w session loaderach i pustego `paths` w split loaderze.

### F11 — P2: błąd / cancellation może rozjechać Swift cache i pamięć C w starszej ścieżce

`generateNextToken` dopisuje token do `processedTokens` i zwiększa position przed sukcesem decode. `processPrompt` dopisuje tokeny przed wykonaniem batcha. Fatal error / abort może zostawić w C tylko przetworzone ubatche, a Swift liczy całą przygotowaną porcję. Sampler również zaakceptował już token.

Bieżący `LlamaExecutorRuntime` w catch wykonuje `resetCompletion`, więc zabezpiecza tę ścieżkę. `LlamaService` tylko kończy stream błędem i nie resetuje stanu. Jego consumer cancellation nie jest połączony z `continuation.onTermination`; przerwanie iteracji nie musi zatrzymać producenta. Actor reentrancy między `stopCompletion`, `initializeCompletion` i `updateSamplingConfig` pozwala też przeplatać dwa wywołania legacy API; actor sam w sobie nie jest blokadą całej operacji zawierającej await.

Naprawa: commit cache po udanym decode; przy fatal / abort jawne uzgodnienie pamięci albo reset całej generacji; lifecycle i busy guard dla legacy. Powiązać zakończenie konsumenta z cancel, odłączyć `currentTask` po ukończeniu i nie połykać CancellationError w typed `respond`.

`loadStateData` i `clearKV` w actorze są obecnie helperami testowymi: modyfikują C bez spójnej zmiany `processedTokens`, position i sampler state. Nie traktować ich jako pełnego zapisu / odtworzenia rozmowy. Publiczne `LlamaContext` state APIs opisują stan C, nie stan wysokiego poziomu.

### F12 — P2: gramatyka może zostać cicho wyłączona; typed JSON jest best-effort

Niepoprawny `grammarConfig` powoduje `llama_sampler_init_grammar == nil`, ale initializer samplera kontynuuje bez gramatyki. Potwierdzono chain `[top-p, penalties, temp, dist]` dla `grammar="invalid"`. Caller żądający ograniczonego wyjścia nie otrzymuje błędu.

`LlamaTypedJSONGrammarBuilder` obserwuje jeden syntetyczny przebieg `Decodable.init(from:)`, a nie pełny schemat typu. Enumy, alternatywne ścieżki własnego decoder’a, dictionaries, rekursja, superDecoder i ręczne nested containers mogą dać częściowy lub błędny schemat. Required properties nie są wymagane, keys mogą się powtarzać. Nazwy reguł mogą kolidować po `sanitize`, kolejność dictionary jest niestabilna. Parser JSON w `respond<T>` liczy nawiasy także wewnątrz stringów i ma kosztowne powtarzane skanowanie prefixu. Dla scalar `T` nie znajduje klamry i czeka do końca generacji. Catch połyka błąd strumienia, także cancellation.

To legacy API; bieżący executor jawnie odrzuca guided generation, co jest uczciwym kontraktem. Naprawa: nie ignorować błędu grammar; preferować jawny JSON Schema i sprawdzony konwerter upstream, a best-effort inference zachować wyłącznie jako wyraźnie ograniczoną opcję. Structured generation wymaga limitu odpowiedzi i parsera JSON respektującego string / escape.

### F13 — P2: C chat API nie jest pełnym rendererem Jinja

`llama_chat_apply_template` rozpoznaje znane szablony heurystycznie i uruchamia wbudowany C++ formatter. Nie wykonuje arbitralnego Jinja z GGUF. `nil` template oznacza domyślne ChatML; nie oznacza pewności, że model został wytrenowany na ChatML. Unsupported template zwraca -1, co wrapper zamienia na pusty string / ogólny błąd lub Gemma fallback.

Dedykowany renderer LFM ma kwalifikację konkretnego hasha i preserve_thinking=true; to rozsądna, wąska polityka. Gemma4 fallback jest prostym formatterem po `general.architecture`, a nie pełną implementacją każdej gałęzi szablonu. Parser testowany na syntetycznych fragmentach nie dowodzi zachowania wszystkich GGUF. Dodatkowa próba rzeczywistego słownika Gemma potwierdziła, że `<|channel>thought\nPlan<channel|>` renderuje się identycznie przy `renderSpecial=false` i `true`: dla tego fixture nie znaleziono utraty tych delimiterów. Inne rodziny wymagają własnej kwalifikacji control tokens; nie zakładać, że jedno ustawienie renderowania obejmuje każdy protokół.

Naprawa: porównanie wygenerowanego promptu bajt po bajcie i token po tokenie z `common_chat_templates_apply` dla każdej kwalifikowanej rodziny, jawne błędy unsupported template, rozdzielenie kontroli renderowania protocol tokens i widocznego tekstu. Dla szerokiego importu dowolnych GGUF dodać niewielki bridge do warstwy `common/chat` (wymaga własnej budowy common, której oficjalny framework nie zawiera) lub osobno kwalifikować renderery. Nie udawać, że samo pobranie nowego modelu zapewnia poprawny chat / thinking / tools.

### F14 — P2: konfiguracja nie daje dostępu do ważnych decyzji o pamięci i kosztach

`LlamaConfig` ma tylko batch, context i useGPU. Łączy `n_batch` z `n_ubatch`, nie pozwala określić liczby host threads, stopnia offload, typu KV, SWA policy, load mode ani instrumentacji. W API C te możliwości już istnieją.

Context alokowany jest z pełnym `config.maxTokenCount`, dopiero później logiczny limit zostaje ograniczony do `min(n_ctx_train, config.maxTokenCount)`. Przy konfiguracji ponad limitem modelu można płacić pamięcią za nieużywany context. C może też normalizować `n_ctx` i `n_batch`; wrapper nie raportuje efektywnych wartości ani realnego offload.

Domyślne 8192 wybrane tylko z fizycznego RAM nie uwzględnia wielkości modelu, architektury KV, presji pamięci, limitu procesu iOS ani wielu jednocześnie załadowanych ownerów. Dla klasycznego KV koszt wynosi w przybliżeniu `n_ctx × suma_po_warstwach(n_embd_k_gqa × bytes_K + n_embd_v_gqa × bytes_V)`. Dla SWA i modeli recurrent/hybrid ten wzór nie wystarcza. C API `llama_model_size` podaje model, nie całkowity footprint model + KV + compute + aplikacja.

`useGPU=false` ustawia tylko `n_gpu_layers=0`, pozostawiając `offload_kqv=true` i `op_offload=true`. Jest to wyłączenie offload warstw, nie pełna deklaracja CPU-only. Rozdzielić te pojęcia i zweryfikować logami backend assignments.

Naprawa: niewielka typowana konfiguracja z niezależnym batch / ubatch, threads, policy GPU, KV i diagnostyką efektywnych parametrów. Dodać preflight budżetu pamięci, ograniczenie liczby równocześnie resident modeli i tryb reagowania na memory pressure / thermal state. Nie włączać mlock domyślnie na telefonie.

### F15 — P2: lifecycle backendu jest procesowy, a wrapper zwalnia go na każdy model

`Llama.init` wywołuje `llama_backend_init`, a jego `deinit` — `llama_backend_free`. Publiczny `LlamaBackend` pozwala dodatkowo zrobić shutdown niezależnie. Header mówi „once at start/end of program”. W przypiętym kodzie free nie niszczy Metal device registry; zwalnia globalne tablice quantization IQ. Nie ma podstaw, żeby twierdzić, że zamknięcie jednego Q4_K_M ownera zawsze psuje drugi. Dla współistniejących modeli / quantization, zwłaszcza IQ, własność jest jednak źle umiejscowiona i brak koordynacji.

Naprawa: inicjalizacja raz per proces i kontrolowane shutdown po wszystkich context/model/adapter, ewentualnie reference-counted runtime owner. W Swift `deinit` actora wykonuje się przed automatycznym zwolnieniem jego właściwości; nie używać globalnego free jako zamiennika zwalniania poszczególnych zasobów.

### F16 — P2/P3: częściowe kontrakty stanu, embeddings i raw C params

`embeddings(at:)` i pooled embeddings poza RANK używają `llama_model_n_embd`, a przypięty runtime alokuje wyjście według `n_embd_out`. Dla wielu modeli wymiary są równe, lecz nie jest to uniwersalny kontrakt (projekcje wyjścia). Użyć `llama_model_n_embd_out`, dla input embeddings osobno `n_embd_inp`. To pomocnicze API, nie bieżąca text generation.

`saveState` zamienia 0 bytes na pusty Data zamiast błędu; `loadState` uznaje dowolny dodatni wynik za sukces bez określenia polityki pełnego odczytu. `loadStateForSequence` nie używa argumentu `seqId` (API C potrzebuje destination i bytes). Stan C nie zawiera Swift transcript i historii kar samplera. Dane nie powinny być ładowane jako arbitralny, niesprawdzony format od użytkownika.

Publiczne initializery przyjmują `llama_model_params`, `llama_context_params` i quantize params zawierające pointers / callbacks. Nie wystawiają bezpośrednio osobnego `OpaquePointer` jako metody, ale pozwalają callerowi przekazać raw pointer members o niezarządzanym lifetime. Nie spełnia to w pełni reguły bezpiecznego Swift API. Niskopoziomowe klasy nie są thread-safe; actor chroni wysokopoziomową inferencję, nie dowolne równoległe użycie `LlamaContext` i `LlamaSampler`.

### F17 — P2/P3: synchronizacja, warmup i metrics wymagają właściwego pomiaru

`LlamaContext.decode` zawsze wykonuje `llama_synchronize`. Getter logits już synchronizuje, a następny `llama_sampler_sample` korzysta z synchronizujących getterów. Przy prefill z kilkoma batchami jawna synchronizacja może zmniejszyć możliwość nakładania pracy; przy pojedynczym tokenie często i tak trzeba czekać na poprzednie logits.

Nie usuwać synchronizacji na ślepo: błędy asynchroniczne, ownership buforów i prawidłowość metryk muszą pozostać sprawdzone. Benchmark ma wariant bez dodatkowego sync i granice pomiaru nadal synchronizowane. To kandydat do pomiaru, nie z góry gwarantowany speedup.

`generateNextToken` najpierw sampluje token, następnie dekoduje go do kolejnych logits, a dopiero potem oddaje tekst konsumentowi. C++ simple example emituje wybrany token przed jego kolejnym decode. Obecna kolejność dokłada jeden decode do czasu widocznego pierwszego tokenu, a przy limicie odpowiedzi może wykonywać decode, którego wyniki nie zostaną użyte w tej generacji. Można wprowadzić stan pending token i przesunąć decode na następną iterację, ale trzeba zachować spójny cache, EOG, cancellation i kontynuację rozmowy. Nie zmierzono osobno zysku tej zmiany; jest kandydatem szczególnie dla voice / TTFT.

`setWarmup` jest już deprecated w b10964. Prewarm aplikacji rzeczywiście ładuje i przetwarza transcript, więc pomaga TTFT, ale pusty transcript tylko ładuje model/context; nie wykonuje rzeczywistego warmup graph. Referencyjne warmup trzeba robić przez decode, a potem przywrócić / wyczyścić stan i samplery. Warmup musi być oddzielony od request timing i energy.

`no_perf` ma w C domyślnie true; wrapper go nie zmienia. `performanceData` jest dostępne, ale nie zapewnia pełnych pomiarów wyłączonych w context. `tokensPerSecond` emitera to output / czas obejmujący prefill i emitowanie, czyli end-to-end, nie czysta szybkość decode. Input tokens to pełny prompt, cached token count jest niemierzony; zero nie znaczy brak cache. Metadane wyraźnie odróżniają nieznany reasoning split, co jest dobre.

Per-token wysyłanie całego narastającego rawOutput i utrzymywanie kilku kopii tekstu może rosnąć ponadliniowo przez COW oraz koszt transcript snapshots. Istniejący synthetic test 1000 tokenów trwał ok. 41 ms na tym hoście; sam ten narzut nie tłumaczy wolnej inferencji. Dłuższe próby są w BENCHMARKS. Można emitować metryki okresowo i końcowy raw output raz, ale replay i przerwanie odpowiedzi muszą zachować poprawność.

### F18 — P3: stałe bufory metadanych i niepełne error handling

`description` używa 1024 bajtów i fatalError; metadata getters — 512 bajtów, split helpers — 1024. API string getterów podaje wymaganą długość, często jak snprintf. Wrapper nie ponawia, więc może zwracać skrócone metadata / template strings / paths. `detokenize` poprawnie zwiększa bufor, ale konwertuje przez NUL zamiast jawnej zwróconej długości: `abc\0def` daje `abc`. To samo C-string podejście gubi NUL w `piece` i C chat message content, choć dla normalnego chatu NUL jest rzadki.

`save`, `removeAllLoraAdapters`, default quantize params i status quantization nie tworzą jednolitego throws API. `perfDataDescription` zawsze zwraca pusty String. `attachAutoThreadpool(nil,nil)` odłącza explicit pool i korzysta z fallbacku ggml, nie tworzy jawnego zarządzanego poola. Poprawić nazwy, dynamiczne bufory i opis ownership zamiast poszerzać fasadę przypadkowymi one-linerami.

## Co jest zrobione dobrze

- Posiadane model/context/batch/sampler mają klasy i deinit; context utrzymuje model. Typowe wskaźniki inferencji pozostają wewnętrzne.
- Actor `Llama` serializuje C na bieżącej ścieżce; lifecycle runtime ma osobny mutex, busy guard i oczekiwanie na warmup/response/count przed unload. Nie trzyma locka podczas ładowania modelu ani cancellation handlers.
- `llama_sampler_sample` już wykonuje accept; wrapper nie robi podwójnego accept w pętli. `isEogToken` używa poprawnego vocab i obsługuje EOS/EOT zamiast tylko jednego tokenu.
- Prompt prosi o logits tylko na końcu. Dokładnie pełny ostatni batch nie jest dekodowany drugi raz jako pusty. Jednotokenowy batch generacji jest prawidłowy dla standardowego autoregressive decoding.
- Cache porównuje token IDs, a nie same stringi. Suffix removal sprawdza bool; modele recurrent/hybrid mogą przejść na pełne przetworzenie. Krótszy prompt odświeża ostatnie logits przez ponowny decode.
- Identyczny prompt bez nowych tokenów nie wymaga ponownego decode, gdy context/logits są spójne: próba wykazała 128256 dostępnych logits przed i po oraz dokładną równość. Nie należy zgłaszać tego jako błędu tylko dlatego, że suffix jest pusty.
- Aktualne domyślne model params `n_gpu_layers=-1` zapewniają pełny offload wspieranych warstw. Brak ręcznego `999` jest prawidłowy w tej wersji. `flash_attn_type=AUTO`, `op_offload=true`, `offload_kqv=true` są już domyślnie aktywne.
- Framework Apple ma Metal, embedded shaders i Accelerate/BLAS; upstream buduje go z OpenMP OFF i GGML_NATIVE OFF. Te flagi są rozsądne dla przenośnej dystrybucji; GGML_NATIVE OFF nie znaczy „bez ARM NEON”. Nie widać uzasadnienia do losowej zmiany flags w projekcie.
- Manifest spójnie deklaruje pin, checksum ZIP i revision; wszystkie trzy lokalne `llama.h` są identyczne z nagłówkiem przypiętej rewizji. Simulator setup zachowuje oficjalne device/Mac slices i buduje simulator z tej samej rewizji. W audycie nie pobrano ponownie ZIP dla weryfikacji checksum binariów. Lokalny framework jest zaufanym artifact path, którego checksum nie jest ponownie weryfikowany przy każdym resolve.
- Tools są opt-in dla jednego zweryfikowanego modelu, z kontrolą enabled names i całego batcha arguments przed wykonaniem. Wykonanie narzędzi należy do Apple session, nie zdublowanej pętli wrappera. Nieobsługiwane capability są odrzucane jawnie.
- Nie znaleziono prompt logging w bieżącej inferencji. Logger C jest ograniczonym buforem diagnostics; globalne log markers mogą mieszać komunikaty niezależnych modeli i wymagają ostrożności przy przypisywaniu błędów do requesta.

## Ocena pozostałych metod i kontraktów

Pełne rozliczenie deklaracji jest w METHODS.md. Poniższe grupy opisują również cienkie delegacje, których nie należy traktować jako optymalizacji.

| Obszar | Ocena po porównaniu z C++ |
| --- | --- |
| Model load single / splits, free | Single load i deinit właściwe; sprawdzić pustą listę splits, failed strdup i raw params lifetime. Jeśli vocab kiedykolwiek byłby nil po udanym model load, guard nie zwalnia modelu; obecny upstream zawsze zwraca adres vocab. |
| Model size / parameters / encoder-decoder / recurrent / diffusion | Delegacje odpowiadają C. Wysoki poziom nie obsługuje jednak encode-decoder, encoder-only, diffusion ani mtmd tylko przez fakt, że getters istnieją. Dla nieobsługiwanego typu jawnie odmówić inferencji. |
| Vocab text / score / attrs / special tokens / FIM | Czyste delegacje poprawne, z wyjątkiem EOS i BOS. `string(from:)` to reprezentacja słownika, nie zamiennik token_to_piece do wyjścia. Invalid token IDs wymagają kontraktu; C może asertować. |
| Templates default / named / builtins | Alokacje C strings zwalniane po call; retries bufora są poprawne dla dodatniego wyniku. Brak jawnej obsługi -1 / brakującego named template. Builtins list ograniczona maxCount, bez query pełnego count; ujemne maxCount nie jest walidowane. F13, F18. |
| Context creation / getters / pooling / thread setters | Właściwe symbole C i model retention; wartości muszą odzwierciedlać znormalizowany context. Walidować dodatnie threads w bezpiecznym API. |
| Decode / encode | Kody błędów i diagnostics raportowane; encode status nie ma identycznej semantyki wszystkich decode statusów, wspólny opis może być mylący. F11, F17. |
| Logits getters | Bezpieczna kopia n_vocab floats, brak kopii całych logits w normalnej generacji Swift. Getter C obsługuje sync i invalid index przez null. |
| Embeddings / RANK pooling | Dobra kopia i specjalne n_cls_out dla RANK; konieczna poprawka n_embd_out dla pozostałych. Sam toggle embeddings nie tworzy embedding engine / pooling pipeline. |
| Abort callback bridge | Pass-retained box i C thunk odpowiadają lifetime callbacka przy serialnym użyciu. Closure nie jest Sendable i może być wywoływana z backend thread; nie dotykać UI/actor state bez synchronizacji. Przed zwolnieniem boxa odłączyć callback i poczekać na pracę context. |
| LoRA / cvec apply / clear | Prawidłowe C function i kontrola status dla apply/clear cvec. LoRA ownership wymaga całościowej poprawki F09; sprawdzać zgodność modelu i vector dimensions. |
| Whole-state / sequence-state / files | Prawidłowe API C, rozmiary Data i token pointer w scope; F16 dla pełnego kontraktu, error surface, nieużywanego source seqId i capacity validation. |
| Performance / threadpool | Właściwe funkcje, ale wyłączona instrumentacja, pusty opis samplera i nieprecyzyjne „auto threadpool”. F17–F18. |
| Memory clear/rm/cp/keep/add/div/min/max/canShift | Dokładne mapowanie API memory; borrowed handle nie posiada pamięci. Przechowywać owner context, gdy handle ma uciekać; nie robić context shifting dla recurrent bez canShift. Funkcje C tolerują null memory, ale to nie nadaje zdolności encoderowi. Divider powinien być >1. |
| Sampler name/count/get/reset/clone/free | Prawidłowe delegacje i ownership adoptowanego clone. `name(at:-1)` może zwracać nazwę samego chain zgodnie z C. Remove ma F09; sample wymaga poprawnych logits i selectora. |
| Runtime factory / warmup / prepare / respond / stop | W bieżącej aplikacji dobre rozdzielenie ownership, serialne engine calls i reset after error. Warmup może pozostawiać completed Task w state do kolejnej odpowiedzi/stop; nie znaleziono z tego błędu generacji. Backpressure Apple channel może ograniczać pętlę, więc mierzyć osobno C i session. |
| LanguageModel / executor configuration | Dostępność SDK/OS i odrzucanie unsupported capability poprawne. Configuration equality po UUID model ownera ma sens. Validation batch/context tylko w nowym executorze; legacy nadal może dostać zero lub overflow. ContextSizeError raportuje konfigurację, nie faktyczny limit train, tokenCount=0 jest placeholderem. |
| Transcript mapper | Zachowuje raw reasoning/answer, scala assistant turn i tool outputs; po F02 raw nie jest naprawdę lossless. Sampling map ma F06 i ignoruje model sampling metadata. Nil seed staje się 0; aplikacja jawnie przekazuje losowy seed, więc nie oznacza deterministycznego defaultu Enclave. |
| Tool profile / JSON renderer / parser / schema validation | Hash sprawdzany podczas load; koszt pełnego czytania pliku może zwiększać cold load. Parser jest ograniczony, bez eval, z depth limit i walidacją batcha; obsługuje kwalifikowany Pythonic protocol, nie każdy format tools. Schema validator obsługuje podzbiór JSON Schema, np. nie pełne allOf/oneOf/format/multipleOf; unsupported constraint nie może udawać pełnej walidacji dowolnego schematu. |
| Response emitter / reasoning parser | Odpowiednio odrębne entry IDs dla kontynuacji tools i jawne unknown reasoning token count. Obsługiwane są konkretne delimitery think/Gemma; Harmony lub inne protokoły potrzebują własnego parsera. Timing i koszty tekstu F17. |
| Typed schema recorder / JSON grammar generator | Przeczytano wszystkie overloads primitive/container/optional/nested/super. Nie są pełnym reflection nad Codable ani konwerterem upstream; zbiorczo ograniczenia F12. |
| Logger / errors / cancellation helper | Locks i statyczny sink utrzymują C userdata. CONT diagnostics dziedziczy last level; Unified Logging forward używa oryginalnego level, więc może nie zachować severity. Error text jest przydatny; cancelAndWait świadomie ignoruje error taska i nie powinien zastępować propagacji błędu generacji. |

## Nowe i niewykorzystane mechanizmy

Rozróżnienie: funkcje istniejące już w przypiętym runtime vs zmiany rzeczywiście dodane później. Nie każdy nowy sampler daje lepszą jakość i nie każda funkcja serwera należy do biblioteki C.

| Mechanizm | Stan | Wartość / decyzja |
| --- | --- | --- |
| Flash Attention | Już AUTO w b10964 | Już korzystamy. Benchmarkować AUTO / enabled / disabled dla konkretnej architektury; nie obiecywać speedupu z „włączenia” czegoś już aktywnego. |
| K/V cache quantization | Dostępne w b10964 przez type_k/type_v | Q8_0 jest sensownym pierwszym eksperymentem pamięci; Q4 wymaga mocniejszych quality checks. V quantization wymaga właściwego FA i zgodnych dimensions. Zysk pamięci nie jest dowodem zysku szybkości. |
| Niezależny n_batch / n_ubatch | Dostępne teraz | Duży logical batch może mieć mniejszy physical microbatch. Najważniejsze dla szczytowej pamięci prefill i długich promptów; mierzyć TTFT, footprint i cancellation latency. |
| SWA compact cache | Dostępne; C default swa_full=true, common default false | Możliwy spadek pamięci dla Gemma/SWA. Trzeba skorygować cache reuse względem zachowanego pos_min i sprawdzić jakość; nie zmieniać jako drop-in flag. |
| Sampling metadata modelu | `common_init_sampler_from_model` w b10964 | Upstream potrafi odczytać modelowe top-k/top-p/min-p/temp/penalties. Wrapper stale narzuca własny profil wszystkim modelom. Preferować defaults modelu / jawny profil, a potem świadome overrides użytkownika. |
| Greedy selector | Dostępny teraz | Po potrzebnych transforms użyć init_greedy; nie sortować top-p przed deterministycznym wyborem, jeśli nie jest to zamierzone. |
| min-p, typical-p, DRY, XTC, top-n-sigma, adaptive-p, Mirostat, dynamic temperature | Dostępne w b10964 | Brak wrapper API. Priorytet: modelowe top-k/min-p i polityka repeat. Pozostałe tylko gdy quality evaluation daje korzyść. Więcej samplerów nie gwarantuje najlepszej konfiguracji. |
| Grammar rejection sampling / lazy grammar | `common/sampling.cpp` | Można unikać filtrowania całego vocab grammar przy każdym tokenie; lazy triggers przy reasoning/tools. Wymaga poprawnych stanów samplerów i bridge do common lub świadomej portowanej logiki. |
| Backend sampling | Experimental params.samplers / llama_set_sampler | Potwierdzono działanie chain na Metal: attach=true i dostępny token wybrany przez backend. W osobnej próbie Llama 3.2 było jednak około 28% wolniejsze (118,8 vs 166,0 tok/s). Nie włączać automatycznie. Obsługa operacji poszczególnych samplerów, fallback, koszty grafu i grammar/reasoning wymagają kwalifikacji. |
| Recurrent snapshots | n_rs_seq, experimental | Ograniczony rollback może pomóc hybrid models po suffix edit. Obecny fallback full reprocessing jest bezpieczny, ale kosztowny. Nie traktować snapshots jako dowolnego KV trim. |
| Prefix state snapshots na granicy turn | Dostępne state seq APIs | Alternatywa dla recurrent cache, również szybszy powrót do poprzedniego promptu. Bilans pamięci i czas serialize; snapshot musi uwzględniać Swift tokens i sampling semantics. |
| Speculative decoding / MTP / EAGLE-3 / DFlash / n-gram | Upstream common/speculative i examples/server | Nie ma tego w pętli wrappera. Większy projekt, dodatkowe wagi / stany / acceptance / rollback / wydatki energii. Najpierw większe modele na Macu; małe 1B na iPhone nie mają automatycznej korzyści. |
| Vision/audio mtmd | Framework ma mtmd headers/binary | Wrapper text-only nie używa multimodal projector / input pipeline. To nowa capability, nie darmowa optymalizacja tekstu. |
| load mode / lazy mode | Już w model params b10964 | AUTO jest rozsądną bazą na Apple unified memory. mmap/direct I/O/mlock/lazy MoE należy mierzyć cold load, pressure, resident pages; nie przenosić porad CUDA 1:1. |
| Extended batch API | Dodane między pin a sprawdzonym HEAD | `llama_batch_ext_init/free/clear`, add token/embedding, set positions/output flags, `llama_process`. Bezpieczniejszy fundament kolejnego wrappera, obsługuje token/embedding/state. W HEAD pozostaje TODO dla getterów output; nie jest automatycznym speedupem ani obecnym symbolem frameworka. |
| LoRA z FILE pointer i causal attention getter | Dodane po pin | Zastosowania embedded GGUF / introspection; mały priorytet wobec błędów lifetime i tokenization. |

Porównanie obu `include/llama.h` pokazuje 103 dodane i 2 usunięte linie. Model/context params w tym diffie nie zostały zastąpione nowym uniwersalnym profilem „Apple optimal”. Nowe defaulty już używane w b10964 nie powinny być prezentowane jako brakujące funkcje.

## Zalecana kolejność prac

1. **Poprawność tokenów i pamięci:** F01–F05, F08, F09–F10. Byte API, dynamic buffers, poprawny BOS, batch ownership i walidacja. Dodać testy odtwarzające dzisiejsze awarie oraz multilingual generation; wykonać ASan przy zmianach unsafe Swift.
2. **Sampling i modele:** F06, F12–F13. Prawidłowa kolejność, jawny błąd grammar, wzorce promptów względem upstream, profile modelu. Ocenić jakość na stałych seedach i niezależnie czat / reasoning / tools.
3. **Minimalne strojenie Apple:** F07, F14, F17. Rozdzielić threads/batch/ubatch, raportować efektywne params, zachować AUTO FA i pełny GPU jako bazę. Zmierzyć Q8 cache i compact SWA tylko na pasujących modelach.
4. **Lifecycle:** F11, F15–F16. Posprzątać legacy, global backend owner i state contracts. Dla shipped hybrid modelu porównać current suffix fallback z turn snapshots.
5. **Dopiero następnie większe funkcje:** Jinja bridge, speculative decoding i multimodal według uzasadnionego zakresu produktu.

Nie proponuję jednego magicznego zestawu parametrów dla wszystkich GGUF i urządzeń. Dobre docelowe API to kilka jasno opisanych policy i pomiar najlepszej konfiguracji model × urządzenie × context. Upstream CLI defaults są użytecznym punktem odniesienia, nie dowodem optymalności dla aplikacji mobilnej.

## Walidacja i ograniczenia

Dokładne wyniki, polecenia i artefakty są w [BENCHMARKS.md](BENCHMARKS.md) oraz [evidence](evidence/). Izolowane crash probes celowo kończą proces i są wyłączone w zwykłym zestawie testów; nie wolno uruchamiać ich jako oczekiwanego zielonego regression suite.

Nie zmieniono kodu produkcyjnego ani nie wykonano release/TestFlight. Brak pomiarów na fizycznym iPhonie, Intel Mac i nowszych układach M-series. Mac benchmark ma ciepłe wagi, jednego ownera, krótkie serie i określone GGUF; nie dowodzi battery/energy, jakości po quantization KV, stabilności thermals przez 15 minut ani peak memory na iOS. Statyczna poprawność i testy na symulatorze nie zastępują tych pomiarów.

Przed wdrożeniem optymalizacji: 3–5 powtórzeń warm i cold, realne prompty około 64/512/2048/8192 tokenów, przedłużona generacja, normalna aplikacja z transcript/UI, first token / prefill / decode oddzielnie, peak footprint, energy, thermal state, memory pressure, cancel w prefill/decode, ponowna generacja po cancel/error i quality checks. Dla nowych GGUF użyć istniejącego `Tools-Model-Update` na Macu i iOS; performance mierzyć fizycznie.
