#!/usr/bin/env python3
"""Extract the complete method list and attach the audit's manually reviewed assessments."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
REVISION = "b29c606e28a01b1bc8c1351026a0fa6e616bf6c4"
HEADER = ROOT / "Reference/llama.cpp/include/llama.h"
header = HEADER.read_text() if HEADER.exists() else ""

DEFAULTS = {
    "Llama": "Actor serializuje core; kontrakty cache/generacji F11 i ustawienia F07/F14/F15/F17.",
    "LlamaBackend": "Delegacja C zgodna; procesowy lifecycle F15 i semantyka threadpool F18.",
    "LlamaBatch": "Posiadany batch ma RAII; wymagane sprawdzanie mode/capacity F10.",
    "LlamaChatMessage": "Swift value type; role i content używane przez renderer F13.",
    "LlamaConfig": "Niepełna, niewalidowana konfiguracja publiczna; F07/F14.",
    "LlamaContext": "Delegacja C zgodna przy serialnym użyciu; szczegóły context/state/embeddings w raporcie.",
    "LlamaContextUsage": "Swift value type do niemutującego accounting.",
    "LlamaError": "Opis błędu i diagnostics; brak zmiany runtime.",
    "LlamaExecutorRuntime": "Sprawdzono mutex, task ownership, busy/stop/reset; poprawny lifecycle nowej ścieżki.",
    "LlamaGrammarConfig": "Przechowuje grammar/root; inicjalizacja grammar musi failować jawnie F12.",
    "LlamaLanguageModel": "Model owner i availability poprawne; raw params/context reporting F14/F16.",
    "LlamaLogger": "Lock i lifetime globalnego sink poprawne; global attribution/CONT severity opisane w raporcie.",
    "LlamaLoraAdapter": "Ownership/lifetime model i adapter wymagają wspólnej poprawki F09.",
    "LlamaMemory": "Zgodna cienka delegacja memory C; borrowed context lifetime i warunki shifting w raporcie.",
    "LlamaModel": "Delegacja model/vocab C zgodna; sprawdzono właściwy wskaźnik i długości wyników.",
    "LlamaRepetitionPenaltyConfig": "Parametry przechowywane bez walidacji; kolejność i zakres history F06.",
    "LlamaResponseEmitter": "Sprawdzono reasoning/tools/metadata; UTF-8 przed emiterem F02 i metryki/koszt F17.",
    "LlamaSampler": "Chain RAII i sample/accept poprawne; kolejność i error surface F06/F12.",
    "LlamaSamplingConfig": "Niewalidowane ranges i wspólny profil modeli; F06/F14.",
    "LlamaService": "Legacy; reentrancy, stream termination, reset i typed response F11/F12.",
    "LlamaToolCalling": "Sprawdzono hash gate, ograniczony parser i schema subset; kwalifikacja modelu w raporcie.",
    "LlamaTranscriptMapper": "Sprawdzono role/turn/raw replay; ograniczenia renderer/sampling F02/F06/F13/F14.",
    "LlamaTypedJSONGrammarBuilder": "Przeczytany overload syntetycznego decoder/generator; best-effort ograniczenia F12.",
    "Task+Extensions": "Cancel i await taska; świadomie pomija jego error, wymaga właściwego call site.",
}

OVERRIDES = {
    ("LlamaModel", "piece"): "BŁĄD potwierdzony: 64-byte buffer/ujemna długość F01, niepełny UTF-8 i NUL F02/F18.",
    ("LlamaModel", "tokenize"): "BŁĄD potwierdzony: capacity zamiast n_ctx_train i ujemny wynik F03; empty ignoruje specials.",
    ("LlamaModel", "shouldAddBos"): "BŁĄD potwierdzony: wymóg BOS ignorowany dla BPE F04.",
    ("LlamaModel", "eosToken"): "BŁĄD potwierdzony: modelPointer zamiast vocabPointer F08.",
    ("LlamaModel", "detokenize"): "Retry size poprawny; string konwertowany do pierwszego NUL zamiast written bytes F18.",
    ("LlamaModel", "applyChatTemplate"): "Zwalnia C strings i poprawnie zwiększa bufor; unsupported/nil/Jinja/model protocol F13/F18.",
    ("LlamaModel", "applyGemma4ChatTemplate"): "Wąski ręczny text renderer; nie pełny Jinja, wymagane golden token tests F13.",
    ("LlamaModel", "renderLFMToolPrompt"): "Świadomy renderer kwalifikowanego LFM profile, nie generyczne tools F13.",
    ("LlamaModel", "metaValue"): "Możliwa truncation >512 bytes, brak retry długości F18.",
    ("LlamaModel", "metaKey"): "Możliwa truncation >512 bytes, brak retry długości F18.",
    ("LlamaModel", "description"): "Stały bufor/fatalError w publicznej metodzie F18.",
    ("LlamaModel", "splitPath"): "Stałe 1024 bytes, ignorowanie required length/status F18.",
    ("LlamaModel", "splitPrefix"): "Waliduje n<=0, nie obsługuje truncation F18.",
    ("LlamaModel", "builtinChatTemplates"): "Delegacja poprawna; maxCount ogranicza wynik, ujemny count niewalidowany F10/F18.",
    ("LlamaModel", "quantizeModel"): "Raw quantize params i UInt32 status wymagają świadomego caller ownership/error handling F16.",
    ("LlamaBatch", "singleSequence"): "BŁĄD potwierdzony: borrowed array, utracona allocation, invalid free F05.",
    ("LlamaBatch", "addToken"): "Pętla aplikacji mieści się w capacity; publiczne API nie waliduje capacity ani trybu F10.",
    ("LlamaBatch", "setLastTokenLogits"): "Bez guard size>0 publiczny call indeksuje -1 F10.",
    ("LlamaBatch", "setEmbedding"): "Zły stride, brak mode/dimension/capacity i brak kompletnego append F10.",
    ("LlamaContext", "failureDescription"): "Przydatne decode statuses; encode nie ma identycznej semantyki wszystkich kodów.",
    ("LlamaContext", "decode"): "Zgodne decode status + diagnostics; zawsze synchronize F17, częściowa pamięć przy abort/error F11.",
    ("LlamaContext", "encode"): "Zgodny encode call i error; dodatkowe sync F17, wysoki poziom nie jest encode-decoder engine.",
    ("LlamaContext", "embeddings"): "Wymiar n_embd zamiast n_embd_out dla części modeli F16.",
    ("LlamaContext", "pooledEmbeddings"): "RANK używa poprawnego n_cls_out; pozostałe powinny n_embd_out F16.",
    ("LlamaContext", "setAbortCallback"): "Bridging poprawne serialnie; callback backend thread, owner lifetime i detach opisane w raporcie.",
    ("LlamaContext", "loadStateForSequence"): "C destination poprawne, source seqId nieużywany; raw C state nie obejmuje Swift/sampler F16.",
    ("LlamaContext", "saveState"): "Pusty Data zamiast błędu; format/lifetime sprawdzone, pełny session contract F16.",
    ("LlamaContext", "loadState"): "Uznaje read>0; nie uzgadnia high-level tokens/sampler, F16.",
    ("LlamaContext", "loadSession"): "Delegacja i token buffer scope poprawne; walidować capacity i state contract F10/F16.",
    ("LlamaContext", "loadSequenceState"): "Delegacja i token buffer scope poprawne; walidować capacity F10/F16.",
    ("LlamaContext", "setWarmup"): "Już deprecated w pin; ręczny decode warmup F17.",
    ("LlamaContext", "attachAutoThreadpool"): "nil pools oznaczają fallback / odłączenie explicit pools; nazwa obiecuje zbyt dużo F18.",
    ("LlamaContext", "performanceData"): "C getter poprawny; context no_perf domyślnie true F17.",
    ("LlamaSampler", "init"): "Penalties po filtrach F06; invalid grammar cicho znika F12.",
    ("LlamaSampler", "sample"): "C sample automatycznie acceptuje; poprawne idx=-1, potrzebne logits i selector.",
    ("LlamaSampler", "accept"): "Delegacja poprawna; komentarz o karmieniu całej grammar promptem błędny F06.",
    ("LlamaSampler", "remove"): "BŁĄD ownership: zwrócony sampler nie jest zwalniany; usunięcie selectora może asertować F09.",
    ("LlamaSampler", "perfDataDescription"): "C print działa, String zawsze pusty F18.",
    ("Llama", "generateNextToken"): "EOG/limit poprawne; mutacja przed udanym decode F11, piece F01/F02; emit po kolejnym decode opóźnia TTFT i może robić nadmiarowy decode przy limicie F17.",
    ("Llama", "processPrompt"): "Batch boundaries/logits poprawne; cache commit przed decode F11, threads i sync F07/F17.",
    ("Llama", "shouldUsePartialOptimization"): "Heurystyka 10 tokens/50% poprawności nie narusza, ale może odrzucać korzystny reuse krótszego prefixu.",
    ("Llama", "optimizedReprocessing"): "Poprawny trim bool/fallback i końcowe logits; recurrent rollback/SWA constraints w raporcie.",
    ("Llama", "contextUsage"): "Niemutująca dokładna tokenizacja aktualnego formatu; odziedzicza F03/F04/F13.",
    ("Llama", "loadStateData"): "Helper testowy przywraca C, nie Swift token/position/sampler state F11/F16.",
    ("Llama", "clearKV"): "Helper testowy czyści C bez processedTokens/position F11.",
    ("Llama", "clear"): "Spójny full reset wysokiego poziomu; data=true zeruje również buffers, nowy batch nie zawsze potrzebny.",
    ("LlamaTranscriptMapper", "sampling"): "Mapowanie jawnych options działa; profile/seed/top-k defaults i greedy/penalties F06/F14.",
    ("LlamaResponseEmitter", "append"): "Token count właściwy na obecnym seam; raw bytes są już utracone F02, narastające metadata F17.",
    ("LlamaResponseEmitter", "metadata"): "Jawne unknown reasoning; tokens/sec obejmuje prefill, cached count niezmierzony F17.",
    ("LlamaService", "extractLikelyJSON"): "Nawiasy w strings/escapes i koszt powtarzanego prefix scan F12.",
    ("LlamaLanguageModel", "prewarm"): "Awaited load/prefill i cancel lifecycle poprawne; pusty transcript nie rozgrzewa graph F17.",
    ("LlamaTypedJSONGrammarBuilder", "emitRule"): "Nie wymusza required/unique keys; kolizje sanitize/order i best-effort schema F12.",
    ("LlamaTypedJSONGrammarBuilder", "sanitize"): "Kolizje nazw (np. a-b/a_b), unicode rule identifiers, wymagane unikalne IDs F12.",
    ("LlamaToolCalling", "validate"): "Hash profile lub ograniczony JSON Schema subset; zweryfikowany scope, nie pełna dowolna schema.",
}

def mask_literals(source):
    # Mask comments and Swift literals while preserving offsets; interpolation is not a C call in this inventory.
    pattern = r'//[^\n]*|/\*[\s\S]*?\*/|#+"""[\s\S]*?"""#+|"""[\s\S]*?"""|#+"[^\n]*?"#+|"(?:\\.|[^"\\])*"'
    return re.sub(pattern, lambda m: re.sub(r'[^\n]', ' ', m.group()), source)

declaration = re.compile(r'(?m)^\s*(?:(?:public|private|internal|fileprivate|static|mutating|final|nonisolated|override|convenience|required)\s+)*(?P<kind>func\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*|[=<>!+*/&|~^-]+)|init\??(?=\s*\()|deinit\b)')
lines = ["# Ocena każdej metody SwiftLlama", "",
         "Baza: `b7f9e68`. Lista deklaracji jest wyciągnięta mechanicznie; oceny i problemy pochodzą z przeczytania kodu i porównania z C++ opisanego w [raporcie](README.md). Wiersze z tym samym kontraktem mają wspólną ocenę, ale każdy overload ma osobny numer linii. C symbols to bezpośrednie wywołania w body (łącznie z nested helperami), nie kompletny call graph. Computed properties i enum cases dodatkowo omówiono w raporcie.", ""]
total = 0
for path in sorted((ROOT / "Sources/SwiftLlama").glob("*.swift")):
    source = path.read_text()
    masked = mask_literals(source)
    matches = list(declaration.finditer(masked))
    lines += [f"## {path.name}", "", "| Deklaracja | Bezpośrednie C API | Ocena |", "| --- | --- | --- |"]
    for index, match in enumerate(matches):
        total += 1
        line = source.count('\n', 0, match.start('kind')) + 1
        name = match.group('name') or ('deinit' if match.group('kind').startswith('deinit') else 'init')
        opening = masked.find('{', match.end())
        next_declaration = matches[index + 1].start('kind') if index + 1 < len(matches) else len(masked)
        between = masked[match.end():opening]
        if next_declaration < opening or re.search(r'(?m)^\s*(?:\}|extension\b|(?:final\s+)?(?:class|actor|struct|protocol)\b)', between):
            opening = -1
        signature_end = opening if opening >= 0 else source.find('\n', match.end())
        signature = re.sub(r'\s+', ' ', source[match.start('kind'):signature_end]).strip()
        depth, end = 1, opening + 1
        while end < len(masked) and depth:
            depth += (masked[end] == '{') - (masked[end] == '}')
            end += 1
        body = masked[opening:end] if opening >= 0 else ''
        symbols = sorted(set(re.findall(r'\b(?:llama|ggml)_[A-Za-z0-9_]+(?=\s*\()', body)))
        links = []
        for symbol in symbols:
            located = re.search(r'\b' + re.escape(symbol) + r'\s*\(', header)
            fragment = f"#L{header.count(chr(10), 0, located.start()) + 1}" if located else ''
            links.append(f"[`{symbol}`](https://github.com/ggml-org/llama.cpp/blob/{REVISION}/include/llama.h{fragment})")
        assessment = OVERRIDES.get((path.stem, name), DEFAULTS[path.stem])
        link = f"[`{signature}`](../Sources/SwiftLlama/{path.name}#L{line})"
        lines.append(f"| {link} (L{line}) | {', '.join(links) or 'Swift logic / helper'} | {assessment} |")
    lines.append('')
lines += [f"Łącznie: **{total} deklaracji** w {len(DEFAULTS)} plikach. Indeks obejmuje kod produkcyjny; opt-in audit probes są osobnym test targetem.", '']
(ROOT / 'Audit/METHODS.md').write_text('\n'.join(lines))
print(f"Indexed {total} method declarations")
