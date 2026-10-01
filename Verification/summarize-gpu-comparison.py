#!/usr/bin/env python3
"""Validate GPU A/B workload identity and summarize all measured trials."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median


def summarize(directory):
    metadata = json.loads((directory / "metadata.json").read_text())
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
    revisions = metadata["revisions"]
    expected = metadata["blocks"] * metadata["repetitionsPerBlock"]
    assert len(rows) == 18 * 2 * metadata["blocks"] * (metadata["repetitionsPerBlock"] + 1)
    groups = defaultdict(lambda: defaultdict(list))
    warmup_keys = set()
    warmups = 0
    for row in rows:
        assert row["actualPrompt"] == row["requestedPrompt"] - int(row.get("omittedLeadingToken") is not None)
        if row.get("omittedLeadingToken") is not None:
            assert row["omittedLeadingToken"] == {"llama-1b": 128000, "lfm-1.2b": 1}[row["model"]]
            assert row["revision"] == revisions["before"] and row["sampling"] != "greedy-controlled"
        assert row["outputTokens"] == len(row["generatedTokens"]) == 32
        assert row["threads"] == 1 and row["batch"] == row["microBatch"] == 1024
        assert row["thermalState"] == 0, "A trial had a non-nominal thermal state"
        assert row["generationSeconds"] > row["firstTokenSeconds"] > 0
        assert row["prefillSeconds"] > 0
        version = next(name for name, revision in revisions.items() if revision == row["revision"])
        if row["repetition"] == 0:
            key = (row["model"], row["sampling"], row["requestedPrompt"], version, row["block"])
            assert key not in warmup_keys, "Duplicate warmup"
            warmup_keys.add(key)
            warmups += 1
            continue
        key = (row["model"], row["sampling"], row["requestedPrompt"])
        groups[key][version].append(row)
    assert len(groups) == 18, "Missing workloads"
    assert warmups == 18 * 2 * metadata["blocks"]
    print(f"{len(rows)} trials: {warmups} warmups, {len(rows) - warmups} measured; {expected} per revision/workload.\n")
    print("| Model | Sampler | Prompt | TG before | TG after | TG change | PP change | TTFT before ms | TTFT after ms | TG change by block | Same output pairs |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    details = []
    for (model, sampling, length), versions in sorted(groups.items()):
        before = sorted(versions["before"], key=lambda row: row["repetition"])
        after = sorted(versions["after"], key=lambda row: row["repetition"])
        for records in [before, after]:
            assert len(records) == expected
            assert [row["repetition"] for row in records] == list(range(1, expected + 1))
        assert len({row["promptHash"] for row in before + after}) == 1
        equal = sum(old["generatedTokens"] == new["generatedTokens"] for old, new in zip(before, after))
        if sampling == "greedy-controlled":
            assert equal == expected, "Greedy output changed; cannot treat as identical token workload"
        def tg(records):
            return median(row["outputTokens"] / row["generationSeconds"] for row in records)
        def pp(records):
            return median(row["actualPrompt"] / row["prefillSeconds"] for row in records)
        def ttft(records):
            return median(1000 * (row["prefillSeconds"] + row["firstTokenSeconds"]) for row in records)
        block_changes = []
        for block in range(metadata["blocks"]):
            old = [row for row in before if row["block"] == block]
            new = [row for row in after if row["block"] == block]
            assert len(old) == len(new) == metadata["repetitionsPerBlock"]
            block_changes.append(100 * (tg(new) / tg(old) - 1))
        delta = 100 * (tg(after) / tg(before) - 1)
        print(f"| {model} | {sampling} | {length} | {tg(before):.1f} | {tg(after):.1f} | {delta:+.1f}% | "
              f"{100 * (pp(after) / pp(before) - 1):+.1f}% | {ttft(before):.1f} | {ttft(after):.1f} | "
              + ", ".join(f"{value:+.1f}%" for value in block_changes) + f" | {equal}/{expected} |")
        details.append({"model": model, "sampling": sampling, "prompt": length,
                        "tgBefore": tg(before), "tgAfter": tg(after), "tgChangePercent": delta,
                        "ppBefore": pp(before), "ppAfter": pp(after),
                        "ttftBeforeMs": ttft(before), "ttftAfterMs": ttft(after),
                        "blockChangesPercent": block_changes, "sameOutputPairs": equal,
                        "tgBeforeRange": [min(r["outputTokens"] / r["generationSeconds"] for r in before), max(r["outputTokens"] / r["generationSeconds"] for r in before)],
                        "tgAfterRange": [min(r["outputTokens"] / r["generationSeconds"] for r in after), max(r["outputTokens"] / r["generationSeconds"] for r in after)]})
    for model in metadata["models"]:
        for length in [64, 512, 2048]:
            controlled = sorted(groups[(model, "greedy-controlled", length)]["after"], key=lambda row: row["repetition"])
            native = sorted(groups[(model, "greedy-native", length)]["after"], key=lambda row: row["repetition"])
            assert all(a["generatedTokens"] == b["generatedTokens"] for a, b in zip(controlled, native)), "Replay differs from native prompt"
    (directory / "summary.json").write_text(json.dumps(details, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    summarize(parser.parse_args().directory)
