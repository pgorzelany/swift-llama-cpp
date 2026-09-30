#!/usr/bin/env python3
"""Summarize completed warm benchmark records; retain every raw measurement."""
import csv
import json
import statistics
import sys
from pathlib import Path

for filename in sys.argv[1:]:
    path = Path(filename)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    groups = sorted({(row["profile"], row["requestedPrompt"]) for row in rows})
    print(f"\n{path.name}")
    print("| Profile | Prompt | Prefill tok/s | Generation tok/s | TTFT s | Load s | TG min–max |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for profile, prompt in groups:
        data = [r for r in rows if r["profile"] == profile and r["requestedPrompt"] == prompt and r["repetition"] > 0]
        assert sorted(r["repetition"] for r in data) == [1, 2, 3], (profile, prompt, "incomplete")
        assert all(r["actualPrompt"] == prompt and r["outputTokens"] == 32 for r in data)
        pp = [r["actualPrompt"] / r["prefillSeconds"] for r in data]
        tg = [r["outputTokens"] / r["generationSeconds"] for r in data]
        ttft = [r["prefillSeconds"] + r["firstTokenSeconds"] for r in data]
        print(f"| {profile} | {prompt} | {statistics.median(pp):.1f} | {statistics.median(tg):.1f} | {statistics.median(ttft):.3f} | {data[0]['loadSeconds']:.3f} | {min(tg):.1f}–{max(tg):.1f} |")
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
