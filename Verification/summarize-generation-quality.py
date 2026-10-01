#!/usr/bin/env python3
"""Check native token/byte agreement and expose complete answers for human review."""
import argparse
import json
from pathlib import Path


def summarize(directory):
    metadata = json.loads((directory / "metadata.json").read_text())
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 72
    expected_scenarios = {"instruction", "arithmetic", "polish", "translation", "summary", "conversation", "story", "unicode", "json"}
    keyed = {}
    counts = {}
    for version, revision in metadata["revisions"].items():
        records = [row for row in rows if row["revision"] == revision]
        assert len(records) == 36
        for row in records:
            key = (version, row["model"], row["sampling"], row["scenario"])
            assert key not in keyed
            keyed[key] = row
            assert row["model"] in metadata["models"]
            assert row["sampling"] in {"greedy", "production"}
            assert row["scenario"] in expected_scenarios
            assert row["endedNaturally"], "Output cap reached; review an incomplete answer separately"
            assert row["text"] and row["generatedTokens"]
            if version == "after":
                assert row["text"] == row["nativeText"], "Streaming damaged the C token bytes"
                assert row["generatedTokens"] == row["referenceTokens"], "Wrapper diverged from independent C generation"
                assert "\ufffd" not in row["text"], "Replacement character in completed answer"
                assert "<|" not in row["text"], "Unexpected raw control-token marker"
        counts[version] = {"answers": len(records),
                           "streamMatchesC": sum(row["text"] == row["nativeText"] for row in records),
                           "naturalEOS": sum(row["endedNaturally"] for row in records),
                           "matchesIndependentC": sum(row["generatedTokens"] == row.get("referenceTokens") for row in records) if version == "after" else None}
    assert set(counts) == {"before", "after"}
    assert len(keyed) == 2 * 2 * 2 * len(expected_scenarios)
    (directory / "checks.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts, indent=2))
    output = ["# Complete generation samples", "", "These are raw synthetic test answers, not a claim that every answer is correct.", ""]
    for model in metadata["models"]:
        for scenario in sorted(expected_scenarios):
            output += [f"## {model}: {scenario}", ""]
            prompt = keyed[("after", model, "production", scenario)]["messages"]
            for message in prompt:
                output += [f"**{message['role']}**: {message['content']}", ""]
            for sampling in ["greedy", "production"]:
                output += [f"### {sampling}", ""]
                for version in ["before", "after"]:
                    row = keyed[(version, model, sampling, scenario)]
                    output += [f"**{version}** ({len(row['generatedTokens'])} tokens; EOS; stream=C: {row['text'] == row['nativeText']}):", "", "````text", row["text"], "````", ""]
                    if row["text"] != row["nativeText"]:
                        output += ["**Exact text represented by sampled C tokens:**", "", "````text", row["nativeText"], "````", ""]
    (directory / "answers.md").write_text("\n".join(output).rstrip() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    summarize(parser.parse_args().directory)
