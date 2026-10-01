#!/usr/bin/env python3
"""Inspect complete native GPU answers and compare streamed text with C bytes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-model", type=Path, required=True)
    parser.add_argument("--lfm-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("gpu_comparison", repo / "Verification/run-gpu-comparison.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    revisions = {"before": "b7f9e68250aff79d11232e8905c7f2db41a4bc18", "after": "bcb9e0518a9f62031d7093d8ec0330aa2b8bc7fe"}
    models = {"llama-1b": args.llama_model.resolve(), "lfm-1.2b": args.lfm_model.resolve()}
    output = args.output.resolve()
    output.mkdir(exist_ok=False, parents=True)
    snapshots = Path(tempfile.mkdtemp(prefix="enclave-generation-quality-"))
    metadata = {"revisions": revisions, "snapshots": str(snapshots),
                "models": {name: {"path": str(path), "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest()}
                           for name, path in models.items()},
                "harnessSHA256": hashlib.sha256((repo / "Verification/GenerationQuality.swift").read_bytes()).hexdigest(),
                "sampling": "temperature 0/0.5, seed 42, default penalties, no grammar",
                "runtime": "b10964", "gpu": True, "batch": 1024, "context": 4096, "outputLimit": 256}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Building immutable wrapper versions", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {name: pool.submit(helper.build, repo, revision, snapshots / name, repo / "Verification/GenerationQuality.swift")
                   for name, revision in revisions.items()}
        executables = {name: result.result() for name, result in futures.items()}
    helper.run(["swift", "build", "-c", "release", "-Xswiftc", "-DQUALITY_FIXED"],
               cwd=snapshots / "after", log=snapshots / "after/build.log")
    for name in revisions:
        shutil.copyfile(snapshots / name / "build.log", output / f"build-{name}.log")
    metadata["verifiedUnmodifiedSourceFiles"] = {}
    metadata["runtimeFrameworkSHA256"] = {}
    metadata["referenceScope"] = "Corrected wrapper only; independent C context/batch/sampler/generation with the same prompt tokens"
    for name, revision in revisions.items():
        files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", revision, "Sources/SwiftLlama"],
                                        cwd=repo, text=True).splitlines()
        for path in files:
            if (snapshots / name / path).read_bytes() != subprocess.check_output(["git", "show", revision + ":" + path], cwd=repo):
                raise RuntimeError(f"Source snapshot changed: {name}/{path}")
        metadata["verifiedUnmodifiedSourceFiles"][name] = len(files)
        with (executables[name].parent / "llama.framework/Versions/Current/llama").open("rb") as stream:
            metadata["runtimeFrameworkSHA256"][name] = hashlib.file_digest(stream, "sha256").hexdigest()
    if len(set(metadata["runtimeFrameworkSHA256"].values())) != 1:
        raise RuntimeError("Linked llama binaries differ")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    combined = output / "results.jsonl"
    combined.touch()
    for label, model in models.items():
        for name, revision in revisions.items():
            print(f"Generating {label}: {name}", flush=True)
            part = output / f"{label}-{name}.jsonl"
            helper.run([str(executables[name]), str(model), str(part), revision, label],
                       log=output / f"{label}-{name}.log")
            with combined.open("ab") as stream:
                stream.write(part.read_bytes())
    print(f"Completed: {combined}", flush=True)


if __name__ == "__main__":
    main()
