#!/usr/bin/env python3
"""Build immutable source snapshots, then run sequential, balanced GPU A/B trials."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor


def run(command, *, cwd=None, log=None):
    if log:
        with log.open("w") as stream:
            subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=True)
    else:
        subprocess.run(command, cwd=cwd, check=True)


def build(repo, revision, destination, harness):
    destination.mkdir()
    archive = subprocess.check_output(["git", "archive", revision, "Sources/SwiftLlama"], cwd=repo)
    archive_path = destination / "sources.tar"
    archive_path.write_bytes(archive)
    with tarfile.open(archive_path) as source:
        source.extractall(destination, filter="data")
    archive_path.unlink()
    shutil.copyfile(harness, destination / "Sources/SwiftLlama/GPUComparison.swift")
    (destination / "llama.xcframework").symlink_to(repo / "Artifacts/llama-b10964.xcframework")
    (destination / "Package.swift").write_text('''// swift-tools-version: 6.1
import PackageDescription
let package = Package(name: "GPUComparison", platforms: [.macOS(.v14)], targets: [
    .binaryTarget(name: "llama", path: "llama.xcframework"),
    .executableTarget(name: "GPUComparison", dependencies: ["llama"], path: "Sources/SwiftLlama",
        linkerSettings: [.unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", "@executable_path"])])
])
''')
    run(["swift", "build", "-c", "release"], cwd=destination, log=destination / "build.log")
    product = destination / ".build/release/GPUComparison"
    framework = product.parent / "llama.framework"
    if not framework.exists():
        framework.symlink_to(repo / "Artifacts/llama-b10964.xcframework/macos-arm64_x86_64/llama.framework")
    return product


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", default="b7f9e68250aff79d11232e8905c7f2db41a4bc18")
    parser.add_argument("--after", default="bcb9e0518a9f62031d7093d8ec0330aa2b8bc7fe")
    parser.add_argument("--llama-model", type=Path, required=True)
    parser.add_argument("--lfm-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--repetitions-per-block", type=int, default=4)
    args = parser.parse_args()
    if args.blocks < 2 or args.blocks % 2 or args.repetitions_per_block < 1:
        parser.error("Use an even block count >= 2 and positive repetitions")
    repo = Path(__file__).resolve().parent.parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    temporary = Path(tempfile.mkdtemp(prefix="enclave-gpu-ab-"))
    revisions = {name: subprocess.check_output(["git", "rev-parse", value], cwd=repo, text=True).strip()
                 for name, value in [("before", args.before), ("after", args.after)]}
    models = {"lfm-1.2b": args.lfm_model.resolve(), "llama-1b": args.llama_model.resolve()}
    metadata = {"revisions": revisions, "snapshots": str(temporary),
                "harnessSHA256": hashlib.sha256((repo / "Verification/GPUComparison.swift").read_bytes()).hexdigest(),
                "models": {name: {"path": str(path), "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest()}
                           for name, path in models.items()},
                "blocks": args.blocks, "repetitionsPerBlock": args.repetitions_per_block,
                "llamaRevision": "b29c606e28a01b1bc8c1351026a0fa6e616bf6c4",
                "machine": os.uname().machine,
                "os": subprocess.check_output(["sw_vers"], text=True),
                "swift": subprocess.check_output(["swift", "--version"], text=True)}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Building both immutable revisions in {temporary}", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {name: pool.submit(build, repo, revision, temporary / name, repo / "Verification/GPUComparison.swift")
                   for name, revision in revisions.items()}
        executables = {name: future.result() for name, future in futures.items()}
    for name in revisions:
        shutil.copyfile(temporary / name / "build.log", output / f"build-{name}.log")
    metadata["verifiedUnmodifiedSourceFiles"] = {}
    metadata["runtimeFrameworkSHA256"] = {}
    for name, revision in revisions.items():
        files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", revision, "Sources/SwiftLlama"],
                                        cwd=repo, text=True).splitlines()
        for path in files:
            expected = subprocess.check_output(["git", "show", revision + ":" + path], cwd=repo)
            if (temporary / name / path).read_bytes() != expected:
                raise RuntimeError(f"Source snapshot changed: {name}/{path}")
        metadata["verifiedUnmodifiedSourceFiles"][name] = len(files)
        framework = executables[name].parent / "llama.framework/Versions/Current/llama"
        with framework.open("rb") as stream:
            metadata["runtimeFrameworkSHA256"][name] = hashlib.file_digest(stream, "sha256").hexdigest()
    if len(set(metadata["runtimeFrameworkSHA256"].values())) != 1:
        raise RuntimeError("Linked llama binaries differ")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for label, model in models.items():
        print(f"Preparing common exact-token prompts for {label}", flush=True)
        run([str(executables["after"]), "prepare", str(model), str(output / f"{label}-prompts.json"),
             "/dev/null", revisions["after"], label, "0", "1"], log=output / f"prepare-{label}.log")
    combined = output / "results.jsonl"
    combined.touch()
    for block in range(args.blocks):
        order = ["before", "after"] if block % 2 == 0 else ["after", "before"]
        for label in models if block % 2 == 0 else reversed(models):
            for name in order:
                print(f"Block {block + 1}/{args.blocks}: {label} {name}", flush=True)
                part = output / f"{label}-{name}-{block}.jsonl"
                run([str(executables[name]), "benchmark", str(models[label]), str(output / f"{label}-prompts.json"),
                     str(part), revisions[name], label, str(block), str(args.repetitions_per_block)],
                    log=output / f"{label}-{name}-{block}.log")
                with combined.open("ab") as stream:
                    stream.write(part.read_bytes())
    print(f"Completed. Raw results: {combined}", flush=True)


if __name__ == "__main__":
    main()
