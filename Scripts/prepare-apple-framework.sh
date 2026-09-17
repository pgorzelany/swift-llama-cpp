#!/bin/bash
set -euo pipefail

package_root="$(cd "$(dirname "$0")/.." && pwd)"
llama_version="$(sed -n 's/^let llamaVersion = "\(.*\)"/\1/p' "$package_root/Package.swift")"
llama_checksum="$(sed -n 's/^let llamaChecksum = "\(.*\)"/\1/p' "$package_root/Package.swift")"
llama_revision="$(sed -n 's/^let llamaRevision = "\(.*\)"/\1/p' "$package_root/Package.swift")"
[[ -n "$llama_version" && -n "$llama_checksum" && -n "$llama_revision" ]]
destination="$package_root/Artifacts/llama-$llama_version.xcframework"

if [[ -d "$destination" ]]; then
    printf 'Apple framework already prepared: %s\n' "$destination"
    exit 0
fi

for tool in cmake git curl xcrun; do
    command -v "$tool" >/dev/null || { printf 'Required tool missing: %s\n' "$tool" >&2; exit 1; }
done

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT
curl --fail --location --retry 3 \
    "https://github.com/ggml-org/llama.cpp/releases/download/$llama_version/llama-$llama_version-xcframework.zip" \
    --output "$work_dir/release.zip"
printf '%s  %s\n' "$llama_checksum" "$work_dir/release.zip" | shasum -a 256 --check --status
ditto -x -k "$work_dir/release.zip" "$work_dir/release"

git clone --depth 1 --branch "$llama_version" https://github.com/ggml-org/llama.cpp.git "$work_dir/source"
[[ "$(git -C "$work_dir/source" rev-parse HEAD)" == "$llama_revision" ]]
(
    cd "$work_dir/source"
    bash build-xcframework.sh ios-sim
)

# Preserve the official device/Mac binaries and their symbols without rebuilding them.
release_framework="$work_dir/release/build-apple/llama.xcframework"
simulator_framework="$work_dir/source/build-apple/llama.xcframework"
framework_args=()
for slice in "$release_framework"/*/ "$simulator_framework"/*/; do
    framework_args+=(-framework "${slice}llama.framework")
    framework_args+=(-debug-symbols "${slice}dSYMs/llama.dSYM")
done
xcrun xcodebuild -create-xcframework "${framework_args[@]}" -output "$work_dir/llama.xcframework"
mkdir -p "$package_root/Artifacts"
mv "$work_dir/llama.xcframework" "$destination"
printf 'Apple framework ready: %s\n' "$destination"
