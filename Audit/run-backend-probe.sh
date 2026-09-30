#!/bin/bash
set -euo pipefail
package_root="$(cd "$(dirname "$0")/.." && pwd)"
framework_dir="$package_root/Artifacts/llama-b10964.xcframework/macos-arm64_x86_64"
model="${1:-$package_root/Tests/SwiftLlamaTests/Resources/Llama-3.2-1B-Instruct-Q4_K_M.gguf}"
[[ -f "$model" && -d "$framework_dir/llama.framework" ]]
binary="$(mktemp /tmp/enclave-llama-backend-audit.XXXXXX)"
trap 'rm -f "$binary"' EXIT
xcrun clang++ -std=c++17 -O3 "$package_root/Audit/backend-sampling-probe.cpp" \
    -I "$framework_dir/llama.framework/Headers" -F "$framework_dir" -framework llama \
    -Wl,-rpath,"$framework_dir" -o "$binary"
"$binary" "$model"
