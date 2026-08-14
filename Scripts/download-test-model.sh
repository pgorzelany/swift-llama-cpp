#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
package_dir=$(dirname "$script_dir")
model_name="Llama-3.2-1B-Instruct-Q4_K_M.gguf"
model_path="$package_dir/Tests/SwiftLlamaTests/Resources/$model_name"
partial_path="$model_path.partial"
expected_sha256="6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83"
model_url="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/$model_name"

checksum() {
    shasum -a 256 "$1" | awk '{print $1}'
}

if [ -f "$model_path" ] && [ "$(checksum "$model_path")" = "$expected_sha256" ]; then
    echo "Test model is already present and verified."
    exit 0
fi

mkdir -p "$(dirname "$model_path")"
curl --fail --location --continue-at - --output "$partial_path" "$model_url"

actual_sha256=$(checksum "$partial_path")
if [ "$actual_sha256" != "$expected_sha256" ]; then
    echo "Test model checksum mismatch: expected $expected_sha256, received $actual_sha256" >&2
    exit 1
fi

mv "$partial_path" "$model_path"
echo "Downloaded and verified $model_path"
