# Upstream llama.cpp (reference only)

This folder holds a **local checkout of [llama.cpp](https://github.com/ggml-org/llama.cpp)** so you can compare headers and implementation with **SwiftLlama**. It is **not** built or linked by SwiftPM; the device/Mac binaries come from the xcframework URL in `Package.swift`. The optional simulator setup script uses a separate temporary source checkout at the same revision.

## Pin

Keep the checkout aligned with `llamaVersion` in `Package.swift` (currently `b10964`, the binary release for llama.cpp v0.4.1).

## (Re)fetch

From the `swift-llama-cpp` package root:

```bash
rm -rf Reference/llama.cpp
git clone --depth 1 --branch <TAG> https://github.com/ggml-org/llama.cpp.git Reference/llama.cpp
```

Replace `<TAG>` with the same string as `llamaVersion`. A shallow clone (`--depth 1`) is enough for reading source at that tag; use a full clone if you need deep history.

The `Reference/llama.cpp` directory is gitignored in this package so the large tree is not committed.
