// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let llamaVersion = "b5954"
let llamaChecksum = "929ce60d47a6aa1ecdd0dd301526a390c9637baa6d5c24121398846c47e4de49"

let package = Package(
    name: "swift-llama-cpp",
    platforms: [
        .macOS(.v14),
        .iOS(.v17)
    ],
    products: [
        .library(
            name: "SwiftLlama",
            targets: ["SwiftLlama"]),
    ],
    dependencies: [
    ],
    targets: [
        .target(
            name: "SwiftLlama",
            dependencies: [
                "llama"
            ]
        ),
        .binaryTarget(
            name: "llama",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/\(llamaVersion)/llama-\(llamaVersion)-xcframework.zip",
            checksum: llamaChecksum
        ),
        .testTarget(
            name: "SwiftLlamaTests",
            dependencies: ["SwiftLlama"],
            resources: [.copy("Models")]
        ),
    ]
)
