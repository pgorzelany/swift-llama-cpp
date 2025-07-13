// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

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
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b5880/llama-b5880-xcframework.zip",
            checksum: "95fefedf06f445a9c36b06b65bbfcfe401e38a84190ceeb14360e3bc22a3a5d6"
        ),
        .testTarget(
            name: "SwiftLlamaTests",
            dependencies: ["SwiftLlama"],
            resources: [.copy("Models")]
        ),
    ]
)
