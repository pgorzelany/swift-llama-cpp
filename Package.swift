// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import Foundation
import PackageDescription

let llamaVersion = "b10964"
let llamaChecksum = "b342a31c3160095d02777cbf1719013d4e6ff7556a1d46ab9eb5df0a93d13c47"
let llamaRevision = "b29c606e28a01b1bc8c1351026a0fa6e616bf6c4"

// Upstream releases omit simulator slices. The setup script adds them at the same revision.
let localFramework = "Artifacts/llama-\(llamaVersion).xcframework"
let localFrameworkURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().appendingPathComponent(localFramework)
let llamaTarget: Target = FileManager.default.fileExists(atPath: localFrameworkURL.path)
    ? .binaryTarget(name: "llama", path: localFramework)
    : .binaryTarget(
        name: "llama",
        url: "https://github.com/ggml-org/llama.cpp/releases/download/\(llamaVersion)/llama-\(llamaVersion)-xcframework.zip",
        checksum: llamaChecksum
    )

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
        llamaTarget,
        .testTarget(
            name: "SwiftLlamaTests",
            dependencies: ["SwiftLlama"],
            resources: [.copy("Resources")]
        ),
    ]
)
