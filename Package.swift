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
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "SwiftLlama",
            targets: ["SwiftLlama"]),
    ],
    dependencies: [
//        .package(path: "/Users/mlody/Projects/AI/llama.cpp"),
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
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
            name: "LlamaSwiftTests",
            dependencies: ["SwiftLlama"],
            resources: [.copy("Models")]
        ),
    ]
)
