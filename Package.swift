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
//        .binaryTarget(
//            name: "llama",
//            url: "https://github.com/ggml-org/llama.cpp/releases/download/b5526/llama-b5526-xcframework.zip",
//            checksum: "d8aa166eb90f4235ae09a8f88dc8c59d416dbc8a0e531eacda08166e315cf2fa"
//        ),
        .binaryTarget(
            name: "llama",
            path: "./llama-b5215-xcframework.zip"
        ),
        .testTarget(
            name: "LlamaSwiftTests",
            dependencies: ["SwiftLlama"],
            resources: [.copy("Models")]
        ),
    ]
)
