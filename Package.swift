// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "spiketrans",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "Spiketrans",
            targets: ["Spiketrans"]
        ),
        .executable(
            name: "train",
            targets: ["train"]
        ),
        .executable(
            name: "benchmark",
            targets: ["benchmark"]
        )
    ],
    dependencies: [],
    targets: [
        .target(
            name: "Spiketrans",
            dependencies: [],
            path: "Sources/Spiketrans"
        ),
        .executableTarget(
            name: "train",
            dependencies: ["Spiketrans"],
            path: "script/train",
            exclude: ["run.sh"]
        ),
        .executableTarget(
            name: "benchmark",
            dependencies: ["Spiketrans"],
            path: "script/benchmark",
            exclude: ["run.sh"]
        ),
        .testTarget(
            name: "SpiketransTests",
            dependencies: ["Spiketrans"],
            path: "Tests/SpiketransTests"
        )
    ]
)
