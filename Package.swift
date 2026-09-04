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
        ),
        .executable(
            name: "stage2",
            targets: ["stage2"]
        ),
        .executable(
            name: "transcribe",
            targets: ["transcribe"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.21.0")
    ],
    targets: [
        .target(
            name: "Spiketrans",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift")
            ],
            path: "Sources/Spiketrans"
        ),
        .executableTarget(
            name: "train",
            dependencies: [
                "Spiketrans",
                .product(name: "MLX", package: "mlx-swift")
            ],
            path: "script/train",
            exclude: ["run.sh"]
        ),
        .executableTarget(
            name: "benchmark",
            dependencies: ["Spiketrans"],
            path: "script/benchmark",
            exclude: ["run.sh"]
        ),
        .executableTarget(
            name: "stage2",
            dependencies: [
                "Spiketrans",
                .product(name: "MLX", package: "mlx-swift")
            ],
            path: "script/stage2"
        ),
        .executableTarget(
            name: "transcribe",
            dependencies: ["Spiketrans"],
            path: "script/transcribe"
        ),
        .testTarget(
            name: "SpiketransTests",
            dependencies: [
                "Spiketrans",
                .product(name: "MLX", package: "mlx-swift")
            ],
            path: "Tests/SpiketransTests"
        )
    ]
)
