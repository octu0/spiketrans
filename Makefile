# Spiketrans Build & Test Makefile
# Pure Swift / MLX Metal SNN Speech-to-Text Framework

SCHEME_PKG := spiketrans-Package
SCHEME_TRAIN := train
DESTINATION := platform=macOS
CONFIG := Release

# 引数パラメータ (CLIから DIR=... や EPOCHS=... 等で上書き可能)
DIR ?= .tmp/loanword128
EPOCHS ?= 200
SAMPLES ?= 100
CONFIDENCE ?= 0.05
DURATION ?= 3
WORKERS ?= 8
WEIGHTS ?= .tmp/weights_twostage100_e200.json

.PHONY: all build build-train test train train-ctc eval eval-ctc clean help

all: build test

## ビルド (パッケージ全体)
build:
	xcodebuild build -scheme $(SCHEME_PKG) -destination '$(DESTINATION)' -configuration $(CONFIG)

## 実行可能バイナリ (train) のビルド
build-train:
	xcodebuild build -scheme $(SCHEME_TRAIN) -destination '$(DESTINATION)' -configuration $(CONFIG)

## ユニットテスト・敵対的テストの実行 (Release 最適化構成)
test:
	xcodebuild test -scheme $(SCHEME_PKG) -destination '$(DESTINATION)' -configuration $(CONFIG) ENABLE_TESTABILITY=YES

## GPU 深層学習 & 2段階漢字 CER 評価 (引数: DIR=..., EPOCHS=..., SAMPLES=...)
train: build-train
	@BIN=$$(ls -1d ~/Library/Developer/Xcode/DerivedData/spiketrans-*/Build/Products/$(CONFIG)/train | head -n 1); \
	$$BIN -s $(SAMPLES) -e $(EPOCHS) --min-confidence $(CONFIDENCE) --min-duration $(DURATION) --export-weights $(WEIGHTS) -d $(DIR)

## SNN-CTC 損失並列学習 & CTC Prefix Beam Search 評価
train-ctc: build-train
	@BIN=$$(ls -1d ~/Library/Developer/Xcode/DerivedData/spiketrans-*/Build/Products/$(CONFIG)/train | head -n 1); \
	$$BIN -s $(SAMPLES) -e $(EPOCHS) -p $(WORKERS) --device cpu --ctc --export-weights .tmp/weights_ctc100.json -d $(DIR)

## 学習済み重みを用いた全発話 2段階漢字 CER 評価 (引数: DIR=..., WEIGHTS=...)
eval: build-train
	@BIN=$$(ls -1d ~/Library/Developer/Xcode/DerivedData/spiketrans-*/Build/Products/$(CONFIG)/train | head -n 1); \
	$$BIN -s $(SAMPLES) --import-weights $(WEIGHTS) --min-confidence $(CONFIDENCE) --min-duration $(DURATION) -d $(DIR)

## 学習済み重みを用いた全発話 SNN-CTC Prefix Beam Search CER 評価 (引数: DIR=..., WEIGHTS=...)
eval-ctc: build-train
	@BIN=$$(ls -1d ~/Library/Developer/Xcode/DerivedData/spiketrans-*/Build/Products/$(CONFIG)/train | head -n 1); \
	$$BIN -s $(SAMPLES) --import-weights $(WEIGHTS) --ctc -d $(DIR)

## クリーン
clean:
	xcodebuild clean -scheme $(SCHEME_PKG) -destination '$(DESTINATION)' -configuration $(CONFIG)
	xcodebuild clean -scheme $(SCHEME_TRAIN) -destination '$(DESTINATION)' -configuration $(CONFIG)
	swift package clean

## ヘルプ表示
help:
	@echo "Spiketrans Makefile Commands:"
	@echo "  make build               - パッケージ全体を Release ビルド"
	@echo "  make build-train         - train 実行バイナリを Release ビルド"
	@echo "  make test                - Release 構成で全ユニットテスト・敵対的テストを実行"
	@echo "  make train [DIR=...]     - GPU 深層学習 & 2段階漢字 CER 評価を実行 (例: make train DIR=.tmp/loanword128 EPOCHS=200)"
	@echo "  make train-ctc [DIR=...] - SNN-CTC 損失並列学習 & CTC ビーム探索評価を実行"
	@echo "  make eval [DIR=...]      - 学習済み重みを用いて全発話の 2段階漢字 CER 評価を実行 (例: make eval DIR=.tmp/loanword128)"
	@echo "  make eval-ctc [DIR=...]  - 学習済み重みを用いて CTC Prefix Beam Search 評価を実行"
	@echo "  make clean               - ビルド成果物と中間生成物をクリーン"
