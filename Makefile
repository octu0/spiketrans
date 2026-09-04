# Spiketrans Build & Test Makefile
# Pure Swift / MLX Metal SNN Speech-to-Text Framework

SCHEME_PKG := spiketrans-Package
SCHEME_TRAIN := train
DESTINATION := platform=macOS
CONFIG := Release

# 引数パラメータ (CLIから DATA=... や EPOCHS=... 等で上書き可能)
DATA ?= /path/to/train.jsonl
EPOCHS ?= 200
SAMPLES ?= 100
WORKERS ?= 8
WEIGHTS ?= .tmp/weights.json

.PHONY: all build build-train test train eval clean help

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

## GPU CTC 学習 & 全発話 CER 評価 (引数: DATA=<manifest.jsonl>, EPOCHS=..., SAMPLES=...)
## 学習パラメータは script/train/main.swift の Defaults を参照
train: build-train
	@BIN=$$(ls -1d ~/Library/Developer/Xcode/DerivedData/spiketrans-*/Build/Products/$(CONFIG)/train | head -n 1); \
	$$BIN -s $(SAMPLES) -e $(EPOCHS) -p $(WORKERS) --export-weights $(WEIGHTS) -d $(DATA)

## 学習済み重みを用いた全発話 CER 評価 (引数: DATA=<manifest.jsonl>, WEIGHTS=...)
eval: build-train
	@BIN=$$(ls -1d ~/Library/Developer/Xcode/DerivedData/spiketrans-*/Build/Products/$(CONFIG)/train | head -n 1); \
	$$BIN -s $(SAMPLES) -e 0 --import-weights $(WEIGHTS) -d $(DATA)

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
	@echo "  make train DATA=<manifest.jsonl> [EPOCHS=... SAMPLES=... WEIGHTS=...]"
	@echo "                           - GPU CTC 学習と全発話 CER 評価"
	@echo "  make eval DATA=<manifest.jsonl> WEIGHTS=... [SAMPLES=...]"
	@echo "                           - 学習済み重みで全発話 CER 評価"
	@echo "  make clean               - ビルド成果物と中間生成物をクリーン"
	@echo ""
	@echo "学習マニフェストは script/dataset/ の各コーパス用スクリプトで生成します。"
	@echo "学習パラメータは"
	@echo "script/train/main.swift の Defaults に定数として定義しています。"
