# Spiketrans Build & Test Makefile
# Pure Swift / MLX Metal SNN Speech-to-Text Framework

SCHEME_PKG := spiketrans-Package
SCHEME_TRAIN := train
DESTINATION := platform=macOS
CONFIG := Release

# 引数パラメータ (CLIから DIR=... や EPOCHS=... 等で上書き可能)
DIR ?= /path/to/loanword128
EPOCHS ?= 200
SAMPLES ?= 100
CONFIDENCE ?= 0.05
DURATION ?= 3
WORKERS ?= 8
WEIGHTS ?= .tmp/weights_twostage100_e200.json
# ALIF (適応型発火閾値) パラメータ。GAMMA=0 で従来の固定閾値 LIF と等価
ALIF_BETA ?= 0.92
ALIF_RHO ?= 0.85
ALIF_GAMMA ?= 0.15
# 第2段 Viterbi DP における言語 SNN 予測一致の加点 (0 で言語 SNN 不使用)
LM_BONUS ?= 4.0
# 切り詰め BPTT の窓幅 (フレーム単位)。1 でフレーム間の勾配を切る
BPTT_WINDOW ?= 16
# GPU 学習の Cosine 学習率
LR_MAX ?= 0.035
LR_MIN ?= 0.0005
# N エポックごとの重み保存 (0 で無効)
CKPT_EVERY ?= 0
# マトリョーシカ各スライスの損失重み
SW_BASE ?= 0.1
SW_MID ?= 0.2
SW_HIGH ?= 1.0
ALIF_ARGS := --alif-beta $(ALIF_BETA) --alif-rho $(ALIF_RHO) --alif-gamma $(ALIF_GAMMA) --language-bonus $(LM_BONUS) --bptt-window $(BPTT_WINDOW) --lr-max $(LR_MAX) --lr-min $(LR_MIN) --checkpoint-every $(CKPT_EVERY) --slice-weight-base $(SW_BASE) --slice-weight-middle $(SW_MID) --slice-weight-high $(SW_HIGH)

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
	$$BIN -s $(SAMPLES) -e $(EPOCHS) --min-confidence $(CONFIDENCE) --min-duration $(DURATION) $(ALIF_ARGS) --export-weights $(WEIGHTS) -d $(DIR)

## SNN-CTC 損失並列学習 & CTC Prefix Beam Search 評価
train-ctc: build-train
	@BIN=$$(ls -1d ~/Library/Developer/Xcode/DerivedData/spiketrans-*/Build/Products/$(CONFIG)/train | head -n 1); \
	$$BIN -s $(SAMPLES) -e $(EPOCHS) -p $(WORKERS) --ctc $(ALIF_ARGS) --export-weights $(WEIGHTS) -d $(DIR)

## 学習済み重みを用いた全発話 2段階漢字 CER 評価 (引数: DIR=..., WEIGHTS=...)
eval: build-train
	@BIN=$$(ls -1d ~/Library/Developer/Xcode/DerivedData/spiketrans-*/Build/Products/$(CONFIG)/train | head -n 1); \
	$$BIN -s $(SAMPLES) --import-weights $(WEIGHTS) --min-confidence $(CONFIDENCE) --min-duration $(DURATION) $(ALIF_ARGS) -d $(DIR)

## 学習済み重みを用いた全発話 SNN-CTC Prefix Beam Search CER 評価 (引数: DIR=..., WEIGHTS=...)
eval-ctc: build-train
	@BIN=$$(ls -1d ~/Library/Developer/Xcode/DerivedData/spiketrans-*/Build/Products/$(CONFIG)/train | head -n 1); \
	$$BIN -s $(SAMPLES) --import-weights $(WEIGHTS) --ctc $(ALIF_ARGS) -d $(DIR)

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
	@echo "  make train-ctc [DIR=...] - SNN-CTC 損失学習 (GPU) & CTC ビーム探索評価を実行"
	@echo "  make eval [DIR=...]      - 学習済み重みを用いて全発話の 2段階漢字 CER 評価を実行 (例: make eval DIR=.tmp/loanword128)"
	@echo "  make eval-ctc [DIR=...]  - 学習済み重みを用いて CTC Prefix Beam Search 評価を実行"
	@echo "  make clean               - ビルド成果物と中間生成物をクリーン"
	@echo ""
	@echo "追加パラメータ:"
	@echo "  ALIF_GAMMA=0.15          - ALIF 発火時閾値上昇幅 (0 で固定閾値 LIF と等価)"
	@echo "  ALIF_RHO=0.85            - ALIF 適応閾値減衰率"
	@echo "  ALIF_BETA=0.92           - 膜電位減衰率"
	@echo "  LM_BONUS=4.0             - 第2段 Viterbi DP への言語 SNN 加点 (0 で言語 SNN 不使用)"
	@echo "  BPTT_WINDOW=16           - 切り詰め BPTT 窓幅 (1 でフレーム間の勾配を切る)"
