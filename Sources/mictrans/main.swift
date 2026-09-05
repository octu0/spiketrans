@preconcurrency import AVFoundation
import CoreAudio
import Foundation
import Spiketrans

setbuf(stdout, nil)

// マイク入力をその場で文字起こしする体験用アプリ。
//
// 無音で区切って発話単位に文字起こしする。認識は 2 段構成 (音響 SNN でかな →
// 辞書 Viterbi で漢字) で、推論は CPU の Event-driven 疎スパイク。

// MARK: - 引数

var weightsPath = ""
var corpusPath = ""
var silenceSeconds = 0.6      // この長さの無音で発話終了とみなす
var minSpeechSeconds = 0.3    // これより短い有声区間は雑音として捨てる
var maxSpeechSeconds = 15.0   // 話し続けた場合の強制区切り
var showKana = false
var listMicrophones = false
var showMeter = false
var micSelector = ""          // 番号または名前の一部

var argIdx = 1
let args = CommandLine.arguments
while argIdx < args.count {
    let arg = args[argIdx]
    switch arg {
    case "-w", "--weights":
        if (argIdx + 1) < args.count {
            weightsPath = args[argIdx + 1]
            argIdx += 1
        }
    case "-d", "--corpus":
        if (argIdx + 1) < args.count {
            corpusPath = args[argIdx + 1]
            argIdx += 1
        }
    case "--silence":
        if (argIdx + 1) < args.count {
            if let v = Double(args[argIdx + 1]) {
                silenceSeconds = max(0.1, v)
            }
            argIdx += 1
        }
    case "--kana":
        showKana = true
    case "--meter":
        showMeter = true
    case "--mic-list":
        listMicrophones = true
    case "--mic":
        if (argIdx + 1) < args.count {
            micSelector = args[argIdx + 1]
            argIdx += 1
        }
    case "-h", "--help":
        print("""
        マイク入力の文字起こし

          mictrans -w <重み.json> [-d <辞書テキスト>] [オプション]

        オプション:
          --mic-list         入力デバイスの一覧を表示して終了する
          --mic <番号|名前>  使う入力デバイスを指定する (既定はシステムの既定デバイス)
          --silence <秒>     無音で発話を区切る長さ (既定 0.6)
          --kana             かな (第1段の生出力) も表示する
          --meter            入力音量と VAD の判定を表示する

        かな語彙は重みファイルに同梱されているものを使う。
        -d に文の一覧を渡すと第2段のかな漢字変換も行う (省略時はかなのみ)。
        """)
        exit(0)
    default:
        break
    }
    argIdx += 1
}

// MARK: - 入力デバイスの列挙と選択

/// CoreAudio 上の入力デバイス
struct AudioInputDevice {
    let id: AudioDeviceID
    let name: String
    let channels: Int
    let sampleRate: Double
}

/// 数値プロパティの取得。参照型は扱わない (CFString は別途 getDeviceName で扱う)
func deviceProperty<T: Numeric>(
    _ deviceId: AudioDeviceID,
    selector: AudioObjectPropertySelector,
    scope: AudioObjectPropertyScope,
    initial: T
) -> T? {
    var address = AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: scope,
        mElement: kAudioObjectPropertyElementMain
    )
    var size = UInt32(MemoryLayout<T>.size)
    var value = initial
    let status = withUnsafeMutablePointer(to: &value) { ptr in
        AudioObjectGetPropertyData(deviceId, &address, 0, nil, &size, ptr)
    }
    if status != noErr {
        return nil
    }
    return value
}

/// デバイス名の取得。CFString は Unmanaged 経由で受け取る
func getDeviceName(_ deviceId: AudioDeviceID) -> String {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioObjectPropertyName,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var unmanaged: Unmanaged<CFString>? = nil
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    let status = withUnsafeMutablePointer(to: &unmanaged) { ptr in
        AudioObjectGetPropertyData(deviceId, &address, 0, nil, &size, ptr)
    }
    if status != noErr {
        return "(名称不明)"
    }
    switch unmanaged {
    case .some(let ref):
        return ref.takeRetainedValue() as String
    case .none:
        return "(名称不明)"
    }
}

/// 入力チャネルを持つデバイスだけを返す
func listInputDevices() -> [AudioInputDevice] {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var dataSize: UInt32 = 0
    if AudioObjectGetPropertyDataSize(
        AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &dataSize) != noErr {
        return []
    }
    let count = Int(dataSize) / MemoryLayout<AudioDeviceID>.size
    var ids = [AudioDeviceID](repeating: 0, count: count)
    if AudioObjectGetPropertyData(
        AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &dataSize, &ids) != noErr {
        return []
    }

    var results: [AudioInputDevice] = []
    for deviceId in ids {
        // 入力チャネル数を数える
        var streamAddress = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamConfiguration,
            mScope: kAudioDevicePropertyScopeInput,
            mElement: kAudioObjectPropertyElementMain
        )
        var streamSize: UInt32 = 0
        if AudioObjectGetPropertyDataSize(deviceId, &streamAddress, 0, nil, &streamSize) != noErr {
            continue
        }
        if streamSize == 0 {
            continue
        }
        let bufferList = UnsafeMutableRawPointer.allocate(
            byteCount: Int(streamSize), alignment: MemoryLayout<AudioBufferList>.alignment)
        defer { bufferList.deallocate() }
        if AudioObjectGetPropertyData(
            deviceId, &streamAddress, 0, nil, &streamSize, bufferList) != noErr {
            continue
        }
        let lists = UnsafeMutableAudioBufferListPointer(
            bufferList.assumingMemoryBound(to: AudioBufferList.self))
        var channels = 0
        for buffer in lists {
            channels += Int(buffer.mNumberChannels)
        }
        if channels == 0 {
            continue
        }

        let name = getDeviceName(deviceId)
        let rate = deviceProperty(
            deviceId,
            selector: kAudioDevicePropertyNominalSampleRate,
            scope: kAudioObjectPropertyScopeGlobal,
            initial: Float64(0)
        ) ?? 0
        results.append(AudioInputDevice(
            id: deviceId, name: name, channels: channels, sampleRate: rate))
    }
    return results
}

func defaultInputDeviceId() -> AudioDeviceID? {
    return deviceProperty(
        AudioObjectID(kAudioObjectSystemObject),
        selector: kAudioHardwarePropertyDefaultInputDevice,
        scope: kAudioObjectPropertyScopeGlobal,
        initial: AudioDeviceID(0)
    )
}

if listMicrophones {
    let devices = listInputDevices()
    if devices.isEmpty {
        print("入力デバイスが見つかりません。")
        exit(1)
    }
    let defaultId = defaultInputDeviceId()
    print("入力デバイス一覧:")
    var i = 0
    while i < devices.count {
        let d = devices[i]
        let mark: String
        if d.id == defaultId {
            mark = " [既定]"
        } else {
            mark = ""
        }
        print(String(format: "  %d: %@ (%d ch, %.0f Hz)%@",
                     i, d.name, d.channels, d.sampleRate, mark))
        i += 1
    }
    print("")
    print("--mic <番号> または --mic <名前の一部> で選べます。")
    exit(0)
}

if weightsPath.isEmpty {
    print("使い方: mictrans -w <重み.json> [-d <辞書テキスト>] [--mic <番号>]  (詳細は --help)")
    exit(1)
}

// MARK: - 終了処理

/// Ctrl+C の受け口。
///
/// main キューに載せると、辞書構築中やマイク許可待ちで main スレッドが
/// 止まっている間はハンドラが動けず、SIG_IGN のせいで無視だけが残ってしまう。
/// そのためグローバルキューで動かし、状態はこのクラスに持たせる
/// (トップレベル変数は @MainActor 隔離されるため別スレッドから触れない)。
final class ShutdownController: @unchecked Sendable {
    private let lock = NSLock()
    private var session: MicSession? = nil
    private var seen = false

    func attach(_ session: MicSession) {
        lock.lock()
        self.session = session
        lock.unlock()
    }

    func handle() {
        lock.lock()
        if seen {
            lock.unlock()
            print("")
            print("強制終了します。")
            exit(130)
        }
        seen = true
        let current = session
        lock.unlock()

        print("")
        print("終了します...")
        current?.finish()
        exit(0)
    }
}

/// シグナル監視を仕掛ける。
/// controller を引数で受けることで、クロージャがトップレベル変数を掴まないようにする
func installInterruptHandler(_ controller: ShutdownController) -> DispatchSourceSignal {
    let source = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
    source.setEventHandler {
        controller.handle()
    }
    source.resume()
    signal(SIGINT, SIG_IGN)
    return source
}

let shutdown = ShutdownController()
let interruptSource = installInterruptHandler(shutdown)

// MARK: - モデルと辞書の準備

guard let weights = try? SpikingNetworkWeights.load(from: URL(fileURLWithPath: weightsPath)) else {
    print("エラー: 重みを読み込めません: \(weightsPath)")
    exit(1)
}

// 第2段の辞書テキスト。省略した場合は第1段のかな出力だけを表示する
var corpusLines: [String] = []
if corpusPath.isEmpty != true {
    guard let corpusContent = try? String(contentsOfFile: corpusPath, encoding: .utf8) else {
        print("エラー: テキストを読み込めません: \(corpusPath)")
        exit(1)
    }
    for line in corpusContent.components(separatedBy: "\n") {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty != true {
            corpusLines.append(trimmed)
        }
    }
}

// かな語彙は重みに同梱されているものを使う。
// 古い重みには入っていないので、その場合だけテキストから作り直す
let phoneticVocabulary: TextVocabulary
switch weights.vocabulary {
case .some(let embedded):
    phoneticVocabulary = embedded
case .none:
    if corpusLines.isEmpty {
        print("エラー: この重みにはかな語彙が含まれていません。")
        print("  学習時と同じテキストを -d に指定するか、語彙を含む重みを使ってください。")
        exit(1)
    }
    let converter = KanjiConverter()
    phoneticVocabulary = TextVocabulary(
        corpus: corpusLines.map { converter.convertToHiragana($0) })
}
if phoneticVocabulary.size != weights.outputDim {
    print("エラー: かな語彙 \(phoneticVocabulary.size) 文字が重みの出力次元 \(weights.outputDim) と一致しません。")
    exit(1)
}

let kanaKanjiDict = KanaKanjiDictionary()
if corpusLines.isEmpty != true {
    kanaKanjiDict.buildFromCorpus(rawTexts: corpusLines)
}

let network = SpikingNetwork(weights: weights)

print("==================================================")
print("=== マイク入力 文字起こし ===")
print("==================================================")
print("第1段: 入力 \(weights.inputDim) 次元 / 隠れ \(weights.maxHiddenDim) / 出力 \(weights.outputDim)")
let vocabSource: String
if weights.vocabulary == nil {
    vocabSource = "テキストから再構築"
} else {
    vocabSource = "重みに同梱"
}
print("かな語彙: \(phoneticVocabulary.size) 文字 (\(vocabSource))")
if 0 < kanaKanjiDict.count {
    print("第2段辞書: \(kanaKanjiDict.count) 語")
} else {
    print("第2段辞書: なし (かなのみ表示します)")
}

// MARK: - 認識器 (音声取り込みスレッドとは別に、認識専用の直列キューで回す)

/// 音量と認識結果を 2 行に固定して表示する。
///
/// 端末なら ANSI のカーソル移動でその場を書き換え、
/// パイプへ流している場合は普通に 1 行ずつ出す。
final class StatusDisplay: @unchecked Sendable {
    private let lock = NSLock()
    private let live: Bool
    private let hasLevel: Bool
    private var drawn = false
    private var levelLine = ""
    private var resultLine = "(音声を待っています)"

    init(live: Bool, hasLevel: Bool) {
        self.live = live
        self.hasLevel = hasLevel
    }

    func setLevel(_ text: String) {
        lock.lock()
        levelLine = text
        if live {
            render()
        } else {
            print(text)
        }
        lock.unlock()
    }

    func setResult(_ text: String) {
        lock.lock()
        resultLine = text
        if live {
            render()
        } else {
            print(text)
        }
        lock.unlock()
    }

    /// 固定表示を終えて、続きの出力が重ならないようにする
    func finish() {
        lock.lock()
        if live && drawn {
            print("")
        }
        drawn = false
        lock.unlock()
    }

    /// 端末上の固定行を書き換える (live のときだけ呼ばれる)
    private func render() {
        var out = ""
        if drawn && hasLevel {
            // 1 行目の先頭へ戻る
            out += "\r\u{1B}[1A"
        } else {
            if drawn {
                out += "\r"
            }
        }
        if hasLevel {
            out += levelLine + "\u{1B}[K\n"
        }
        out += resultLine + "\u{1B}[K"
        drawn = true
        FileHandle.standardOutput.write(Data(out.utf8))
    }
}

/// マイクから来た 16kHz の PCM を溜めて、無音で区切って認識する
final class Transcriber: @unchecked Sendable {
    /// VAD の窓長 (64ms)。自己相関の計算長を確保するためホップより十分長く取る
    static let vadWindowSamples = 1024
    /// 入力ゲインの目標音量。学習時の正規化と同じ値に揃える
    static let vadTargetRMS: Float = 0.05

    private let network: SpikingNetwork
    private let phoneticVocabulary: TextVocabulary
    private let dictionary: KanaKanjiDictionary
    private let showKana: Bool
    private let showMeter: Bool
    private let display: StatusDisplay?
    private let frameStack = 4

    private let workspace: AcousticWorkspace
    private let acousticDecoder: AcousticDecoder
    private let beamDecoder: CTCBeamDecoder
    private let kanaDecoder: KanaKanjiDecoder

    /// 音声の蓄積と無音判定だけを行う軽いキュー
    private let audioQueue = DispatchQueue(label: "mictrans.audio")
    /// 認識 (特徴量抽出・SNN・ビーム探索) を行う重いキュー
    private let recognizeQueue = DispatchQueue(label: "mictrans.recognize")
    /// 認識中に次の途中結果を積まないための印 (audioQueue から触る)
    private var recognizing = false
    private var pending: [Float] = []
    private var silenceRun = 0
    private var speechRun = 0
    private var utteranceIndex = 0
    private var meterBlocks = 0
    private var meterPeak: Float = 0.0
    private var samplesSincePartial = 0
    /// 途中結果を出す間隔 (16kHz で 200ms)
    private let partialIntervalSamples = 3200

    private let silenceSamples: Int
    private let minSpeechSamples: Int
    private let maxSpeechSamples: Int
    /// 音量計算と VAD のホップ (10ms)
    private let blockSamples = 160

    /// 多次元 VAD (エネルギー・ゼロ交差率・自己相関有声度)。
    /// 単純な音量閾値と違い、背景ノイズフロアを学習しながら判定するので
    /// マイクの録音レベルが小さくても声を拾える
    private let vad: VAD
    private let dspWorkspace: DSPWorkspace
    /// VAD へ渡す窓。自己相関でピッチを見るため、ホップより長く取る
    private var vadWindow: [Float]
    private var vadScaled: [Float]
    /// 直近の音量の最大値 (ゆっくり減衰)。入力ゲインの算出に使う
    private var recentPeakRMS: Float = 0.0
    private var lastVoicingRatio: Float = 0.0
    private var lastNoiseFloor: Float = 0.0
    private var lastGain: Float = 1.0

    init(
        network: SpikingNetwork,
        phoneticVocabulary: TextVocabulary,
        dictionary: KanaKanjiDictionary,
        silenceSeconds: Double,
        minSpeechSeconds: Double,
        maxSpeechSeconds: Double,
        showKana: Bool,
        showMeter: Bool,
        display: StatusDisplay?
    ) {
        self.network = network
        self.phoneticVocabulary = phoneticVocabulary
        self.dictionary = dictionary
        self.showKana = showKana
        self.showMeter = showMeter
        self.display = display
        self.vad = VAD(config: DSPConfig())
        self.dspWorkspace = DSPWorkspace()
        self.vadWindow = [Float](repeating: 0.0, count: Transcriber.vadWindowSamples)
        self.vadScaled = [Float](repeating: 0.0, count: Transcriber.vadWindowSamples)
        self.silenceSamples = Int(silenceSeconds * 16000.0)
        self.minSpeechSamples = Int(minSpeechSeconds * 16000.0)
        self.maxSpeechSamples = Int(maxSpeechSeconds * 16000.0)

        self.workspace = AcousticWorkspace(
            maxHiddenDim: network.maxHiddenDim,
            outputDim: network.outputDim,
            inputDim: network.inputDim,
            numLayers: network.numLayers
        )
        self.acousticDecoder = AcousticDecoder(
            network: network,
            vocabulary: phoneticVocabulary,
            fallbackVocabulary: PhonemeVocabulary()
        )
        self.beamDecoder = CTCBeamDecoder(
            vocabulary: phoneticVocabulary,
            blankId: TextVocabulary.padId,
            beamWidth: 16
        )
        self.kanaDecoder = KanaKanjiDecoder(
            dictionary: dictionary,
            languageModel: dictionary.makeContextScorer(),
            languageBonus: 0.0,
            lexicalWeight: 1.0,
            lmWeight: 0.3
        )
    }

    /// マイクスレッドから呼ぶ。蓄積と無音判定だけを音声キューで行う
    func append(_ samples: [Float]) {
        audioQueue.async {
            self.consume(samples)
        }
    }

    /// 取り込み終了時に、溜まっているぶんを吐き出す。
    /// 認識が長引いても終了操作を待たせないよう、待ち時間に上限を設ける
    func flushRemaining(timeout: TimeInterval = 2.0) {
        let gate = DispatchSemaphore(value: 0)
        audioQueue.async {
            let snapshot = self.pending
            self.pending.removeAll(keepingCapacity: true)
            self.recognizeQueue.async {
                if self.minSpeechSamples <= snapshot.count {
                    self.recognize(snapshot, isFinal: true)
                }
                gate.signal()
            }
        }
        if gate.wait(timeout: .now() + timeout) == .timedOut {
            // タイムアウト時は待機を終了
        }
    }

    /// 認識を重いキューへ投げる。
    /// 途中結果は前の認識が終わっていなければ捨てる (積み上がると遅延が増え続ける)
    private func schedule(_ pcm: [Float], isFinal: Bool) {
        if isFinal != true {
            if recognizing {
                return
            }
            recognizing = true
            recognizeQueue.async {
                self.recognize(pcm, isFinal: false)
                self.audioQueue.async {
                    self.recognizing = false
                }
            }
            return
        }
        recognizeQueue.async {
            self.recognize(pcm, isFinal: true)
        }
    }

    private func consume(_ samples: [Float]) {
        var offset = 0
        while offset < samples.count {
            let end = min(offset + blockSamples, samples.count)
            let block = Array(samples[offset..<end])
            offset = end

            var sumSquares: Float = 0.0
            for v in block {
                sumSquares += v * v
            }
            let rms = sqrtf(sumSquares / Float(max(1, block.count)))
            let isSpeech = judgeSpeech(block: block, rms: rms)

            if showMeter {
                if meterPeak < rms {
                    meterPeak = rms
                }
                meterBlocks += 1
                // 10 ブロック = 約 0.1 秒ごとに書き換える
                if 10 <= meterBlocks {
                    let bars = Int(min(24.0, meterPeak * 400.0))
                    let bar = String(repeating: "=", count: max(0, bars))
                    let mark: String
                    if 0 < speechRun {
                        mark = "有声"
                    } else {
                        mark = "無音"
                    }
                    // 行末はエスケープで消すので詰め物は要らない
                    let text = String(
                        format: "[音量] %.4f x%.1f 有声度 %.2f 底 %.5f %@ ",
                        meterPeak, lastGain, lastVoicingRatio, lastNoiseFloor, mark) + bar
                    switch display {
                    case .some(let d):
                        d.setLevel(text)
                    case .none:
                        print("  " + text)
                    }
                    meterBlocks = 0
                    meterPeak = 0.0
                }
            }

            // 発話前の無音は溜め込まない (認識対象を短く保つ)
            if isSpeech != true && speechRun == 0 {
                continue
            }

            pending.append(contentsOf: block)
            if isSpeech {
                speechRun += block.count
                silenceRun = 0
            } else {
                silenceRun += block.count
            }

            let longEnough = (minSpeechSamples <= speechRun)
            let endedBySilence = (silenceSamples <= silenceRun)
            let endedByLength = (maxSpeechSamples <= pending.count)
            if (longEnough && endedBySilence) || endedByLength {
                schedule(pending, isFinal: true)
                pending.removeAll(keepingCapacity: true)
                speechRun = 0
                silenceRun = 0
                samplesSincePartial = 0
                continue
            }
            // 短い有声のまま無音が続いたら雑音として捨てる
            if longEnough != true && endedBySilence {
                pending.removeAll(keepingCapacity: true)
                speechRun = 0
                silenceRun = 0
                samplesSincePartial = 0
                continue
            }

            // 話している間も一定間隔で途中結果を出す。
            // 窓を伸ばして毎回作り直すため、結果は一括処理と厳密に一致する
            samplesSincePartial += block.count
            if longEnough && partialIntervalSamples <= samplesSincePartial {
                samplesSincePartial = 0
                schedule(pending, isFinal: false)
            }
        }
    }

    /// 適応 VAD による有声判定。
    ///
    /// 録音レベルが小さいマイクでも拾えるよう、直近の音量から入力ゲインを
    /// 見積もって VAD へ渡す窓だけを増幅する (蓄積する音声は元のまま。
    /// 特徴量抽出側で発話単位の正規化が改めて行われる)
    private func judgeSpeech(block: [Float], rms: Float) -> Bool {
        // 窓をホップぶんずらして新しいブロックを末尾へ入れる
        let window = Transcriber.vadWindowSamples
        let shift = min(block.count, window)
        if shift < window {
            var i = 0
            while i < (window - shift) {
                vadWindow[i] = vadWindow[i + shift]
                i += 1
            }
        }
        var j = 0
        while j < shift {
            vadWindow[window - shift + j] = block[block.count - shift + j]
            j += 1
        }

        // ゆっくり減衰する最大音量から、目標レベルへ寄せるゲインを決める
        let decayed = recentPeakRMS * 0.995
        if decayed < rms {
            recentPeakRMS = rms
        } else {
            recentPeakRMS = decayed
        }
        var gain: Float = 1.0
        if 1e-5 < recentPeakRMS {
            gain = min(20.0, max(1.0, Transcriber.vadTargetRMS / recentPeakRMS))
        }
        lastGain = gain

        var k = 0
        while k < window {
            vadScaled[k] = vadWindow[k] * gain
            k += 1
        }

        var result: VADResult? = nil
        vadScaled.withUnsafeBufferPointer { ptr in
            result = vad.processFrame(ptr: ptr.baseAddress!, count: window, workspace: dspWorkspace)
        }
        switch result {
        case .some(let r):
            lastVoicingRatio = r.voicingRatio
            lastNoiseFloor = r.noiseFloor
            return r.isSpeech
        case .none:
            return false
        }
    }

    private func recognize(_ pcm: [Float], isFinal: Bool) {
        let started = CFAbsoluteTimeGetCurrent()
        let features: [[Float]]
        switch network.convSubsampling {
        case .some:
            features = SpeechDataset.extractMelSpectrogram(pcmData: pcm)
        case .none:
            features = SpeechDataset.extractFeaturesFromPCM(pcmData: pcm, frameStack: frameStack)
        }
        if features.isEmpty {
            return
        }

        // 毎回ゼロから積分し直す。膜電位を持ち越すと前の発話の状態が混ざる
        workspace.resetHiddenState()

        let frameProbs = acousticDecoder.decodeSequence(
            featuresSeq: features,
            workspace: workspace
        )
        let outDim = network.outputDim
        var logProbs = [[Float]](
            repeating: [Float](repeating: 0.0, count: outDim),
            count: frameProbs.count
        )
        var f = 0
        while f < frameProbs.count {
            let probs = frameProbs[f].probabilities
            var c = 0
            while c < outDim {
                logProbs[f][c] = log(max(1e-30, probs[c]))
                c += 1
            }
            f += 1
        }

        let nBestHyps = beamDecoder.decodeNBest(logProbs: logProbs, n: 5)
        if nBestHyps.isEmpty {
            return
        }
        let kana = nBestHyps[0].text
        if kana.isEmpty {
            return
        }

        // 途中結果はかなだけ。漢字変換は文の区切りが要るので確定時に行う
        if isFinal != true {
            let seconds = Double(pcm.count) / 16000.0
            switch display {
            case .some(let d):
                d.setResult(String(format: "[…] (%.1fs) %@", seconds, kana))
            case .none:
                break
            }
            return
        }

        let hasDictionary = (0 < dictionary.count)
        var kanji = ""
        if hasDictionary {
            kanji = kanaDecoder.decode(acousticHypotheses: nBestHyps)
        }
        let elapsed = (CFAbsoluteTimeGetCurrent() - started) * 1000.0
        let seconds = Double(pcm.count) / 16000.0

        utteranceIndex += 1
        let body: String
        switch (hasDictionary, showKana) {
        case (false, _):
            body = kana
        case (true, true):
            body = "\(kana) → \(kanji)"
        case (true, false):
            body = kanji
        }

        switch display {
        case .some(let d):
            d.setResult(String(format: "[%d] (%.1fs/%.0fms) %@", utteranceIndex, seconds, elapsed, body))
        case .none:
            print("")
            print(String(format: "[%d] (%.1f 秒の音声を %.0f ms で認識)", utteranceIndex, seconds, elapsed))
            print("  → \(body)")
        }
    }
}

// 端末に出しているときは固定行を書き換える表示にする。
// --meter を付けると音量行が 1 行増える
let isTerminal = (isatty(1) == 1)
let statusDisplay: StatusDisplay?
switch isTerminal {
case true:
    statusDisplay = StatusDisplay(live: true, hasLevel: showMeter)
case false:
    statusDisplay = nil
}

let transcriber = Transcriber(
    network: network,
    phoneticVocabulary: phoneticVocabulary,
    dictionary: kanaKanjiDict,
    silenceSeconds: silenceSeconds,
    minSpeechSeconds: minSpeechSeconds,
    maxSpeechSeconds: maxSpeechSeconds,
    showKana: showKana,
    showMeter: showMeter,
    display: statusDisplay
)

// MARK: - マイク入力

// マイク許可を先に取る。初回はシステムのダイアログが出るので、
// ここで待たないとエンジン初期化のところで無言のまま止まって見える
func requestMicrophoneAccess() -> Bool {
    switch AVCaptureDevice.authorizationStatus(for: .audio) {
    case .authorized:
        return true
    case .notDetermined:
        print("")
        print("マイクの使用許可を求めています。表示されたダイアログで「許可」を選んでください...")
        let gate = DispatchSemaphore(value: 0)
        let granted = GrantBox()
        AVCaptureDevice.requestAccess(for: .audio) { ok in
            granted.value = ok
            gate.signal()
        }
        if gate.wait(timeout: .now() + 120.0) == .timedOut {
            print("マイク許可の応答がありませんでした。")
            return false
        }
        return granted.value
    default:
        return false
    }
}

final class GrantBox: @unchecked Sendable {
    var value = false
}

if requestMicrophoneAccess() != true {
    print("エラー: マイクの使用が許可されていません。")
    print("  システム設定 > プライバシーとセキュリティ > マイク で、実行元のターミナル (または本アプリ) を許可してください。")
    exit(1)
}

let engine = AVAudioEngine()
let inputNode = engine.inputNode

// 入力デバイスの指定があれば、フォーマットを読む前に切り替える。
// フォーマットはデバイス依存なので順序が重要
if micSelector.isEmpty != true {
    let devices = listInputDevices()
    var chosen: AudioInputDevice? = nil
    switch Int(micSelector) {
    case .some(let index):
        if 0 <= index && index < devices.count {
            chosen = devices[index]
        }
    case .none:
        let needle = micSelector.lowercased()
        for d in devices {
            if d.name.lowercased().contains(needle) {
                chosen = d
                break
            }
        }
    }
    guard let device = chosen else {
        print("エラー: 入力デバイス「\(micSelector)」が見つかりません。--mic-list で一覧を確認してください。")
        exit(1)
    }
    guard let audioUnit = inputNode.audioUnit else {
        print("エラー: 入力デバイスを切り替えられません。")
        exit(1)
    }
    var deviceId = device.id
    let status = AudioUnitSetProperty(
        audioUnit,
        kAudioOutputUnitProperty_CurrentDevice,
        kAudioUnitScope_Global,
        0,
        &deviceId,
        UInt32(MemoryLayout<AudioDeviceID>.size)
    )
    if status != noErr {
        print("エラー: 入力デバイスの切り替えに失敗しました (status \(status))")
        exit(1)
    }
    print("")
    print("入力デバイス: \(device.name)")
}

// タップが実際に受け取るのはノードの出力フォーマット。
// ハードウェアの入力フォーマットを渡すとチャネル数が食い違い、
// AVAudioEngine の内部アサーションで落ちる (Trace/BPT trap)
let tapFormat = inputNode.outputFormat(forBus: 0)
if tapFormat.sampleRate <= 0 || tapFormat.channelCount == 0 {
    print("エラー: マイクが利用できません。")
    print("  システム設定 > プライバシーとセキュリティ > マイク で、実行元のターミナルに許可を与えてください。")
    exit(1)
}

// 認識器は 16kHz モノラルの Float を前提にする。
// チャネルの取りまとめは自前で行い、AVAudioConverter は
// サンプリングレート変換 (モノラル → モノラル) だけに使う。
// 多チャネルのまま変換器に任せるとチャネル対応の解釈が環境依存になる
guard let monoSourceFormat = AVAudioFormat(
    commonFormat: .pcmFormatFloat32,
    sampleRate: tapFormat.sampleRate,
    channels: 1,
    interleaved: false
) else {
    print("エラー: 中間フォーマットを作成できません。")
    exit(1)
}
guard let targetFormat = AVAudioFormat(
    commonFormat: .pcmFormatFloat32,
    sampleRate: 16000.0,
    channels: 1,
    interleaved: false
) else {
    print("エラー: 変換先フォーマットを作成できません。")
    exit(1)
}
guard let converter = AVAudioConverter(from: monoSourceFormat, to: targetFormat) else {
    print("エラー: マイク入力を 16kHz へ変換できません。")
    exit(1)
}

print("")
let hwFormat = inputNode.inputFormat(forBus: 0)
print("マイク: \(Int(tapFormat.sampleRate)) Hz \(tapFormat.channelCount) ch → 16000 Hz 1 ch へ変換")
print("  (ハードウェア入力: \(Int(hwFormat.sampleRate)) Hz \(hwFormat.channelCount) ch)")
print("無音 \(String(format: "%.1f", silenceSeconds)) 秒で区切ります (適応 VAD: エネルギー・ゼロ交差率・自己相関有声度)")
print("話してください。Ctrl+C で終了します。")

/// マイク入力を 16kHz モノラルの Float 配列へ変換する。
/// AVAudioConverter と入力バッファは Sendable ではないため、
/// タップと同じスレッド内で完結させて外へは配列だけ渡す
final class MicConverter: @unchecked Sendable {
    private let converter: AVAudioConverter
    private let monoSourceFormat: AVAudioFormat
    private let targetFormat: AVAudioFormat
    private var reportedError = false

    init(converter: AVAudioConverter, monoSourceFormat: AVAudioFormat, targetFormat: AVAudioFormat) {
        self.converter = converter
        self.monoSourceFormat = monoSourceFormat
        self.targetFormat = targetFormat
    }

    /// 多チャネル入力をモノラルへ平均してから 16kHz へ変換する
    func convert(_ buffer: AVAudioPCMBuffer) -> [Float] {
        let frames = Int(buffer.frameLength)
        if frames == 0 {
            return []
        }
        guard let channels = buffer.floatChannelData else {
            return []
        }
        let channelCount = Int(buffer.format.channelCount)
        if channelCount == 0 {
            return []
        }

        guard let monoBuffer = AVAudioPCMBuffer(
            pcmFormat: monoSourceFormat,
            frameCapacity: AVAudioFrameCount(frames)
        ) else {
            return []
        }
        monoBuffer.frameLength = AVAudioFrameCount(frames)
        guard let monoChannel = monoBuffer.floatChannelData else {
            return []
        }
        let invChannels = 1.0 / Float(channelCount)
        var i = 0
        while i < frames {
            var sum: Float = 0.0
            var ch = 0
            while ch < channelCount {
                sum += channels[ch][i]
                ch += 1
            }
            monoChannel[0][i] = sum * invChannels
            i += 1
        }

        let ratio = targetFormat.sampleRate / monoSourceFormat.sampleRate
        let capacity = AVAudioFrameCount(Double(frames) * ratio) + 1024
        guard let converted = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else {
            return []
        }

        // 入力ブロックは 1 回だけ渡す。@Sendable なクロージャから可変変数を
        // 触らないよう、状態はクラスに持たせる
        final class FeedState: @unchecked Sendable {
            var consumed = false
            let buffer: AVAudioPCMBuffer
            init(buffer: AVAudioPCMBuffer) {
                self.buffer = buffer
            }
        }
        let feed = FeedState(buffer: monoBuffer)
        var conversionError: NSError? = nil
        let status = converter.convert(to: converted, error: &conversionError) { _, outStatus in
            if feed.consumed {
                outStatus.pointee = .noDataNow
                return nil
            }
            feed.consumed = true
            outStatus.pointee = .haveData
            return feed.buffer
        }
        if status == .error {
            if reportedError != true {
                reportedError = true
                print("警告: 音声の変換に失敗しました: \(conversionError?.localizedDescription ?? "不明")")
            }
            return []
        }
        let count = Int(converted.frameLength)
        if count == 0 {
            return []
        }
        guard let outChannel = converted.floatChannelData else {
            return []
        }
        var samples = [Float](repeating: 0.0, count: count)
        samples.withUnsafeMutableBufferPointer { dst in
            guard let base = dst.baseAddress else {
                return
            }
            base.update(from: outChannel[0], count: count)
        }
        return samples
    }
}

/// 取り込みの実体をまとめて保持する。
///
/// main.swift のトップレベル変数は Swift 6 では暗黙的に @MainActor 隔離される。
/// オーディオのタップは実時間スレッドから呼ばれるため、トップレベル変数を
/// 直接触ると隔離チェック (dispatch_assert_queue) で異常終了する。
/// タップから使うものはすべてこのクラスのインスタンスプロパティにしておく。
final class MicSession: @unchecked Sendable {
    private let engine: AVAudioEngine
    private let tapFormat: AVAudioFormat
    private let micConverter: MicConverter
    private let transcriber: Transcriber
    private let display: StatusDisplay?

    init(
        engine: AVAudioEngine,
        tapFormat: AVAudioFormat,
        micConverter: MicConverter,
        transcriber: Transcriber,
        display: StatusDisplay?
    ) {
        self.engine = engine
        self.tapFormat = tapFormat
        self.micConverter = micConverter
        self.transcriber = transcriber
        self.display = display
    }

    func start() throws {
        let converterRef = micConverter
        let transcriberRef = transcriber
        engine.inputNode.installTap(
            onBus: 0,
            bufferSize: 4096,
            format: tapFormat
        ) { buffer, _ in
            let samples = converterRef.convert(buffer)
            if samples.isEmpty != true {
                transcriberRef.append(samples)
            }
        }
        try engine.start()
    }

    /// 取り込みを止めてから、溜まっているぶんを吐き出す
    func finish() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        // 認識が長引いていても終了操作を待たせない
        transcriber.flushRemaining(timeout: 2.0)
        display?.finish()
    }
}

let session = MicSession(
    engine: engine,
    tapFormat: tapFormat,
    micConverter: MicConverter(
        converter: converter,
        monoSourceFormat: monoSourceFormat,
        targetFormat: targetFormat
    ),
    transcriber: transcriber,
    display: statusDisplay
)

shutdown.attach(session)

do {
    try session.start()
} catch {
    print("エラー: マイクを開始できません: \(error)")
    print("  システム設定 > プライバシーとセキュリティ > マイク で、実行元のターミナルに許可を与えてください。")
    exit(1)
}

RunLoop.main.run()
