import Foundation

/// 逆テキスト正規化 (ITN: Inverse Text Normalization) ルールエンジン
///
/// 音声認識のかな・漢字表記から、漢数字・読み（例: 「じゅうごえん」「にせんにじゅうろくねんきがつむいか」「ぜろきゅうぜろ」「ひゃくぱーせんと」）
/// を実用的なアラビア数字・表記（例: 「15円」「2026年9月6日」「090-」「100%」）へ高速・軽量に整形する。
/// SNN の超軽量性を維持するため、ゼロ外部依存かつ決定論的なストリーム走査で実装される。
public struct InverseTextNormalizer: Sendable {

    public init() {}

    /// 和風月日読み（1日〜10日、14日、20日、24日）のテーブル
    private static let specialDayReadings: [(reading: String, day: Int)] = [
        ("にじゅうよっか", 24),
        ("じゅうよっか", 14),
        ("ついたち", 1),
        ("ふつか", 2),
        ("みっか", 3),
        ("よっか", 4),
        ("いつか", 5),
        ("むいか", 6),
        ("なのか", 7),
        ("ようか", 8),
        ("ここのか", 9),
        ("とおか", 10),
        ("はつか", 20)
    ]

    /// 月のかな読みテーブル
    private static let monthReadings: [(reading: String, month: Int)] = [
        ("じゅうにがつ", 12),
        ("じゅういちがつ", 11),
        ("じゅうがつ", 10),
        ("いちがつ", 1),
        ("にがつ", 2),
        ("さんがつ", 3),
        ("しがつ", 4),
        ("よんがつ", 4),
        ("ごがつ", 5),
        ("ろくがつ", 6),
        ("しちがつ", 7),
        ("なながつ", 7),
        ("はちがつ", 8),
        ("くがつ", 9),
        ("きがつ", 9) // 音響 CTC で頻出する「9月」の音響的表記揺れ
    ]

    /// 電話番号プレフィックスのかな読みテーブル
    private static let phonePrefixReadings: [(reading: String, replacement: String)] = [
        ("ぜろきゅうぜろ", "090-"),
        ("ぜろはちぜろ", "080-"),
        ("ぜろななぜろ", "070-"),
        ("ぜろごぜろ", "050-"),
        ("ぜろさん", "03-"),
        ("〇九〇", "090-"),
        ("〇八〇", "080-"),
        ("〇七〇", "070-"),
        ("〇五〇", "050-")
    ]

    /// 分のかな読みテーブル (1〜59分)
    private static let minuteReadings: [(reading: String, minute: Int)] = [
        ("さんじっぷん", 30),
        ("さんじゅっぷん", 30),
        ("よんじゅうごふん", 45),
        ("じゅうごふん", 15),
        ("じゅっぷん", 10),
        ("じっぷん", 10),
        ("にじっぷん", 20),
        ("にじゅっぷん", 20),
        ("よんじっぷん", 40),
        ("よんじゅっぷん", 40),
        ("ごじっぷん", 50),
        ("ごじゅっぷん", 50),
        ("いっぷん", 1),
        ("にふん", 2),
        ("さんぷん", 3),
        ("よんぷん", 4),
        ("ごふん", 5),
        ("ろっぷん", 6),
        ("ななふん", 7),
        ("はっぷん", 8),
        ("きゅうふん", 9)
    ]

    /// 漢数字文字を数値に変換 (0〜9、それ以外は nil)
    private static func kanjiDigitValue(_ ch: Character) -> Int? {
        switch ch {
        case "〇", "零":
            return 0
        case "一", "壱":
            return 1
        case "二", "弐":
            return 2
        case "三", "参":
            return 3
        case "四":
            return 4
        case "五":
            return 5
        case "六":
            return 6
        case "七":
            return 7
        case "八":
            return 8
        case "九":
            return 9
        default:
            return nil
        }
    }

    /// 漢数字列 (位取り含む) を整数にパース
    /// 例: "十五" -> 15, "二千二十六" -> 2026, "二〇二六" -> 2026, "三百" -> 300, "一万五千" -> 15000
    private static func parseKanjiNumber(chars: [Character], start: Int) -> (value: Int64, length: Int)? {
        let n = chars.count
        if n <= start {
            return nil
        }

        var i = start
        var total: Int64 = 0
        var currentSection: Int64 = 0 // 万単位以内の蓄積
        var currentDigit: Int64 = 0
        var hasDigit = false
        var matched = false

        while i < n {
            let ch = chars[i]

            if let digit = kanjiDigitValue(ch) {
                if hasDigit {
                    currentDigit = currentDigit * 10 + Int64(digit)
                } else {
                    currentDigit = Int64(digit)
                    hasDigit = true
                }
                matched = true
                i += 1
                continue
            }

            switch ch {
            case "十":
                var d: Int64 = 1
                if hasDigit {
                    d = currentDigit
                }
                currentSection += d * 10
                currentDigit = 0
                hasDigit = false
                matched = true
                i += 1
            case "百":
                var d: Int64 = 1
                if hasDigit {
                    d = currentDigit
                }
                currentSection += d * 100
                currentDigit = 0
                hasDigit = false
                matched = true
                i += 1
            case "千":
                var d: Int64 = 1
                if hasDigit {
                    d = currentDigit
                }
                currentSection += d * 1000
                currentDigit = 0
                hasDigit = false
                matched = true
                i += 1
            case "万":
                let sec = currentSection + currentDigit
                var s: Int64 = 1
                if 0 < sec {
                    s = sec
                }
                total += s * 10000
                currentSection = 0
                currentDigit = 0
                hasDigit = false
                matched = true
                i += 1
            case "億":
                let sec = currentSection + currentDigit
                var s: Int64 = 1
                if 0 < sec {
                    s = sec
                }
                total += s * 100000000
                currentSection = 0
                currentDigit = 0
                hasDigit = false
                matched = true
                i += 1
            default:
                break
            }

            // 漢数字以外に到達したら終了
            var isKanjiNumberChar = false
            switch ch {
            case "十", "百", "千", "万", "億":
                isKanjiNumberChar = true
            default:
                if kanjiDigitValue(ch) != nil {
                    isKanjiNumberChar = true
                }
            }
            if isKanjiNumberChar != true {
                break
            }
        }

        if matched != true {
            return nil
        }

        total += currentSection + currentDigit
        let len = i - start
        return (total, len)
    }

    /// かな数詞スパンを整数にパース
    /// 例: "じゅうご" -> 15, "にせんにじゅうろく" -> 2026, "さんびゃく" -> 300, "せんごひゃく" -> 1500
    private static func parseKanaNumber(chars: [Character], start: Int) -> (value: Int64, length: Int)? {
        let n = chars.count
        if n <= start {
            return nil
        }

        var i = start
        var total: Int64 = 0
        var currentSection: Int64 = 0
        var currentDigit: Int64 = 0
        var hasDigit = false
        var matched = false

        while i < n {
            let remaining = n - i

            // 1. 大位単位: "まん" (万), "おく" (億)
            if 2 <= remaining {
                let sub2 = String(chars[i..<(i + 2)])
                switch sub2 {
                case "まん":
                    let sec = currentSection + currentDigit
                    var s: Int64 = 1
                    if 0 < sec {
                        s = sec
                    }
                    total += s * 10000
                    currentSection = 0
                    currentDigit = 0
                    hasDigit = false
                    matched = true
                    i += 2
                    continue
                case "おく":
                    let sec = currentSection + currentDigit
                    var s: Int64 = 1
                    if 0 < sec {
                        s = sec
                    }
                    total += s * 100000000
                    currentSection = 0
                    currentDigit = 0
                    hasDigit = false
                    matched = true
                    i += 2
                    continue
                default:
                    break
                }
            }

            // 2. 中位単位: "ひゃく", "びゃく", "ぴゃく", "せん", "ぜん", "じゅう", "じゅっ"
            if 3 <= remaining {
                let sub3 = String(chars[i..<(i + 3)])
                switch sub3 {
                case "ひゃく", "びゃく", "ぴゃく":
                    var d: Int64 = 1
                    if hasDigit {
                        d = currentDigit
                    }
                    currentSection += d * 100
                    currentDigit = 0
                    hasDigit = false
                    matched = true
                    i += 3
                    continue
                case "じゅう", "じゅっ":
                    var d: Int64 = 1
                    if hasDigit {
                        d = currentDigit
                    }
                    currentSection += d * 10
                    currentDigit = 0
                    hasDigit = false
                    matched = true
                    i += 3
                    continue
                default:
                    break
                }
            }
            if 2 <= remaining {
                let sub2 = String(chars[i..<(i + 2)])
                switch sub2 {
                case "せん", "ぜん":
                    var d: Int64 = 1
                    if hasDigit {
                        d = currentDigit
                    }
                    currentSection += d * 1000
                    currentDigit = 0
                    hasDigit = false
                    matched = true
                    i += 2
                    continue
                default:
                    break
                }
            }

            // 3. 数字 (0〜9)
            // 3文字: "きゅう"
            if 3 <= remaining {
                let sub3 = String(chars[i..<(i + 3)])
                if sub3 == "きゅう" {
                    if hasDigit {
                        currentDigit = currentDigit * 10 + 9
                    } else {
                        currentDigit = 9
                        hasDigit = true
                    }
                    matched = true
                    i += 3
                    continue
                }
            }
            // 2文字: "ぜろ", "れい", "いち", "さん", "よん", "ろく", "なな", "はち"
            if 2 <= remaining {
                let sub2 = String(chars[i..<(i + 2)])
                var dVal: Int64 = -1
                switch sub2 {
                case "ぜろ", "れい":
                    dVal = 0
                case "いち", "いっ":
                    dVal = 1
                case "さん":
                    dVal = 3
                case "よん":
                    dVal = 4
                case "ろく", "ろっ":
                    dVal = 6
                case "なな", "しち":
                    dVal = 7
                case "はち", "はっ":
                    dVal = 8
                default:
                    dVal = -1
                }
                if 0 <= dVal {
                    if hasDigit {
                        currentDigit = currentDigit * 10 + dVal
                    } else {
                        currentDigit = dVal
                        hasDigit = true
                    }
                    matched = true
                    i += 2
                    continue
                }
            }
            // 1文字: "に", "し", "ご", "く"
            if 1 <= remaining {
                let ch = chars[i]
                var dVal: Int64 = -1
                switch ch {
                case "に":
                    dVal = 2
                case "し":
                    dVal = 4
                case "ご":
                    dVal = 5
                case "く":
                    dVal = 9
                default:
                    dVal = -1
                }
                if 0 <= dVal {
                    // 単一の「に」「し」「く」は助詞等の誤判定を防ぐため、
                    // 直後に単位や後続の数詞が続く場合のみ数値と認める
                    let nextPos = i + 1
                    var hasNextUnit = false
                    if nextPos < n {
                        let nextRemain = n - nextPos
                        if 2 <= nextRemain {
                            let nextSub2 = String(chars[nextPos..<(nextPos + 2)])
                            switch nextSub2 {
                            case "じゅ", "ひゃ", "びゃ", "ぴゃ", "せん", "ぜん", "まん", "えん", "ねん", "がつ", "にち", "ほん", "ぼん", "ぽん", "にん", "かい":
                                hasNextUnit = true
                            default:
                                break
                            }
                        }
                        let nextCh = chars[nextPos]
                        switch nextCh {
                        case "円", "年", "月", "日", "時", "分", "秒", "個", "本", "人", "回", "度", "％", "%", "こ", "つ", "十", "百", "千", "万", "億", "番":
                            hasNextUnit = true
                        case "じ": // 時
                            hasNextUnit = true
                        default:
                            break
                        }
                    }
                    if hasNextUnit || matched {
                        if hasDigit {
                            currentDigit = currentDigit * 10 + dVal
                        } else {
                            currentDigit = dVal
                            hasDigit = true
                        }
                        matched = true
                        i += 1
                        continue
                    }
                }
            }

            // 一致する数詞がない場合はループ終了
            break
        }

        if matched != true {
            return nil
        }

        total += currentSection + currentDigit
        let len = i - start
        return (total, len)
    }

    /// テキスト全体に対して逆テキスト正規化を適用
    public func normalize(_ text: String) -> String {
        if text.isEmpty {
            return ""
        }

        let chars = Array(text)
        let n = chars.count

        var result = ""
        result.reserveCapacity(n)

        var i = 0
        while i < n {
            let remaining = n - i

            // -------------------------------------------------------------
            // 1. 電話番号プレフィックス判定 (例: "ぜろきゅうぜろ" -> "090-", "〇九〇" -> "090-")
            // -------------------------------------------------------------
            var phoneMatched = false
            for pref in Self.phonePrefixReadings {
                let pLen = pref.reading.count
                if pLen <= remaining {
                    let sub = String(chars[i..<(i + pLen)])
                    if sub == pref.reading {
                        var allowPhone = true
                        if pref.reading == "ぜろさん" {
                            let nextIdx = i + pLen
                            if nextIdx < n {
                                let nextCh = chars[nextIdx]
                                switch nextCh {
                                case "年", "月", "日", "時", "分", "秒", "円", "ね", "が", "に":
                                    allowPhone = false
                                default:
                                    break
                                }
                            }
                        }
                        if allowPhone {
                            result.append(pref.replacement)
                            i += pLen
                            phoneMatched = true
                            break
                        }
                    }
                }
            }
            if phoneMatched {
                continue
            }

            // -------------------------------------------------------------
            // 2. 特殊日付（和風月日）の判定 (例: "むいか" -> "6日", "ついたち" -> "1日")
            // -------------------------------------------------------------
            var specialDayMatched = false
            for sDay in Self.specialDayReadings {
                let dLen = sDay.reading.count
                if dLen <= remaining {
                    let sub = String(chars[i..<(i + dLen)])
                    if sub == sDay.reading {
                        let isPrecededByMonth = result.hasSuffix("月")
                        var allowConvert = false

                        switch sDay.reading {
                        case "むいか", "ついたち", "ここのか", "はつか", "にじゅうよっか", "じゅうよっか":
                            allowConvert = true
                        case "いつか", "なのか", "ようか":
                            if isPrecededByMonth {
                                allowConvert = true
                            }
                        case "ふつか", "みっか", "よっか", "とおか":
                            if isPrecededByMonth {
                                allowConvert = true
                            } else {
                                let nextIdx = i + dLen
                                if nextIdx == n {
                                    allowConvert = true
                                } else {
                                    let nextCh = chars[nextIdx]
                                    switch nextCh {
                                    case "に", "で", "は", "も", "の", "か", "間", "、", "。", " ", "　", "\n":
                                        allowConvert = true
                                    default:
                                        break
                                    }
                                }
                            }
                        default:
                            break
                        }

                        if allowConvert {
                            result.append("\(sDay.day)日")
                            i += dLen
                            specialDayMatched = true
                            break
                        }
                    }
                }
            }
            if specialDayMatched {
                continue
            }

            // -------------------------------------------------------------
            // 3. 月のかな読み判定 (例: "きがつ" / "くがつ" -> "9月", "じゅうにがつ" -> "12月")
            // -------------------------------------------------------------
            var monthMatched = false
            for mReading in Self.monthReadings {
                let mLen = mReading.reading.count
                if mLen <= remaining {
                    let sub = String(chars[i..<(i + mLen)])
                    if sub == mReading.reading {
                        var allowMonth = true
                        if sub == "きがつ" {
                            // 「きがつ」は「気がつく」「気がついた」等の一般動詞表現が存在するため、
                            // 直前が年（"年"）であるか、または直後に日表現（むいか、ついたち、漢数字等）が続く場合のみ9月とみなす
                            let isPrecededByYear = result.hasSuffix("年")
                            var isFollowedByDay = false
                            let nextIdx = i + mLen
                            let nextRemain = n - nextIdx
                            if 0 < nextRemain {
                                for sDay in Self.specialDayReadings {
                                    if sDay.reading.count <= nextRemain {
                                        let daySub = String(chars[nextIdx..<(nextIdx + sDay.reading.count)])
                                        if daySub == sDay.reading {
                                            isFollowedByDay = true
                                            break
                                        }
                                    }
                                }
                                let nextCh = chars[nextIdx]
                                if Self.kanjiDigitValue(nextCh) != nil {
                                    isFollowedByDay = true
                                }
                            }
                            if isPrecededByYear != true && isFollowedByDay != true {
                                allowMonth = false
                            }
                        }

                        if allowMonth {
                            result.append("\(mReading.month)月")
                            i += mLen
                            monthMatched = true
                            break
                        }
                    }
                }
            }
            if monthMatched {
                continue
            }

            // -------------------------------------------------------------
            // 4. 分のかな読み判定 (例: "さんじっぷん" -> "30分", "じゅうごふん" -> "15分")
            // -------------------------------------------------------------
            var minuteMatched = false
            for minReading in Self.minuteReadings {
                let minLen = minReading.reading.count
                if minLen <= remaining {
                    let sub = String(chars[i..<(i + minLen)])
                    if sub == minReading.reading {
                        result.append("\(minReading.minute)分")
                        i += minLen
                        minuteMatched = true
                        break
                    }
                }
            }
            if minuteMatched {
                continue
            }

            // 直前の文字が単位（日、月、年、時、分、秒、円、人、回、度、個、本、つ、番）の場合、
            // 「に」は格助詞（例:「6日に15円」）であるため数値パースを開始しない
            var isParticleNi = false
            if chars[i] == "に" {
                if let lastChar = result.last {
                    switch lastChar {
                    case "日", "月", "年", "時", "分", "秒", "円", "人", "回", "度", "個", "本", "つ", "％", "%", "番":
                        isParticleNi = true
                    default:
                        break
                    }
                }
            }
            if isParticleNi {
                result.append(chars[i])
                i += 1
                continue
            }

            // -------------------------------------------------------------
            // 5. 漢数字またはかな数詞のパース
            // -------------------------------------------------------------
            var parsedNumber: (value: Int64, length: Int)? = nil

            // 漢数字パース試行
            if let kanjiNum = Self.parseKanjiNumber(chars: chars, start: i) {
                // 1文字の漢数字「一」〜「九」が単独で文脈なしに現れた場合、名詞や熟語の一部である可能性
                // (例: "一度", "一番", "一般的な") を考慮し、後続に助数詞・単位があるか確認
                let nextIdx = i + kanjiNum.length
                var isValid = true
                if kanjiNum.length == 1 && kanjiNum.value < 10 {
                    if nextIdx < n {
                        let nextCh = chars[nextIdx]
                        switch nextCh {
                        case "円", "年", "月", "日", "時", "分", "秒", "個", "本", "人", "回", "度", "％", "%", "倍", "番":
                            isValid = true
                        default:
                            isValid = false
                        }
                    } else {
                        isValid = true
                    }
                }
                if isValid {
                    parsedNumber = kanjiNum
                }
            }

            // かな数詞パース試行
            if parsedNumber == nil {
                if let kanaNum = Self.parseKanaNumber(chars: chars, start: i) {
                    parsedNumber = kanaNum
                }
            }

            // 数値がパースできた場合
            if let numInfo = parsedNumber {
                let numValue = numInfo.value
                let numLen = numInfo.length
                let nextIdx = i + numLen
                let afterRemain = n - nextIdx

                var handledUnit = false

                // A. パーセント: "ぱーせんと", "パーセント", "％", "%"
                if 5 <= afterRemain {
                    let sub5 = String(chars[nextIdx..<(nextIdx + 5)])
                    if sub5 == "ぱーせんと" || sub5 == "パーセント" {
                        result.append("\(numValue)%")
                        i = nextIdx + 5
                        handledUnit = true
                    }
                }
                if handledUnit != true && 1 <= afterRemain {
                    let sub1 = String(chars[nextIdx..<(nextIdx + 1)])
                    if sub1 == "％" || sub1 == "%" {
                        result.append("\(numValue)%")
                        i = nextIdx + 1
                        handledUnit = true
                    }
                }

                // B. 通貨 (円): "えん", "円"
                if handledUnit != true && 2 <= afterRemain {
                    let sub2 = String(chars[nextIdx..<(nextIdx + 2)])
                    if sub2 == "えん" {
                        result.append("\(numValue)円")
                        i = nextIdx + 2
                        handledUnit = true
                    }
                }
                if handledUnit != true && 1 <= afterRemain {
                    let ch = chars[nextIdx]
                    if ch == "円" {
                        result.append("\(numValue)円")
                        i = nextIdx + 1
                        handledUnit = true
                    }
                }

                // C. 年: "ねん", "年"
                if handledUnit != true && 2 <= afterRemain {
                    let sub2 = String(chars[nextIdx..<(nextIdx + 2)])
                    if sub2 == "ねん" {
                        result.append("\(numValue)年")
                        i = nextIdx + 2
                        handledUnit = true
                    }
                }
                if handledUnit != true && 1 <= afterRemain {
                    let ch = chars[nextIdx]
                    if ch == "年" {
                        result.append("\(numValue)年")
                        i = nextIdx + 1
                        handledUnit = true
                    }
                }

                // D. 月: "がつ", "月"
                if handledUnit != true && 2 <= afterRemain {
                    let sub2 = String(chars[nextIdx..<(nextIdx + 2)])
                    if sub2 == "がつ" {
                        result.append("\(numValue)月")
                        i = nextIdx + 2
                        handledUnit = true
                    }
                }
                if handledUnit != true && 1 <= afterRemain {
                    let ch = chars[nextIdx]
                    if ch == "月" {
                        result.append("\(numValue)月")
                        i = nextIdx + 1
                        handledUnit = true
                    }
                }

                // E. 日: "にち", "日"
                if handledUnit != true && 2 <= afterRemain {
                    let sub2 = String(chars[nextIdx..<(nextIdx + 2)])
                    if sub2 == "にち" {
                        result.append("\(numValue)日")
                        i = nextIdx + 2
                        handledUnit = true
                    }
                }
                if handledUnit != true && 1 <= afterRemain {
                    let ch = chars[nextIdx]
                    if ch == "日" {
                        result.append("\(numValue)日")
                        i = nextIdx + 1
                        handledUnit = true
                    }
                }

                // F. かな助数詞 (ほん/ぼん/ぽん -> 本, にん -> 人, かい -> 回, さい -> 歳)
                if handledUnit != true && 2 <= afterRemain {
                    let sub2 = String(chars[nextIdx..<(nextIdx + 2)])
                    switch sub2 {
                    case "ほん", "ぼん", "ぽん":
                        result.append("\(numValue)本")
                        i = nextIdx + 2
                        handledUnit = true
                    case "にん":
                        result.append("\(numValue)人")
                        i = nextIdx + 2
                        handledUnit = true
                    case "かい":
                        result.append("\(numValue)回")
                        i = nextIdx + 2
                        handledUnit = true
                    case "さい":
                        result.append("\(numValue)歳")
                        i = nextIdx + 2
                        handledUnit = true
                    default:
                        break
                    }
                }

                // G. 1文字助数詞 (こ -> 個, つ -> つ)
                if handledUnit != true && 1 <= afterRemain {
                    let ch = chars[nextIdx]
                    switch ch {
                    case "こ":
                        result.append("\(numValue)個")
                        i = nextIdx + 1
                        handledUnit = true
                    case "つ":
                        result.append("\(numValue)つ")
                        i = nextIdx + 1
                        handledUnit = true
                    default:
                        break
                    }
                }

                // H. 漢字助数詞 (時、分、秒、個、本、人、回、度、倍、番)
                if handledUnit != true && 1 <= afterRemain {
                    let ch = chars[nextIdx]
                    switch ch {
                    case "時", "分", "秒", "個", "本", "人", "回", "度", "倍", "番":
                        result.append("\(numValue)\(ch)")
                        i = nextIdx + 1
                        handledUnit = true
                    default:
                        break
                    }
                }
                if handledUnit != true && 1 <= afterRemain && chars[nextIdx] == "じ" {
                    result.append("\(numValue)時")
                    i = nextIdx + 1
                    handledUnit = true
                }

                // I. 単位がない場合: 10以上の数値であればアラビア数字化 (「山田さん」「いち」等の単独数字は保護)
                if handledUnit != true {
                    if 10 <= numValue {
                        result.append("\(numValue)")
                        i = nextIdx
                    } else {
                        // 単独の1桁（漢数字またはかな数詞）で単位がない場合はそのまま通す
                        result.append(chars[i])
                        i += 1
                    }
                }

                continue
            }

            // 通常文字の追加
            result.append(chars[i])
            i += 1
        }

        return result
    }
}
