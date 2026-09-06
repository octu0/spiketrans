import Foundation

/// フィラーフィルタの動作モード
public enum FillerFilterMode: Sendable, Equatable {
    case remove       // フィラーを完全に除去
    case mark         // フィラーをマーキング (例: "(えー)")
    case disabled     // フィルタリングを行わない
}

/// 日本語特有のフィラー・言い淀みの検出および除去フィルタ
///
/// 「えー」「あー」「えっと」「あのー」「そのー」「まあ」「なんか」「うーん」等の
/// 音声認識結果に頻出する言い淀みを検出し、除去または整形する。
/// SNN の軽量性を損なわないよう、外部依存ゼロ・最小限のアロケーションで動作する。
public struct FillerWordFilter: Sendable {
    public let mode: FillerFilterMode

    public init(mode: FillerFilterMode = .remove) {
        self.mode = mode
    }

    /// 波ダッシュ等の引き伸ばし記号を長音符「ー」に正規化
    private static func normalizeProlongedMarks(_ text: String) -> String {
        var result = ""
        result.reserveCapacity(text.count)
        for scalar in text.unicodeScalars {
            switch scalar.value {
            case 0x301C, // WAVE DASH (〜)
                 0xFF5E, // FULLWIDTH TILDE (～)
                 0x3030, // WAVY DASH (〰)
                 0x223C, // TILDE OPERATOR (∼)
                 0xFF70: // HALFWIDTH KATAKANA-HIRAGANA PROLONGED SOUND MARK (ｰ)
                result.append("ー")
            default:
                result.unicodeScalars.append(scalar)
            }
        }
        return result
    }

    /// 区切り文字（句読点、空白、記号）であるか判定
    private static func isDelimiter(_ ch: Character) -> Bool {
        switch ch {
        case "、", "。", "，", "．", " ", "　", "\t", "\n", "\r",
             "！", "？", "!", "?", "…", "・", "-", "ー":
            return true
        default:
            return false
        }
    }

    /// 漢字であるか判定 (CJK統合漢字)
    private static func isKanji(_ ch: Character) -> Bool {
        for scalar in ch.unicodeScalars {
            if 0x4E00 <= scalar.value && scalar.value <= 0x9FFF {
                return true
            }
            if 0x3400 <= scalar.value && scalar.value <= 0x4DBF {
                return true
            }
        }
        return false
    }

    /// 単語がフィラー単体であるか判定
    public func isFiller(_ text: String) -> Bool {
        let normalized = Self.normalizeProlongedMarks(text.trimmingCharacters(in: .whitespacesAndNewlines))
        if normalized.isEmpty {
            return false
        }

        switch normalized {
        case "えー", "えーー", "えーーー", "ええと", "えっと", "えっとー", "えーと", "えーっと", "ええ":
            return true
        case "あー", "あーー", "あーーー", "あのー", "あのーー", "あのね", "あのーっ", "あーっと", "あっ":
            return true
        case "そのー", "そのーー", "そのう", "そのーっ":
            return true
        case "うーん", "うーーん", "うーんと", "うむ", "んー", "んーと":
            return true
        case "まあ", "まー", "まーっ":
            return true
        case "なんか", "なんかー", "なんというか":
            return true
        case "あの", "その":
            return true
        case "ほら", "ほらー":
            return true
        default:
            return false
        }
    }

    /// テキスト中のフィラーを検出・除去（またはマーキング）する
    public func filter(_ text: String) -> String {
        switch mode {
        case .disabled:
            return text
        case .remove, .mark:
            break
        }

        if text.isEmpty {
            return ""
        }

        let normalized = Self.normalizeProlongedMarks(text)
        let chars = Array(normalized)
        let n = chars.count

        var output = ""
        output.reserveCapacity(n)

        var i = 0
        while i < n {
            var matchedLength = 0
            var matchedText = ""

            // 最長一致でフィラー候補を走査
            // 候補長: 最大 6 文字 ("なんというか", "えーっと" など)
            let maxCheckLen = min(6, n - i)
            var l = maxCheckLen
            while 1 <= l {
                let subStr = String(chars[i..<(i + l)])
                var isCandidate = false
                var requireDelimiterOrEnd = false

                switch subStr {
                case "なんというか", "えーっと", "あーっと", "うーんと", "えっとー", "あのねー":
                    isCandidate = true
                case "えーと", "ええと", "えっと", "あのー", "そのー", "うーん", "なんかー", "ほらー":
                    isCandidate = true
                case "えー", "あー", "そのう", "まーっ", "あのーっ", "そのーっ":
                    isCandidate = true
                case "うむ", "んー":
                    isCandidate = true
                case "あの", "その":
                    // 「あの」「その」は直後に句読点・空白・文末がある場合のみフィラーとみなす (「あの人」等の誤消去防止)
                    isCandidate = true
                    requireDelimiterOrEnd = true
                case "まあ", "まー":
                    isCandidate = true
                case "なんか":
                    isCandidate = true
                default:
                    isCandidate = false
                }

                if isCandidate {
                    var isValid = true
                    let nextIdx = i + l

                    if requireDelimiterOrEnd {
                        if nextIdx < n {
                            let nextCh = chars[nextIdx]
                            if Self.isDelimiter(nextCh) != true {
                                isValid = false
                            }
                        }
                    }

                    // 「あー」「えー」「まー」が「あーと」「まーけっと」「えーす」等の語の一部でないことを確認
                    if isValid {
                        switch subStr {
                        case "あー", "えー", "まー":
                            if nextIdx < n {
                                let nextCh = chars[nextIdx]
                                // 直後が区切り・漢字・長音以外（平仮名・カタカナ等）で、フィラー継続でもない場合は語の一部と判定
                                if Self.isDelimiter(nextCh) != true && Self.isKanji(nextCh) != true && nextCh != "ー" {
                                    // ただし「あーえっと」のように後続もフィラー開始語句である場合は許容
                                    let nextRemain = n - nextIdx
                                    var followedByFiller = false
                                    if 2 <= nextRemain {
                                        let nextSub2 = String(chars[nextIdx..<(nextIdx + 2)])
                                        switch nextSub2 {
                                        case "えっ", "あの", "その", "えー", "うー":
                                            followedByFiller = true
                                        default:
                                            break
                                        }
                                    }
                                    if followedByFiller != true {
                                        isValid = false
                                    }
                                }
                            }
                        case "まあ":
                            if nextIdx < n {
                                let nextCh = chars[nextIdx]
                                // 「まあまあ」や平仮名名詞結合の誤消去を防止（区切り・漢字・文末・読点の場合はフィラー認定）
                                if Self.isDelimiter(nextCh) != true && Self.isKanji(nextCh) != true {
                                    isValid = false
                                }
                            }
                        case "なんか":
                            if nextIdx < n {
                                let nextCh = chars[nextIdx]
                                // 「何かが」「何かを」等の不定代名詞＋格助詞結合の誤消去を防止
                                switch nextCh {
                                case "が", "を", "に", "で", "と", "の":
                                    isValid = false
                                default:
                                    // 区切りや漢字、文末でない場合はフィラーとみなさない
                                    if Self.isDelimiter(nextCh) != true && Self.isKanji(nextCh) != true {
                                        isValid = false
                                    }
                                }
                            }
                        default:
                            break
                        }
                    }

                    if isValid {
                        // 連続する長音「ー」もフィラーに吸収 (例: "えーーー" -> "えー")
                        var ext = nextIdx
                        while ext < n {
                            if chars[ext] == "ー" {
                                ext += 1
                            } else {
                                break
                            }
                        }
                        matchedLength = ext - i
                        matchedText = String(chars[i..<ext])
                        break
                    }
                }

                l -= 1
            }

            if 0 < matchedLength {
                switch mode {
                case .remove:
                    // 除去モード: もし直前の文字が読点「、」なら、フィラーに先行する読点も削除
                    if output.hasSuffix("、") {
                        output.removeLast()
                    }
                case .mark:
                    // マークモード: 括弧で囲む
                    output.append("(\(matchedText))")
                case .disabled:
                    ()
                }

                i += matchedLength

                // フィラー直後の読点「、」や空白もスキップして自然に接続
                while i < n {
                    let c = chars[i]
                    if c == "、" || c == " " || c == "　" {
                        i += 1
                    } else {
                        break
                    }
                }
            } else {
                output.append(chars[i])
                i += 1
            }
        }

        // 先頭に残った不要な読点「、」のサニタイズ
        var trimmed = output
        while trimmed.isEmpty != true {
            switch trimmed.first {
            case .some(let ch):
                if ch == "、" || ch == " " || ch == "　" {
                    trimmed.removeFirst()
                } else {
                    break
                }
            case .none:
                break
            }
            // 削除が行われなかった場合は脱出
            if let first = trimmed.first {
                if first != "、" && first != " " && first != "　" {
                    break
                }
            }
        }

        return trimmed
    }
}
