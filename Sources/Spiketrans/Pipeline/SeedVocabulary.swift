import Foundation

/// 固有名詞・外来語の読みと表記。第1段の発音変換と第2段のかな漢字で同じ表を使う。
public struct SeedVocabulary: Sendable {
    /// かな漢字変換用シードエントリ一覧
    public static let entries: [KanaKanjiEntry] = [
        // MARK: - 略称・通称 (Colloquial & Abbreviations)
        KanaKanjiEntry(reading: "えあびー", surface: "AirBnB", frequency: 100),
        KanaKanjiEntry(reading: "えあびーあんどびー", surface: "AirBnB", frequency: 100),
        KanaKanjiEntry(reading: "えあびーえぬびー", surface: "AirBnB", frequency: 100),
        KanaKanjiEntry(reading: "めすか", surface: "メルカリ", frequency: 100),
        KanaKanjiEntry(reading: "めるかり", surface: "メルカリ", frequency: 100),
        KanaKanjiEntry(reading: "めるかり", surface: "Mercari", frequency: 80),
        KanaKanjiEntry(reading: "みらてぃぶ", surface: "ミラティブ", frequency: 100),
        KanaKanjiEntry(reading: "みらてぃぶ", surface: "Mirrativ", frequency: 100),
        KanaKanjiEntry(reading: "すたば", surface: "スターバックス", frequency: 100),
        KanaKanjiEntry(reading: "すたーばっくす", surface: "スターバックス", frequency: 100),
        KanaKanjiEntry(reading: "まくど", surface: "マクドナルド", frequency: 100),
        KanaKanjiEntry(reading: "まくどなるど", surface: "マクドナルド", frequency: 100),
        KanaKanjiEntry(reading: "こんびに", surface: "コンビニ", frequency: 100),
        KanaKanjiEntry(reading: "ふぁみま", surface: "ファミリーマート", frequency: 100),
        KanaKanjiEntry(reading: "ふぁみりーまーと", surface: "ファミリーマート", frequency: 100),
        KanaKanjiEntry(reading: "ろーそん", surface: "ローソン", frequency: 100),
        KanaKanjiEntry(reading: "せぶん", surface: "セブンイレブン", frequency: 100),
        KanaKanjiEntry(reading: "せぶんいれぶん", surface: "セブンイレブン", frequency: 100),
        KanaKanjiEntry(reading: "ぽけもん", surface: "ポケモン", frequency: 100),
        KanaKanjiEntry(reading: "すまほ", surface: "スマホ", frequency: 100),
        KanaKanjiEntry(reading: "ぱそこん", surface: "パソコン", frequency: 100),
        KanaKanjiEntry(reading: "てれわーく", surface: "テレワーク", frequency: 100),
        KanaKanjiEntry(reading: "りもーと", surface: "リモート", frequency: 100),
        KanaKanjiEntry(reading: "あぷり", surface: "アプリ", frequency: 100),
        KanaKanjiEntry(reading: "ねとふり", surface: "Netflix", frequency: 80),
        KanaKanjiEntry(reading: "ねとふり", surface: "ネットフリックス", frequency: 100),
        KanaKanjiEntry(reading: "いんすた", surface: "Instagram", frequency: 80),
        KanaKanjiEntry(reading: "いんすた", surface: "インスタ", frequency: 100),

        // MARK: - 和英併記エントリ (Bilingual Japanese / English)
        KanaKanjiEntry(reading: "あっぷる", surface: "Apple", frequency: 80),
        KanaKanjiEntry(reading: "あっぷる", surface: "アップル", frequency: 100),
        KanaKanjiEntry(reading: "ぐーぐる", surface: "Google", frequency: 100),
        KanaKanjiEntry(reading: "ぐーぐる", surface: "グーグル", frequency: 100),
        KanaKanjiEntry(reading: "いんたーねっと", surface: "Internet", frequency: 100),
        KanaKanjiEntry(reading: "いんたーねっと", surface: "インターネット", frequency: 100),
        KanaKanjiEntry(reading: "あいふぉーん", surface: "iPhone", frequency: 100),
        KanaKanjiEntry(reading: "あいふぉん", surface: "iPhone", frequency: 100),
        KanaKanjiEntry(reading: "うぇぶ", surface: "Web", frequency: 100),
        KanaKanjiEntry(reading: "うぇぶ", surface: "ウェブ", frequency: 100),
        KanaKanjiEntry(reading: "えーあい", surface: "AI", frequency: 100),
        KanaKanjiEntry(reading: "ぴーしー", surface: "PC", frequency: 100),
        KanaKanjiEntry(reading: "わいふぁい", surface: "WiFi", frequency: 100),
        KanaKanjiEntry(reading: "わいふぁい", surface: "Wi-Fi", frequency: 100),
        KanaKanjiEntry(reading: "てぃっくとっく", surface: "TikTok", frequency: 100),
        KanaKanjiEntry(reading: "てぃっくとっく", surface: "ティックトック", frequency: 100),
        KanaKanjiEntry(reading: "ゆーちゅーぶ", surface: "YouTube", frequency: 100),
        KanaKanjiEntry(reading: "ゆーちゅーぶ", surface: "ユーチューブ", frequency: 100),
        KanaKanjiEntry(reading: "ねっとふりっくす", surface: "Netflix", frequency: 100),
        KanaKanjiEntry(reading: "ねっとふりっくす", surface: "ネットフリックス", frequency: 100),
        KanaKanjiEntry(reading: "おーぷんえーあい", surface: "OpenAI", frequency: 100),
        KanaKanjiEntry(reading: "まいくろそふと", surface: "Microsoft", frequency: 100),
        KanaKanjiEntry(reading: "まいくろそふと", surface: "マイクロソフト", frequency: 100),
        KanaKanjiEntry(reading: "あまぞん", surface: "Amazon", frequency: 100),
        KanaKanjiEntry(reading: "あまぞん", surface: "アマゾン", frequency: 100),
        KanaKanjiEntry(reading: "めた", surface: "Meta", frequency: 100),
        KanaKanjiEntry(reading: "めた", surface: "メタ", frequency: 100),
        KanaKanjiEntry(reading: "ふぇいすぶっく", surface: "Facebook", frequency: 100),
        KanaKanjiEntry(reading: "ふぇいすぶっく", surface: "フェイスブック", frequency: 100),
        KanaKanjiEntry(reading: "いんすたぐらむ", surface: "Instagram", frequency: 100),
        KanaKanjiEntry(reading: "いんすたぐらむ", surface: "インスタグラム", frequency: 100),
        KanaKanjiEntry(reading: "ついったー", surface: "Twitter", frequency: 100),
        KanaKanjiEntry(reading: "ついったー", surface: "ツイッター", frequency: 100),
        KanaKanjiEntry(reading: "うーばー", surface: "Uber", frequency: 100),
        KanaKanjiEntry(reading: "うーばー", surface: "ウーバー", frequency: 100),
        KanaKanjiEntry(reading: "ずーむ", surface: "Zoom", frequency: 100),
        KanaKanjiEntry(reading: "ずーむ", surface: "ズーム", frequency: 100),
        KanaKanjiEntry(reading: "すらっく", surface: "Slack", frequency: 100),
        KanaKanjiEntry(reading: "すらっく", surface: "スラック", frequency: 100),
        KanaKanjiEntry(reading: "らいん", surface: "LINE", frequency: 100),
        KanaKanjiEntry(reading: "らいん", surface: "ライン", frequency: 100),
        KanaKanjiEntry(reading: "やふー", surface: "Yahoo", frequency: 100),
        KanaKanjiEntry(reading: "やふー", surface: "ヤフー", frequency: 100),
        KanaKanjiEntry(reading: "ぎっとはぶ", surface: "GitHub", frequency: 100),
        KanaKanjiEntry(reading: "ぎっと", surface: "Git", frequency: 100),
        KanaKanjiEntry(reading: "のーしょん", surface: "Notion", frequency: 100),
        KanaKanjiEntry(reading: "どっかー", surface: "Docker", frequency: 100),
        KanaKanjiEntry(reading: "あいぱっど", surface: "iPad", frequency: 100),
        KanaKanjiEntry(reading: "まっく", surface: "Mac", frequency: 100),
        KanaKanjiEntry(reading: "まっくぶっく", surface: "MacBook", frequency: 100),
        KanaKanjiEntry(reading: "あんどろいど", surface: "Android", frequency: 100),
        KanaKanjiEntry(reading: "ういんどーず", surface: "Windows", frequency: 100),
        KanaKanjiEntry(reading: "りなっくす", surface: "Linux", frequency: 100),
        KanaKanjiEntry(reading: "ぶるーとぅーす", surface: "Bluetooth", frequency: 100),
        KanaKanjiEntry(reading: "しーぴーゆー", surface: "CPU", frequency: 100),
        KanaKanjiEntry(reading: "じーぴーゆー", surface: "GPU", frequency: 100),
        KanaKanjiEntry(reading: "えーぴーあい", surface: "API", frequency: 100),
        KanaKanjiEntry(reading: "ゆーあい", surface: "UI", frequency: 100),
        KanaKanjiEntry(reading: "ゆーえっくす", surface: "UX", frequency: 100),
        KanaKanjiEntry(reading: "ゆーあーるえる", surface: "URL", frequency: 100),
        KanaKanjiEntry(reading: "おーえす", surface: "OS", frequency: 100),
        KanaKanjiEntry(reading: "あいおーえす", surface: "iOS", frequency: 100),
        KanaKanjiEntry(reading: "あいてぃー", surface: "IT", frequency: 100),
        KanaKanjiEntry(reading: "でぃーえっくす", surface: "DX", frequency: 100),
        KanaKanjiEntry(reading: "えすえぬえす", surface: "SNS", frequency: 100),
        KanaKanjiEntry(reading: "ぴーあーる", surface: "PR", frequency: 100),
        KanaKanjiEntry(reading: "しーいーおー", surface: "CEO", frequency: 100),
        KanaKanjiEntry(reading: "しーてぃーおー", surface: "CTO", frequency: 100),
        KanaKanjiEntry(reading: "さーす", surface: "SaaS", frequency: 100),
        KanaKanjiEntry(reading: "びーとぅーびー", surface: "BtoB", frequency: 100),
        KanaKanjiEntry(reading: "びーとぅーびー", surface: "B-to-B", frequency: 80),
        KanaKanjiEntry(reading: "びーとぅーしー", surface: "BtoC", frequency: 100),
        KanaKanjiEntry(reading: "びーとぅーしー", surface: "B-to-C", frequency: 80),
        KanaKanjiEntry(reading: "すたーばっくす", surface: "Starbucks", frequency: 80),
        KanaKanjiEntry(reading: "まくどなるど", surface: "McDonald", frequency: 80),
        KanaKanjiEntry(reading: "まくどなるど", surface: "McDonalds", frequency: 80),
        KanaKanjiEntry(reading: "えむえる", surface: "ML", frequency: 100),
        KanaKanjiEntry(reading: "でぃーえる", surface: "DL", frequency: 100),
        KanaKanjiEntry(reading: "えるえるえむ", surface: "LLM", frequency: 100),
        KanaKanjiEntry(reading: "じーぴーてぃー", surface: "GPT", frequency: 100),
        KanaKanjiEntry(reading: "ちゃっとじーぴーてぃー", surface: "ChatGPT", frequency: 100),
        KanaKanjiEntry(reading: "ゆーえすびー", surface: "USB", frequency: 100),
        KanaKanjiEntry(reading: "えすえすでぃー", surface: "SSD", frequency: 100),
        KanaKanjiEntry(reading: "えいちでぃーでぃー", surface: "HDD", frequency: 100),
        KanaKanjiEntry(reading: "らむ", surface: "RAM", frequency: 100),
        KanaKanjiEntry(reading: "ろむ", surface: "ROM", frequency: 100),
        KanaKanjiEntry(reading: "えすきゅーえる", surface: "SQL", frequency: 100),
        KanaKanjiEntry(reading: "えいちてぃーえむえる", surface: "HTML", frequency: 100),
        KanaKanjiEntry(reading: "しーえすえす", surface: "CSS", frequency: 100),
        KanaKanjiEntry(reading: "すうぃふと", surface: "Swift", frequency: 100),
        KanaKanjiEntry(reading: "ぱいそん", surface: "Python", frequency: 100),
        KanaKanjiEntry(reading: "らすと", surface: "Rust", frequency: 100),
        KanaKanjiEntry(reading: "ごー", surface: "Go", frequency: 80),
        KanaKanjiEntry(reading: "じゃば", surface: "Java", frequency: 100),
        KanaKanjiEntry(reading: "たいぷすくりぷと", surface: "TypeScript", frequency: 100),
        KanaKanjiEntry(reading: "じゃばすくりぷと", surface: "JavaScript", frequency: 100),

        // MARK: - 第2段デコーダ用: アルファベット単文字エントリ (かな -> 表記 "えー" -> "A")
        // 音響 SNN の CTC デコーダから出力されたかな音素列を、テキスト表記へ復元する
        // ための第2段かな漢字変換候補。Unigram 出現頻度は 50 とする。
        KanaKanjiEntry(reading: "えー", surface: "A", frequency: 50),
        KanaKanjiEntry(reading: "びー", surface: "B", frequency: 50),
        KanaKanjiEntry(reading: "しー", surface: "C", frequency: 50),
        KanaKanjiEntry(reading: "でぃー", surface: "D", frequency: 50),
        KanaKanjiEntry(reading: "いー", surface: "E", frequency: 50),
        KanaKanjiEntry(reading: "えふ", surface: "F", frequency: 50),
        KanaKanjiEntry(reading: "じー", surface: "G", frequency: 50),
        KanaKanjiEntry(reading: "えいち", surface: "H", frequency: 50),
        KanaKanjiEntry(reading: "あい", surface: "I", frequency: 50),
        KanaKanjiEntry(reading: "じぇー", surface: "J", frequency: 50),
        KanaKanjiEntry(reading: "けー", surface: "K", frequency: 50),
        KanaKanjiEntry(reading: "える", surface: "L", frequency: 50),
        KanaKanjiEntry(reading: "えむ", surface: "M", frequency: 50),
        KanaKanjiEntry(reading: "えぬ", surface: "N", frequency: 50),
        KanaKanjiEntry(reading: "おー", surface: "O", frequency: 50),
        KanaKanjiEntry(reading: "ぴー", surface: "P", frequency: 50),
        KanaKanjiEntry(reading: "きゅー", surface: "Q", frequency: 50),
        KanaKanjiEntry(reading: "あーる", surface: "R", frequency: 50),
        KanaKanjiEntry(reading: "えす", surface: "S", frequency: 50),
        KanaKanjiEntry(reading: "てぃー", surface: "T", frequency: 50),
        KanaKanjiEntry(reading: "ゆー", surface: "U", frequency: 50),
        KanaKanjiEntry(reading: "ぶい", surface: "V", frequency: 50),
        KanaKanjiEntry(reading: "だぶりゅー", surface: "W", frequency: 50),
        KanaKanjiEntry(reading: "えっくす", surface: "X", frequency: 50),
        KanaKanjiEntry(reading: "わい", surface: "Y", frequency: 50),
        KanaKanjiEntry(reading: "ぜっと", surface: "Z", frequency: 50)
    ]

    // MARK: - 第1段音響モデル用: 単一アルファベット読み展開テーブル (表記 -> かな "A" -> "えー")
    //
    // 【KanaKanjiEntry との役割・文脈・責務の違い】
    // - KanaKanjiEntry (第2段 言語デコーダ / かな漢字変換用):
    //     方向: 「読み（かな） -> 表記（文字）」
    //     音響モデル (CTC) から出力されたひらがな音素列（例: "えー", "えあびー"）を、
    //     文章表記（"A", "AirBnB"）へと復元するための辞書候補。Unigram 統計頻度を持つ。
    // - letterReadingTable (第1段 音響モデル / 学習時テキスト正規化用):
    //     方向: 「表記（文字） -> 読み（かな）」
    //     教師テキスト中の大文字頭字語（例: "GPU" -> "じーぴーゆー", "SDK" -> "えすでぃーけー"）や
    //     単一英字（"A" -> "えー"）を、音響 SNN が学習可能なひらがな・音素列へと展開するための逆引きテーブル。
    //
    // 同一の概念に対する知識の二重管理を排除するため、entries から動的に導出する。
    private static let letterReadingTable: [String: String] = {
        var table: [String: String] = [:]
        for entry in entries {
            let surface = entry.surface
            if surface.count == 1, let scalar = surface.unicodeScalars.first {
                let v = scalar.value
                switch v {
                case 0x41...0x5A, 0x61...0x7A:
                    let upperKey = String(scalar).uppercased()
                    if table[upperKey] == nil {
                        table[upperKey] = entry.reading
                    }
                default:
                    break
                }
            }
        }
        return table
    }()

    // MARK: - 第1段音響モデル用: 英単語・略称の代表読みテーブル (表記 -> かな "wifi" -> "わいふぁい")
    //
    // 【KanaKanjiEntry との役割・文脈・責務の違い】
    // - 知識の実体 (Single Source of Truth):
    //     語彙の定義データ（英単語・カタカナ語・略称・読み・頻度）は、100% すべて上記の
    //     `entries` ([KanaKanjiEntry]) にのみ一元定義される。
    // - 第2段 (言語デコーダ / かな漢字変換) での役割:
    //     KanaKanjiDictionary にロードされ、音声認識 (CTC) が出力したひらがな音素列から
    //     元の正書法（例: "あっぷる" -> "Apple", "わいふぁい" -> "Wi-Fi"）を復元する変換候補として機能する。
    // - 第1段 (音響モデル / KanjiConverter) での役割:
    //     教師データ中の英単語（例: "Apple", "Wi-Fi"）がローマ字変換規則によって「あっぷれ」「うぃーふぃ」
    //     のように不正に音素崩れを起こすのを防ぐため、代表ひらがな読みに置換する逆引き辞書が必要となる。
    //
    // したがって、alphabetReadingTable は独立した語彙テーブルではなく、`entries` から
    // 英字を含む語彙を抽出し、小文字化・全角半角正規化キーで O(1) ルックアップ可能にした
    // 「第1段用の転置インデックス（キャッシュ）」である。
    // ハードコードされた初期値を持たず、entries のみを真実の情報源として動的に自動導出される。
    private static let alphabetReadingTable: [String: String] = {
        var table: [String: String] = [:]
        var maxFreq: [String: Int] = [:]
        for entry in entries {
            let key = entry.surface.folding(options: .widthInsensitive, locale: nil).lowercased()
            var isAlpha = false
            for s in key.unicodeScalars {
                if 0x61 <= s.value && s.value <= 0x7A {
                    isAlpha = true
                    break
                }
            }
            if isAlpha {
                let currentMax = maxFreq[key] ?? -1
                if currentMax < entry.frequency {
                    maxFreq[key] = entry.frequency
                    table[key] = entry.reading
                }
            }
        }
        return table
    }()

    /// アルファベット単文字の読みを取得 (全角・半角両対応)
    public static func readingForLetter(_ letter: String) -> String? {
        let key = letter.folding(options: .widthInsensitive, locale: nil).uppercased()
        return letterReadingTable[key]
    }

    /// 英単語・略称の表層文字列から対応するかな読みを取得 (全角・半角、大文字・小文字両対応)
    public static func readingForAlphabetSurface(_ surface: String) -> String? {
        let key = surface.folding(options: .widthInsensitive, locale: nil).lowercased()
        return alphabetReadingTable[key]
    }
}
