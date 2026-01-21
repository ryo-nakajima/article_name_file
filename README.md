# PDF Rename Tool for Academic Papers

学術論文PDFを自動的に `著者名_出版年.pdf` 形式にリネームするツール。

## 目的

学術論文のPDFファイルは、ダウンロード時に意味のないファイル名（例: `001173108.pdf`）や不完全な名前になっていることが多い。このツールは:

1. PDFからタイトルを自動抽出
2. タイトルでWeb検索して著者名・出版年を取得
3. `Author_Year.pdf` 形式に統一リネーム

これにより、ファイル名だけで論文を識別できるようになる。

## 機能

### タイトル抽出
- **テキスト抽出**: pdfplumberでPDFテキストを解析
- **フォント解析**: 最大フォントサイズの行をタイトル候補として検出
- **GPT検証**: OpenAI APIでタイトルの妥当性を検証
- **OCRフォールバック**: テキスト抽出失敗時はpytesseractでOCR処理
- **JSTORボイラープレート検出**: JSTOR著作権表示をタイトルとして誤検出するのを防止

### 著者名検索
- **Semantic Scholar API**: 学術論文データベースで検索（優先）
- **CrossRef API**: Semantic Scholarで見つからない場合のフォールバック
- **GPT検証**: 著者名が実際の姓かどうかを検証（NON_SURNAMESリストとの照合）

### ファイル管理
- **インクリメンタル処理**: MD5ハッシュベースで処理済みファイルをスキップ
- **重複検出**: 同一内容のファイルを自動検出して`duplicate/`へ移動
- **パス衝突解決**: 同名ファイルが存在する場合、`_a`, `_b`等のサフィックスを追加
- **カテゴリ分類**: 日本語論文、処理失敗、OCR必要ファイルを自動分類

## 必要環境

### Python パッケージ
```bash
pip install pdfplumber pypdf pytesseract pdf2image openai requests jupytext
```

### 外部ツール
- **Tesseract OCR**: OCR処理用
  ```bash
  # macOS
  brew install tesseract tesseract-lang

  # Ubuntu
  sudo apt install tesseract-ocr tesseract-ocr-jpn
  ```

- **Poppler**: pdf2image用
  ```bash
  # macOS
  brew install poppler

  # Ubuntu
  sudo apt install poppler-utils
  ```

### 環境変数
```bash
export OPENAI_API_KEY="your-api-key"
```

## ディレクトリ構成

```
/Users/ryo2/Dropbox/Articles/     # 作業ディレクトリ（PDFファイル配置場所）
├── README.md                      # このファイル
├── CLAUDE.md                      # Claude Code用ガイダンス
├── .gitignore
├── code/
│   ├── pdf_rename_websearch.py    # メインワークフロー（編集用）
│   ├── pdf_rename_websearch.ipynb # Jupyterノートブック（実行用）
│   └── data/                      # キャッシュファイル
│       ├── pdf_processing_progress.json  # 処理進捗
│       ├── pdf_rename_data.json          # 全データ
│       └── pdf_rename_summary.json       # サマリー
├── japanese/                      # 日本語論文
├── failure/                       # 処理失敗ファイル
├── re-search/                     # OCR処理ファイル（要確認）
└── duplicate/                     # 重複ファイル
```

## 実行方法

### 1. 準備

PDFファイルを作業ディレクトリ（`/Users/ryo2/Dropbox/Articles/`）に配置。

### 2. Jupyter Notebookで実行

```bash
cd /Users/ryo2/Dropbox/Articles/code
jupyter notebook pdf_rename_websearch.ipynb
```

セルを順番に実行:

| セル | 処理内容 |
|------|----------|
| Cell 1-2 | セットアップ、インポート |
| Cell 3-3b | ヘルパー関数、GPT関数 |
| Cell 4-5 | PDF解析・WebSearch関数定義 |
| Cell 5.5 | JSTORボイラープレートリセット |
| Cell 6 | **Step 1**: タイトル・メタデータ抽出 |
| Cell 7 | **Step 2**: WebSearchで著者名取得 |
| Cell 8 | **Step 3**: 結果比較、最終著者・年決定 |
| Cell 9-9.6 | **Step 4**: PDFテキスト検索、OCR+GPTリカバリ |
| Cell 10 | ファイル名生成 |
| Cell 11 | プレビュー確認 |
| Cell 12 | **リネーム実行** |
| Cell 13 | 結果保存 |
| Cell 14 | 重複ファイル検出・移動 |

### 3. 再処理（キャッシュクリア）

全ファイルを最初から処理し直す場合:

```bash
# サブフォルダのPDFを戻す
cd /Users/ryo2/Dropbox/Articles
for dir in duplicate re-search failure japanese; do
  find "$dir" -name "*.pdf" -exec mv {} . \; 2>/dev/null
done

# キャッシュクリア
rm -f code/data/*.json

# ノートブック実行
```

### 4. Pythonファイル編集後の同期

`pdf_rename_websearch.py`を編集した場合、ノートブックに反映:

```bash
cd code
jupytext --to notebook pdf_rename_websearch.py -o pdf_rename_websearch.ipynb
```

## 処理フロー詳細

```
PDFファイル
    │
    ▼
┌─────────────────────────────────────┐
│ Step 1: タイトル抽出 (Cell 6)        │
│  - pdfplumberでテキスト抽出          │
│  - フォントサイズでタイトル検出       │
│  - GPTで検証                         │
│  - 失敗時はOCR                       │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Step 2: WebSearch (Cell 7)          │
│  - Semantic Scholar APIで検索        │
│  - 失敗時はCrossRef APIにフォールバック│
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Step 3: 結果統合 (Cell 8)           │
│  - WebSearch著者 vs メタデータ著者    │
│  - NON_SURNAMESチェック              │
│  - 出版年決定                        │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Step 4: フォールバック (Cell 9)      │
│  - alertファイルのPDFテキスト検索    │
│  - failファイルのOCR+GPTリカバリ     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ ファイル名生成・リネーム (Cell 10-12) │
│  - Author_Year.pdf 形式生成          │
│  - パス衝突解決                      │
│  - 重複検出・移動                    │
└─────────────────────────────────────┘
    │
    ▼
結果:
  ├── success → リネーム完了
  ├── japanese → japanese/ へ移動
  ├── fail → failure/ へ移動
  ├── ocr使用 → re-search/ へ移動（要確認）
  └── 重複 → duplicate/ へ移動
```

## ファイル名規則

### 基本形式
```
著者名_出版年.pdf
```

### 著者名パターン
| パターン | 例 |
|----------|-----|
| 単著 | `Abadie_2005.pdf` |
| 2著者 | `Abadie_Imbens_2010.pdf` |
| 3著者 | `Abadie_Diamond_Hainmueller_2010.pdf` |
| 4著者以上 | `Abadie_et_al_2015.pdf` |

### サフィックス
| サフィックス | 意味 |
|-------------|------|
| `_a`, `_b` | 同一著者・年の複数論文 |
| `_wp` | ワーキングペーパー版（重複検出時） |

## ステータス一覧

| ステータス | 説明 |
|-----------|------|
| `success` | 正常にリネーム完了 |
| `alert` | 低信頼度でリネーム（OCR+GPTフォールバック使用） |
| `fail` | 処理失敗（著者名・出版年取得不可） |
| `japanese` | 日本語論文 |

## トラブルシューティング

### 問題: 一部のファイルが処理されない
**原因**: 同一内容（ハッシュ一致）のファイルが複数存在
**解決**: Cell 12で自動的に`duplicate/`へ移動される

### 問題: ファイル名が変わらない
**原因**: キャッシュに以前の結果が残っている
**解決**: `rm -f code/data/*.json` でキャッシュクリア

### 問題: OCRが遅い
**原因**: Tesseract OCRは処理に時間がかかる
**解決**: OCR対象ファイルは最小限に抑えられている。`re-search/`のファイルは手動確認推奨

### 問題: API制限エラー
**原因**: Semantic Scholar / CrossRef のレート制限
**解決**: 自動リトライ機能あり。それでも失敗する場合は時間をおいて再実行

## ライセンス

Private use only.
