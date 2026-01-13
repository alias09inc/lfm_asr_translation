# LFM2.5-Audio リアルタイム ASR + EN→JP 翻訳

LiquidAI の LFM2.5-Audio を使って音声をリアルタイムに文字起こしし、英語テキストを日本語に翻訳する Gradio WebRTC アプリです。

## できること

- マイク入力をリアルタイムに ASR（英語想定）してテキスト表示
- セグメント確定ごとに日本語翻訳を追記
- Gradio UI でブラウザから操作（ローカル or 共有リンク）

## 必要環境

- Python 3.12 以上
- GPU 推奨（CPU でも動きますが遅くなる可能性があります）
- 初回起動時に Hugging Face からモデルをダウンロードします

## セットアップ

```bash
uv sync
```

## 起動方法

リポジトリ直下で実行します。

```bash
uv run main.py
```

起動後、`http://127.0.0.1:7860` にアクセスしてマイクを許可してください。

## オプション

```bash
uv run main.py \
  --asr-repo LiquidAI/LFM2.5-Audio-1.5B \
  --mt-repo LiquidAI/LFM2-350M-ENJP-MT \
  --device cuda \
  --mt-device cuda \
  --host 127.0.0.1 \
  --port 7860 \
  --share \
  --max-new-tokens 256 \
  --mt-max-new-tokens 256 \
  --max-segment-s 10.0
```

主なオプション:

- `--asr-repo`: ASR に使う Hugging Face リポジトリ（既定: `LiquidAI/LFM2.5-Audio-1.5B`）
- `--mt-repo`: 翻訳モデルのリポジトリ（既定: `LiquidAI/LFM2-350M-ENJP-MT`）
- `--device`: ASR のデバイス指定（`cuda` または `cpu`）
- `--mt-device`: 翻訳モデルのデバイス指定（省略時は `--device` を継承）
- `--host` / `--port`: Gradio サーバのバインド先
- `--share`: 外部公開用のリンクを作成
- `--max-new-tokens`: ASR の 1 セグメントあたり最大トークン数
- `--mt-max-new-tokens`: 翻訳の最大生成トークン数
- `--max-segment-s`: 無音待ちに関係なく強制的に区切る最大秒数
- `--repo`: `--asr-repo` の旧名（互換のため残っています）

## 動作のポイント

- 音声入力は 24kHz で処理されます。
- 文字起こしは随時更新され、セグメント確定時に改行で追記されます。
- 翻訳はセグメント確定時にのみ更新され、翻訳欄に 1 行ずつ追記されます。
