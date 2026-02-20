# デモ8：求人提案オペ「下書き自動生成」（求職者×求人）

求人票テキスト・求職者プロフィール・過去提案例（任意）から、以下を自動生成するデモです。

- 提案文案（短文/長文）
- マッチ根拠ポイント（引用つき）
- 送付前チェックリスト（Must/Should）と確認質問

## 起動方法（ローカル）

```bash
cd /Users/tm/_Workspace/draft/demo6
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port 8502
```

## LLM設定（任意）

このデモは **Ollama / Mock** を切り替えできます。未設定でも `mock` で動きます。

### Ollama（ローカル）

```bash
export OLLAMA_BASE_URL="http://localhost:11434"  # 任意
export OLLAMA_MODEL="llama3.1"                   # 任意
```

## Docker（Ollama同梱で起動）

Ollama とアプリをまとめて立ち上げたい場合は以下。

```bash
cd /Users/tm/_Workspace/draft/demo6
docker compose up --build
```

初回はモデルを取得してください（例: `llama3.1`）。

```bash
docker compose exec ollama ollama pull llama3.1
```

起動後はブラウザで `http://localhost:8502` を開きます。

## デモの見せ方（おすすめ）

1. 「サンプル投入（1）」→「下書きを生成」
2. まず「根拠ポイント（引用つき）」を見せて安心感を作る
3. 「短文→長文→チェックリスト」の順で、送付前の事故防止まで一気に出る体験を見せる

