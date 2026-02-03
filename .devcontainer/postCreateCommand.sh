#!/bin/bash

# 1. gcloud CLIのインストール (Featureが失敗するための代替策)
if ! command -v gcloud &> /dev/null; then
    echo "Installing gcloud CLI..."
    curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts > /dev/null
fi

# 2. パスをシェルの設定ファイルに追加（次回ログイン時用）
echo 'export PATH=$PATH:$HOME/google-cloud-sdk/bin' >> ~/.bashrc
# 現在のプロセス用にもパスを通す
export PATH=$PATH:$HOME/google-cloud-sdk/bin

# 3. JSONキーの作成
if [ -n "$GCP_SA_KEY" ]; then
    echo "$GCP_SA_KEY" > /tmp/gcp-key.json
    chmod 600 /tmp/gcp-key.json
    echo "GCP JSON key created."
else
    echo "Error: GCP_SA_KEY is empty."
    exit 1
fi

# 4. 認証実行
gcloud auth activate-service-account --key-file=/tmp/gcp-key.json