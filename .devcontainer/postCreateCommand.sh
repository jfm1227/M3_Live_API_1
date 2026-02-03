#!/bin/bash

# 1. Secrets (GCP_SA_KEY) の中身をファイルに書き出す
if [ -n "$GCP_SA_KEY" ]; then
    echo "$GCP_SA_KEY" > /tmp/gcp-key.json
    echo "Success: GCP JSON key created at /tmp/gcp-key.json"
else
    echo "Error: GCP_SA_KEY is empty. Please check Codespaces Secrets."
    exit 1
fi

# 2. gcloud CLI の認証を実行
gcloud auth activate-service-account --key-file=/tmp/gcp-key.json

# 3. ファイルの権限を絞る（セキュリティ上の推奨設定）
chmod 600 /tmp/gcp-key.json