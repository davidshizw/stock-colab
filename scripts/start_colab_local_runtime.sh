#!/usr/bin/env zsh
# 启动本地 Jupyter Server 以供 Google Colab 网页端通过 "Connect to a local runtime" (连接到本地运行时) 接入
set -e

echo "=== Preparing Local Runtime for Google Colab Web UI ==="
uv run pip install -q jupyterlab notebook jupyter_http_over_ws

echo "=== Starting Jupyter Server on port 8888 (Colab origin allowed) ==="
echo "Copy the http://localhost:8888/?token=... URL printed below and paste it into:"
echo "Google Colab -> Top Right 'Connect' dropdown -> 'Connect to a local runtime'"
echo "--------------------------------------------------------------------------------"

uv run jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=8888 \
  --NotebookApp.port_retries=0 \
  --no-browser
