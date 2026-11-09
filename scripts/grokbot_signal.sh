#!/usr/bin/env zsh
# One-command Grok Bot + Moomoo MCP Quantitative Signal & Paper-Trading Plan Runner
set -euo pipefail

REPO_DIR="/Users/davidshizw/Documents/GitHub/stock-colab"
cd "$REPO_DIR"

export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
uv run quant-platform grokbot "$@"
