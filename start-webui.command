#!/bin/bash
# 双击启动 CogMirror Web UI（macOS .command；Linux 可 bash start-webui.command）
# 首次给真人试用前：朋友在欢迎页右上「切换用户」用自己的名字开新用户，
# 与你的 local_user 数据完全隔离。
cd "$(dirname "$0")" || exit 1

if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
if ! $PY -c "import cogmirror" >/dev/null 2>&1; then
  echo "未找到 cogmirror，请先在项目目录安装："
  echo "  pip install -e ."
  echo ""
  read -r -p "按回车键关闭窗口…" _dummy
  exit 1
fi

# 端口被占用（上次没关）时换一个
PORT="${COGMIRROR_PORT:-8300}"
exec $PY -m cogmirror.webui --port "$PORT" --open "$@"
