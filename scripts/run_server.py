"""便捷启动脚本：``python scripts/run_server.py``。

等价于::

    .\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

额外做了两件事：
    1. 把仓库根目录放进 ``sys.path``，因此从任意工作目录运行都能找到 ``app`` / ``learnpath``；
    2. 启动前打印前端地址与 API 文档地址，方便演示时直接打开。

用法::

    python scripts/run_server.py                # 默认 127.0.0.1:8000
    python scripts/run_server.py --port 8080    # 换端口
    python scripts/run_server.py --reload       # 开发模式（代码改动自动重启）
    python scripts/run_server.py --host 0.0.0.0 # 局域网内其他设备访问
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# scripts/run_server.py -> scripts/ -> 仓库根目录
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import uvicorn  # noqa: E402  (必须在 sys.path 调整之后)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="启动 LearnPath Web 服务（FastAPI + 单页应用）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--reload", action="store_true", help="开发模式：代码改动自动重启")
    parser.add_argument("--log-level", default="info", help="日志级别")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """启动 uvicorn。"""
    args = parse_args(argv)
    print("=" * 62)
    print("  LearnPath · 在线学习个性化课程推荐与学习路径规划")
    print(f"  前端页面 : http://{args.host}:{args.port}/")
    print(f"  API 文档 : http://{args.host}:{args.port}/docs")
    print(f"  健康检查 : http://{args.host}:{args.port}/api/health")
    print("=" * 62)
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
