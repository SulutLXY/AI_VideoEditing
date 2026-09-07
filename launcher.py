#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-AutoCut 一键启动器

功能：
- 检测 Python 环境
- 检查并安装项目依赖
- 启动 Web UI 服务
- 自动打开浏览器

Windows 使用方式：
    双击 start.bat

Linux/macOS 使用方式：
    ./start.sh

也可以直接运行：
    python launcher.py
    python launcher.py --port 7860 --config config/config.yaml
"""
import argparse
import os
import platform
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
WEBUI = PROJECT_ROOT / "webui.py"


class Launcher:
    def __init__(self, port: int = 7860, config: str = None, share: bool = False):
        self.port = port
        self.config = config
        self.share = share
        self.system = platform.system()
        self.process = None

    def log(self, msg: str):
        print(f"[启动器] {msg}")

    def check_python(self) -> bool:
        """检查 Python 版本是否 >= 3.9"""
        version = sys.version_info
        self.log(f"Python 版本: {version.major}.{version.minor}.{version.micro}")
        if version.major < 3 or (version.major == 3 and version.minor < 9):
            self.log("错误：需要 Python 3.9 或更高版本")
            return False
        return True

    def check_dependencies(self) -> bool:
        """检查核心依赖是否已安装"""
        required = ["gradio", "openai", "yaml", "lxml", "PIL", "cv2", "docx", "pypdf"]
        missing = []
        for module in required:
            try:
                __import__(module)
            except ImportError:
                missing.append(module)
        if missing:
            self.log(f"缺少依赖: {', '.join(missing)}")
            return False
        self.log("依赖检查通过")
        return True

    def install_dependencies(self):
        """安装 requirements.txt 中的依赖"""
        if not REQUIREMENTS.exists():
            self.log(f"错误：找不到 {REQUIREMENTS}")
            sys.exit(1)

        self.log("正在安装依赖，请稍候...")
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)]
        try:
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
            self.log("依赖安装完成")
        except subprocess.CalledProcessError as e:
            self.log(f"依赖安装失败: {e}")
            sys.exit(1)

    def kill_process_on_port(self, port: int) -> bool:
        """释放指定端口：结束占用该端口的进程，实现双击重启。

        Windows 下通过 netstat 找到 LISTENING 状态的 PID 再 taskkill；
        其他平台暂不支持，返回 False 表示未做任何处理。
        """
        if self.system != "Windows":
            return False
        try:
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True, text=True, check=False
            )
            pids = set()
            for line in result.stdout.splitlines():
                if f":{port}" not in line:
                    continue
                if "LISTENING" not in line.upper():
                    continue
                parts = line.split()
                if parts and parts[-1].isdigit():
                    pids.add(parts[-1])
            killed = False
            for pid in pids:
                r = subprocess.run(
                    ["taskkill", "/F", "/PID", pid],
                    capture_output=True, check=False
                )
                if r.returncode == 0:
                    self.log(f"已结束占用端口 {port} 的旧进程 (PID {pid})")
                    killed = True
                else:
                    self.log(f"结束 PID {pid} 失败: {r.stderr.decode(errors='ignore').strip()}")
            return killed
        except Exception as e:
            self.log(f"释放端口 {port} 时出错: {e}")
            return False

    def start_webui(self):
        """启动 webui.py 服务"""
        cmd = [sys.executable, str(WEBUI), "--port", str(self.port)]
        if self.config:
            cmd += ["--config", str(self.config)]
        if self.share:
            cmd += ["--share"]

        self.log(f"启动 Web UI: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

    def wait_for_server(self, timeout: int = 60) -> bool:
        """等待服务启动"""
        import urllib.request
        url = f"http://127.0.0.1:{self.port}/"
        start = time.time()
        while time.time() - start < timeout:
            try:
                with urllib.request.urlopen(url, timeout=2):
                    return True
            except Exception:
                time.sleep(0.5)
        return False

    def open_browser(self):
        """自动打开浏览器"""
        url = f"http://127.0.0.1:{self.port}/"
        self.log(f"正在打开浏览器: {url}")
        webbrowser.open(url)

    def stream_output(self):
        """在控制台实时输出 Web UI 日志"""
        if self.process and self.process.stdout:
            for line in self.process.stdout:
                print(line, end="")

    def run(self):
        """主流程"""
        self.log("=== LLM-AutoCut 启动器 ===")

        if not self.check_python():
            input("按回车键退出...")
            sys.exit(1)

        if not self.check_dependencies():
            self.install_dependencies()

        # 先释放端口，再启动，实现双击即重启
        self.kill_process_on_port(self.port)
        time.sleep(1)

        self.start_webui()

        # 启动日志输出线程
        log_thread = threading.Thread(target=self.stream_output, daemon=True)
        log_thread.start()

        # 等待服务启动
        self.log("等待服务启动...")
        if self.wait_for_server():
            self.open_browser()
        else:
            self.log("服务启动超时，请检查日志")

        try:
            self.process.wait()
        except KeyboardInterrupt:
            self.log("收到退出信号，正在关闭服务...")
            self.process.terminate()
            self.process.wait(timeout=5)
        finally:
            self.log("服务已停止")


def main():
    parser = argparse.ArgumentParser(description="LLM-AutoCut 一键启动器")
    parser.add_argument("--port", type=int, default=7860, help="监听端口（默认 7860）")
    parser.add_argument("--config", default=None, help="预加载配置文件路径")
    parser.add_argument("--share", action="store_true", help="生成公开分享链接")
    parser.add_argument("--install-only", action="store_true", help="仅安装依赖，不启动")
    args = parser.parse_args()

    launcher = Launcher(port=args.port, config=args.config, share=args.share)

    if args.install_only:
        launcher.install_dependencies()
        return

    launcher.run()


if __name__ == "__main__":
    main()
