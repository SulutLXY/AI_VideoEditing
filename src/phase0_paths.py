"""Phase 0 视频发现：兼容平铺及每镜头子目录。"""
from pathlib import Path

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}


def list_rough_clips(directory):
    root = Path(directory)
    if not root.is_dir():
        return []
    files = []
    for entry in root.iterdir():
        if entry.is_file() and entry.suffix.lower() in VIDEO_EXTENSIONS:
            files.append(str(entry))
        elif entry.is_dir():
            files.extend(str(p) for p in entry.iterdir()
                         if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    return sorted(files)
