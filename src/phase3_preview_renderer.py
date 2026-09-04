"""
Phase 3 预览视频渲染器

为 VLM 终审生成轻量 preview.mp4：
- 按 timeline decisions 截取原始素材片段
- 应用变速
- 拼接成无声 preview（不配 AI 配音/BGM）
"""
import os
import subprocess
from typing import List, Dict, Any, Optional

from src.utils import logger, ensure_dir, tc_to_sec, sec_to_tc


class PreviewRenderer:
    """生成 Phase 3 终审用的轻量 preview 视频"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = config.get("paths", {}).get("output", "./output")
        self.temp_dir = config.get("paths", {}).get("temp", "./temp")

    def render(
        self,
        decisions: List[Any],
        output_name: str = "preview.mp4",
        include_audio: bool = False,
    ) -> str:
        """渲染 preview.mp4，返回输出路径"""
        preview_dir = os.path.join(self.output_dir, "phase3_attempts")
        ensure_dir(preview_dir)
        output_path = os.path.join(preview_dir, output_name)

        if not decisions:
            logger.warning("[PreviewRenderer] decisions 为空，无法渲染 preview")
            return ""

        # 渲染每个 segment
        segments = []
        seg_dir = os.path.join(self.temp_dir, "preview_segments")
        ensure_dir(seg_dir)

        for i, d in enumerate(decisions):
            segment_path = self._render_segment(d, i, seg_dir, include_audio)
            if segment_path:
                segments.append(segment_path)

        if not segments:
            logger.error("[PreviewRenderer] 没有成功渲染任何 segment")
            return ""

        # 生成 concat list
        concat_list = os.path.join(seg_dir, "concat_list.txt")
        with open(concat_list, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(f"file '{seg.replace(os.sep, '/')}'\n")

        # 拼接
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            output_path,
        ]
        logger.info(f"[PreviewRenderer] 拼接 preview: {output_path}")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if result.returncode != 0:
            logger.error(f"[PreviewRenderer] 拼接失败: {result.stderr}")
            return ""

        logger.info(f"[PreviewRenderer] preview 已生成: {output_path}")
        return output_path

    def _render_segment(
        self,
        decision: Any,
        index: int,
        seg_dir: str,
        include_audio: bool,
    ) -> str:
        """渲染单个 segment"""
        d = self._decision_to_dict(decision)
        source_path = d.get("source_path") or d.get("clip_path") or d.get("video_path") or ""
        tc_in = d.get("tc_in") or d.get("source_in") or "00:00:00:00"
        tc_out = d.get("tc_out") or d.get("source_out") or ""
        speed_str = d.get("speed", "1x")
        speed_mult = self._parse_speed(speed_str)

        if not source_path or not os.path.exists(source_path):
            logger.error(f"[PreviewRenderer] segment {index} 源文件不存在: {source_path}")
            return ""

        # 默认 fps=30
        fps = 30.0
        try:
            src_in_sec = tc_to_sec(tc_in, fps)
            src_out_sec = tc_to_sec(tc_out, fps) if tc_out else None
        except Exception as e:
            logger.error(f"[PreviewRenderer] segment {index} 时间码解析失败: {e}")
            return ""

        if src_out_sec is None or src_out_sec <= src_in_sec:
            # 尝试读取视频总时长
            duration = self._get_video_duration(source_path)
            src_out_sec = src_in_sec + duration

        seg_duration = src_out_sec - src_in_sec
        output_path = os.path.join(seg_dir, f"seg_{index:04d}.mp4")

        # 变速滤镜
        vf = f"setpts=PTS/{speed_mult},fps=30"
        audio_flag = [] if include_audio else ["-an"]

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(src_in_sec),
            "-t", str(seg_duration),
            "-i", source_path,
            "-vf", vf,
        ] + audio_flag + [
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            output_path,
        ]

        logger.debug(f"[PreviewRenderer] 渲染 segment {index}: {source_path} [{src_in_sec:.2f}-{src_out_sec:.2f}] x{speed_mult}")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if result.returncode != 0:
            logger.error(f"[PreviewRenderer] segment {index} 渲染失败: {result.stderr}")
            return ""

        return output_path

    @staticmethod
    def _decision_to_dict(decision: Any) -> Dict[str, Any]:
        if hasattr(decision, "to_dict"):
            return decision.to_dict()
        if isinstance(decision, dict):
            return decision
        return decision.__dict__

    @staticmethod
    def _parse_speed(speed: str) -> float:
        s = str(speed).strip().lower()
        if s.endswith("%"):
            try:
                return float(s[:-1]) / 100.0
            except ValueError:
                return 1.0
        if s.endswith("x"):
            try:
                return float(s[:-1])
            except ValueError:
                return 1.0
        try:
            return float(s)
        except ValueError:
            return 1.0

    @staticmethod
    def _get_video_duration(video_path: str) -> float:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
            return float(result.stdout.strip()) if result.returncode == 0 else 0.0
        except Exception:
            return 0.0
