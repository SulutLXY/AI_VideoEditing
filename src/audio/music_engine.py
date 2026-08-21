"""
Music Engine: 配乐引擎（从 ai-voice-dubber 合并）
支持：本地音乐库情绪匹配 / MusicGen AI生成
"""
import os
import random
from pathlib import Path
from typing import Optional, List, Dict

import librosa

from src.utils import logger


class MusicEngine:
    """配乐引擎"""

    def __init__(self, config: dict):
        self.config = config
        self.music_config = config.get("music", {})
        self.emotion_keywords = config.get("emotion_keywords", {})
        paths = config.get("paths", {})
        self.library_path = Path(paths.get("music_library", "music/library"))
        self.generated_path = Path(paths.get("music_generated", "music/generated"))
        self.generated_path.mkdir(parents=True, exist_ok=True)
        self.library_path.mkdir(parents=True, exist_ok=True)

        # 加载本地音乐库索引
        self.library_index: Dict[str, List[Path]] = {}
        self._build_library_index()

    # ------------------------------------------------------------------
    # 本地音乐库管理
    # ------------------------------------------------------------------
    def _build_library_index(self):
        """扫描本地音乐库，按情绪分类建立索引"""
        if not self.library_path.exists():
            return

        for emotion_dir in self.library_path.iterdir():
            if not emotion_dir.is_dir():
                continue
            emotion = emotion_dir.name
            files = []
            for ext in ("*.mp3", "*.wav", "*.flac", "*.ogg"):
                files.extend(emotion_dir.glob(ext))
            if files:
                self.library_index[emotion] = files

        total = sum(len(v) for v in self.library_index.values())
        logger.info(f"[MusicEngine] 本地音乐库: {total} 首 ({len(self.library_index)} 种情绪)")
        for emotion, files in sorted(self.library_index.items()):
            logger.info(f"  - {emotion}: {len(files)} 首")

    def add_to_library(self, file_path: str, emotion: str):
        """向音乐库添加新音乐"""
        src = Path(file_path)
        if not src.exists():
            raise FileNotFoundError(f"文件不存在: {src}")

        target_dir = self.library_path / emotion
        target_dir.mkdir(parents=True, exist_ok=True)

        import shutil
        dst = target_dir / src.name
        shutil.copy2(src, dst)

        if emotion not in self.library_index:
            self.library_index[emotion] = []
        self.library_index[emotion].append(dst)

        logger.info(f"[MusicEngine] 已添加: {src.name} -> {emotion}/")

    # ------------------------------------------------------------------
    # 情绪分析
    # ------------------------------------------------------------------
    def analyze_text_emotion(self, text: str) -> str:
        """根据文本内容分析情绪"""
        text_lower = text.lower()
        emotion_scores = {}

        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                emotion_scores[emotion] = score

        if emotion_scores:
            return max(emotion_scores, key=emotion_scores.get)

        return "平静"

    def analyze_segments_emotion(self, segments: List[dict]) -> List[tuple]:
        """分析多段文本的情绪分布"""
        results = []
        for seg in segments:
            text = seg.get("text", "")
            emotion = self.analyze_text_emotion(text)
            results.append((text, emotion))
        return results

    # ------------------------------------------------------------------
    # 本地配乐匹配
    # ------------------------------------------------------------------
    def match_from_library(self, emotion: str, duration: float,
                           exclude: Optional[List[str]] = None) -> Optional[Path]:
        """从本地库匹配配乐"""
        candidates = self.library_index.get(emotion, [])

        if not candidates:
            emotion_map = {
                "兴奋": "欢快", "开心": "欢快", "高兴": "欢快",
                "难过": "悲伤", "痛苦": "悲伤", "哭": "悲伤",
                "害怕": "紧张", "恐惧": "紧张", "逃跑": "紧张",
                "安静": "平静", "日常": "平静", "普通": "平静",
                "爱": "浪漫", "喜欢": "浪漫", "心动": "浪漫",
                "战斗": "史诗", "荣耀": "史诗", "胜利": "史诗",
                "谜": "悬疑", "诡异": "悬疑", "调查": "悬疑",
                "家": "温馨", "妈妈": "温馨", "谢谢": "温馨",
            }
            mapped = emotion_map.get(emotion)
            if mapped:
                candidates = self.library_index.get(mapped, [])

        if not candidates:
            logger.warning(f"[MusicEngine] 未找到情绪 '{emotion}' 的配乐")
            return None

        if exclude:
            candidates = [c for c in candidates if str(c) not in exclude]

        if not candidates:
            return None

        chosen = random.choice(candidates)

        try:
            actual_duration = librosa.get_duration(path=str(chosen))
            logger.info(f"[MusicEngine] 匹配配乐: {chosen.name} (情绪: {emotion}, 时长: {actual_duration:.1f}s)")
        except Exception:
            logger.info(f"[MusicEngine] 匹配配乐: {chosen.name} (情绪: {emotion})")

        return chosen

    def match_for_segments(self, segments: List[dict]) -> List[tuple]:
        """为多个段落匹配连续配乐"""
        timeline = []
        current_time = 0.0
        used_files = []

        for seg in segments:
            text = seg.get("text", "")
            duration = seg.get("duration", 5.0)
            emotion = self.analyze_text_emotion(text)

            music = self.match_from_library(emotion, duration, exclude=used_files)
            if music:
                used_files.append(str(music))
                timeline.append((music, current_time, current_time + duration))

            current_time += duration

        return timeline

    # ------------------------------------------------------------------
    # AI 音乐生成 (MusicGen)
    # ------------------------------------------------------------------
    def generate_music(self, prompt: str, duration: int = 30,
                       output_name: Optional[str] = None) -> Path:
        """
        使用 MusicGen 生成 BGM
        优先顺序: API 服务 -> audiocraft 本地 -> transformers 本地
        """
        musicgen_cfg = self.music_config.get("musicgen", {})
        api_url = musicgen_cfg.get("api_url")

        if api_url:
            return self._generate_music_api(prompt, duration, output_name, api_url)

        # 方案1: audiocraft 本地推理
        try:
            from audiocraft.models import MusicGen
            from audiocraft.data.audio import audio_write
            return self._generate_music_audiocraft(prompt, duration, output_name, musicgen_cfg)
        except ImportError:
            logger.warning("[MusicEngine] audiocraft 未安装，尝试 transformers 版本 MusicGen")

        # 方案2: transformers 本地推理（无需 audiocraft/PyAV）
        try:
            return self._generate_music_transformers(prompt, duration, output_name, musicgen_cfg)
        except Exception as e:
            logger.error(f"[MusicEngine] transformers 版本 MusicGen 也失败: {e}")
            raise RuntimeError(
                "MusicGen 无法运行。可选方案:\n"
                "1. 本地推理: 安装 audiocraft（需约 4GB 显存，首次下载模型约 2-3GB）\n"
                "2. 使用 transformers 版本: 当前已尝试，但失败（见上方错误）\n"
                "3. API 模式: 运行 scripts/start_musicgen_api.ps1 启动服务，"
                "然后在 config.yaml 中配置 music.musicgen.api_url"
            )

    def _generate_music_audiocraft(self, prompt: str, duration: int,
                                    output_name: Optional[str],
                                    musicgen_cfg: dict) -> Path:
        """使用 audiocraft 生成 BGM"""
        from audiocraft.models import MusicGen
        from audiocraft.data.audio import audio_write

        model_name = musicgen_cfg.get("model", "facebook/musicgen-small")
        logger.info(f"[MusicEngine] MusicGen(audiocraft) 生成中: '{prompt}' ({duration}s)")

        model = MusicGen.get_pretrained(model_name)
        model.set_generation_params(duration=duration)

        wav = model.generate([prompt])

        output_name = output_name or f"generated_{hash(prompt) % 100000:05d}"
        output_path = self.generated_path / f"{output_name}.wav"

        audio_write(
            str(output_path.with_suffix("")),
            wav[0].cpu(),
            model.sample_rate,
            strategy="loudness",
            loudness_compressor=True,
        )

        logger.info(f"[MusicEngine] audiocraft 生成完成: {output_path}")
        return output_path

    def _generate_music_transformers(self, prompt: str, duration: int,
                                     output_name: Optional[str],
                                     musicgen_cfg: dict) -> Path:
        """使用 transformers 生成 BGM（不依赖 audiocraft/PyAV）"""
        import os
        import torch
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        from scipy.io import wavfile

        # 国内网络环境下优先使用 hf-mirror 镜像
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_HUB_OFFLINE", "0")

        model_name = musicgen_cfg.get("model", "facebook/musicgen-small")
        logger.info(f"[MusicEngine] MusicGen(transformers) 生成中: '{prompt}' ({duration}s)")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MusicgenForConditionalGeneration.from_pretrained(model_name).to(device)
        processor = AutoProcessor.from_pretrained(model_name)

        # MusicGen small 默认采样率 32000 Hz，每 50 tokens 约 1s
        max_new_tokens = int(duration * model.config.audio_encoder.frame_rate)

        inputs = processor(
            text=[prompt],
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)

        audio = audio_values[0, 0].cpu().numpy()
        sample_rate = model.config.sample_rate

        output_name = output_name or f"generated_{hash(prompt) % 100000:05d}"
        output_path = self.generated_path / f"{output_name}.wav"

        wavfile.write(str(output_path), rate=sample_rate, data=audio)

        logger.info(f"[MusicEngine] transformers 生成完成: {output_path}")
        return output_path

    def _generate_music_api(self, prompt: str, duration: int,
                            output_name: Optional[str], api_url: str) -> Path:
        """通过 MusicGen API 服务生成 BGM"""
        import requests

        output_name = output_name or f"generated_{hash(prompt) % 100000:05d}"
        output_path = self.generated_path / f"{output_name}.wav"

        logger.info(f"[MusicEngine] 调用 MusicGen API: {api_url}/generate")
        try:
            response = requests.post(
                f"{api_url}/generate",
                json={"prompt": prompt, "duration": min(duration, 300)},
                timeout=600,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"无法连接 MusicGen API: {api_url}\n"
                "请先运行 scripts/start_musicgen_api.ps1 启动服务。"
            )

        with open(output_path, "wb") as f:
            f.write(response.content)

        logger.info(f"[MusicEngine] API 生成完成: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    def get_audio_duration(self, file_path: str) -> float:
        """获取音频时长"""
        try:
            return librosa.get_duration(path=file_path)
        except Exception:
            return 0.0

    def list_emotions(self) -> List[str]:
        """列出可用的情绪标签"""
        return list(self.library_index.keys())
