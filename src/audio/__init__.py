"""
ai-voice-dubber 合并后的音频/配音模块

提供：
- VoiceManager: 音色库管理
- VoiceEngine: TTS 合成（edge-tts / gpt_sovits）
- MusicEngine: BGM 匹配/生成
- Dubber: 音频混合与视频合成
"""
from .voice_manager import VoiceProfile, VoiceManager
from .voice_engine import VoiceEngine
from .music_engine import MusicEngine
from .dubber import Dubber

__all__ = [
    "VoiceProfile",
    "VoiceManager",
    "VoiceEngine",
    "MusicEngine",
    "Dubber",
]
