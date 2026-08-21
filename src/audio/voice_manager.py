"""
Voice Manager: 音色管理核心（从 ai-voice-dubber 合并）
支持：内置音色 / 下载音色 / 克隆音色 / 加载 .pth 模型
"""
import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from src.utils import logger


@dataclass
class VoiceProfile:
    """音色配置"""
    id: str              # 唯一标识
    name: str            # 显示名称
    engine: str          # 引擎类型: edge_tts / gpt_sovits / rvc
    source: str          # 来源: builtin / downloaded / cloned
    # Edge TTS 参数
    edge_voice: Optional[str] = None
    # GPT-SoVITS 参数
    gpt_model: Optional[str] = None
    sovits_model: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    # RVC 参数
    rvc_model: Optional[str] = None
    # 元数据
    description: str = ""
    language: str = "zh"
    sample_rate: int = 24000
    tags: Optional[List[str]] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class VoiceManager:
    """音色管理器"""

    # Edge TTS 内置中文音色列表
    EDGE_VOICES = {
        "晓晓": "zh-CN-XiaoxiaoNeural",
        "云希": "zh-CN-YunxiNeural",
        "云健": "zh-CN-YunjianNeural",
        "晓伊": "zh-CN-XiaoyiNeural",
        "晓辰": "zh-CN-XiaochenNeural",
        "晓涵": "zh-CN-XiaohanNeural",
        "晓墨": "zh-CN-XiaomengNeural",
        "晓茹": "zh-CN-XiaorouNeural",
        "晓霜": "zh-CN-XiaoshuangNeural",
        "晓萱": "zh-CN-XiaoxuanNeural",
        "晓颜": "zh-CN-XiaoyanNeural",
        "晓悠": "zh-CN-XiaoyouNeural",
        "晓甄": "zh-CN-XiaozhenNeural",
    }

    def __init__(self, config: Dict):
        self.config = config
        self.paths = config.get("paths", {})
        self.voices: Dict[str, VoiceProfile] = {}
        self._load_all_voices()

    # ------------------------------------------------------------------
    # 初始化加载
    # ------------------------------------------------------------------
    def _load_all_voices(self):
        """加载所有可用音色"""
        self._load_builtin_voices()
        self._load_downloaded_voices()
        self._load_cloned_voices()

    def _ensure_dir(self, key: str):
        path = Path(self.paths.get(key, key))
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load_builtin_voices(self):
        """Edge TTS 内置音色，无需文件"""
        for name, voice_id in self.EDGE_VOICES.items():
            profile = VoiceProfile(
                id=f"edge_{voice_id}",
                name=name,
                engine="edge_tts",
                source="builtin",
                edge_voice=voice_id,
                description=f"Edge TTS 内置音色: {name}",
            )
            self.voices[profile.id] = profile

    def _load_downloaded_voices(self):
        """加载已下载的第三方音色"""
        dl_dir = self._ensure_dir("voices_downloaded")
        if not dl_dir.exists():
            return

        for voice_dir in dl_dir.iterdir():
            if not voice_dir.is_dir():
                continue
            meta_file = voice_dir / "voice.json"
            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    profile = VoiceProfile(**data)
                    self.voices[profile.id] = profile
                except Exception as e:
                    logger.warning(f"加载下载音色失败 {voice_dir}: {e}")

    def _load_cloned_voices(self):
        """加载克隆音色"""
        clone_dir = self._ensure_dir("voices_cloned")
        if not clone_dir.exists():
            return

        for voice_dir in clone_dir.iterdir():
            if not voice_dir.is_dir():
                continue
            meta_file = voice_dir / "voice.json"
            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    profile = VoiceProfile(**data)
                    self.voices[profile.id] = profile
                except Exception as e:
                    logger.warning(f"加载克隆音色失败 {voice_dir}: {e}")

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def list_voices(self, engine: Optional[str] = None, source: Optional[str] = None) -> List[VoiceProfile]:
        """列出可用音色，支持过滤"""
        results = []
        for v in self.voices.values():
            if engine and v.engine != engine:
                continue
            if source and v.source != source:
                continue
            results.append(v)
        return results

    def get_voice(self, voice_id: str) -> Optional[VoiceProfile]:
        """获取指定音色"""
        return self.voices.get(voice_id)

    def get_voice_by_name(self, name: str) -> Optional[VoiceProfile]:
        """按名称查找音色（模糊匹配）"""
        for v in self.voices.values():
            if v.name == name:
                return v
        name_lower = name.lower()
        for v in self.voices.values():
            if name_lower in v.name.lower() or v.name.lower() in name_lower:
                return v
        return None

    # ------------------------------------------------------------------
    # 音色下载 / 导入
    # ------------------------------------------------------------------
    def import_local_model(self, model_path: str, name: str,
                           engine: str = "gpt_sovits",
                           description: str = "") -> VoiceProfile:
        """导入本地 .pth 模型文件"""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        voice_id = f"{engine}_{name}_{hash(name) % 10000:04d}"
        target_dir = self._ensure_dir("voices_downloaded") / voice_id
        target_dir.mkdir(parents=True, exist_ok=True)

        if model_path.is_file():
            shutil.copy2(model_path, target_dir / model_path.name)
            model_file = str(target_dir / model_path.name)
        else:
            for f in model_path.iterdir():
                if f.is_file():
                    shutil.copy2(f, target_dir / f.name)
            model_file = str(target_dir)

        profile = VoiceProfile(
            id=voice_id,
            name=name,
            engine=engine,
            source="downloaded",
            description=description,
        )

        if engine == "gpt_sovits":
            profile.gpt_model = model_file
        elif engine == "rvc":
            profile.rvc_model = model_file

        self._save_voice_meta(profile, target_dir)
        self.voices[voice_id] = profile

        logger.info(f"导入音色成功: {name} (ID: {voice_id})")
        return profile

    def download_from_hub(self, model_id: str, name: str,
                          source: str = "modelscope",
                          engine: str = "gpt_sovits") -> VoiceProfile:
        """从在线模型库下载音色"""
        logger.info(f"从 {source} 下载模型: {model_id}")

        try:
            if source == "modelscope":
                from modelscope import snapshot_download
                downloaded_path = snapshot_download(model_id)
            elif source == "huggingface":
                from huggingface_hub import snapshot_download
                downloaded_path = snapshot_download(repo_id=model_id)
            else:
                raise ValueError(f"不支持的平台: {source}")

            profile = self.import_local_model(downloaded_path, name, engine,
                                              f"从 {source} 下载: {model_id}")
            return profile

        except Exception as e:
            logger.error(f"下载失败: {e}")
            raise

    # ------------------------------------------------------------------
    # 声音克隆
    # ------------------------------------------------------------------
    def clone_voice(self, sample_path: str, name: str,
                    ref_text: Optional[str] = None,
                    engine: str = "gpt_sovits") -> VoiceProfile:
        """从样本音频克隆音色"""
        sample_path = Path(sample_path)
        if not sample_path.exists():
            raise FileNotFoundError(f"样本不存在: {sample_path}")

        voice_id = f"cloned_{name}_{hash(name) % 10000:04d}"
        target_dir = self._ensure_dir("voices_cloned") / voice_id
        target_dir.mkdir(parents=True, exist_ok=True)

        sample_target = target_dir / f"reference{sample_path.suffix}"
        shutil.copy2(sample_path, sample_target)

        profile = VoiceProfile(
            id=voice_id,
            name=f"{name}(克隆)",
            engine=engine,
            source="cloned",
            ref_audio=str(sample_target),
            ref_text=ref_text or "",
            description=f"从 {sample_path.name} 克隆的音色",
        )

        if engine == "gpt_sovits":
            logger.info("GPT-SoVITS 克隆音色已记录，如需训练模型请使用外部脚本")

        self._save_voice_meta(profile, target_dir)
        self.voices[voice_id] = profile

        logger.info(f"克隆音色创建成功: {profile.name} (ID: {voice_id})")
        return profile

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def _save_voice_meta(self, profile: VoiceProfile, directory: Path):
        """保存音色元数据到目录"""
        meta_file = directory / "voice.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(asdict(profile), f, ensure_ascii=False, indent=2)

    def delete_voice(self, voice_id: str) -> bool:
        """删除音色"""
        profile = self.voices.get(voice_id)
        if not profile:
            return False

        if profile.source == "downloaded":
            voice_dir = self._ensure_dir("voices_downloaded") / voice_id
        elif profile.source == "cloned":
            voice_dir = self._ensure_dir("voices_cloned") / voice_id
        else:
            voice_dir = None

        if voice_dir and voice_dir.exists():
            shutil.rmtree(voice_dir)

        del self.voices[voice_id]
        logger.info(f"删除音色: {profile.name}")
        return True
