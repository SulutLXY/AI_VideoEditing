"""
ASR Engine: 语音转文字
基于 SenseVoice Small (自动下载)
"""
import os
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Optional


class ASREngine:
    def __init__(
        self,
        model_id: str = "iic/SenseVoiceSmall",
        device: str = "cuda",
        batch_size: int = 1,
        cache_dir: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "asr"
        )
        self.model = None
        self._loaded = False

    def load(self):
        """加载模型（显存占用 ~1GB）"""
        if self._loaded:
            return
        try:
            from funasr import AutoModel
        except ImportError:
            raise ImportError("funasr not installed. Run: pip install funasr modelscope")

        # 首次会自动从 modelscope 下载
        os.makedirs(self.cache_dir, exist_ok=True)

        self.model = AutoModel(
            model=self.model_id,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=self.device,
            hub="ms",  # ModelScope 国内源
            cache_dir=self.cache_dir,
        )
        self._loaded = True
        print(f"[ASREngine] SenseVoice loaded")

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        self._loaded = False
        import torch
        torch.cuda.empty_cache()
        print("[ASREngine] unloaded")

    def extract_audio(self, video_path: str) -> str:
        """从视频提取音频为临时 WAV 文件"""
        import ffmpeg
        tmp_wav = tempfile.mktemp(suffix=".wav")
        try:
            (
                ffmpeg
                .input(video_path)
                .output(tmp_wav, ac=1, ar=16000, vn=None)
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            print(f"[ASREngine] ffmpeg error: {e}")
            return ""
        return tmp_wav

    def process_video(self, video_path: str) -> Dict:
        """
        处理单个视频，返回语音信息
        格式: {
            "has_speech": true,
            "text": "完整对话文本（去时间戳）",
            "transcriptions": [{"start": 0.5, "end": 3.2, "text": "..."}]
        }
        """
        self.load()

        wav_path = self.extract_audio(video_path)
        if not wav_path or not os.path.exists(wav_path):
            return {"has_speech": False, "text": "", "transcriptions": []}

        try:
            res = self.model.generate(
                input=wav_path,
                language="auto",      # 自动识别语言
                use_itn=True,         # 逆向文本归一化（数字转阿拉伯数字）
                batch_size=self.batch_size,
            )
        except Exception as e:
            print(f"[ASREngine] inference error: {e}")
            return {"has_speech": False, "text": "", "transcriptions": []}
        finally:
            # 清理临时文件
            try:
                os.remove(wav_path)
            except:
                pass

        if not res or len(res) == 0:
            return {"has_speech": False, "text": "", "transcriptions": []}

        # SenseVoice 返回格式解析
        transcriptions = []
        full_text_parts = []
        for item in res:
            # item 可能是 dict 或 list
            if isinstance(item, dict):
                text = item.get("text", "")
                stamp_sents = item.get("sentence_info", [])
            elif isinstance(item, list) and len(item) > 0:
                text = item[0].get("text", "") if isinstance(item[0], dict) else str(item[0])
                stamp_sents = item[0].get("sentence_info", []) if isinstance(item[0], dict) else []
            else:
                text = str(item)
                stamp_sents = []

            if not text:
                continue

            full_text_parts.append(text)

            # 尝试提取时间戳
            if stamp_sents:
                for sent in stamp_sents:
                    ts = sent.get("timestamp", [])
                    if len(ts) >= 2:
                        transcriptions.append({
                            "start": round(ts[0] / 1000.0, 2),
                            "end": round(ts[1] / 1000.0, 2),
                            "text": sent.get("text", "").strip()
                        })
            else:
                # 无时间戳，整段输出
                transcriptions.append({
                    "start": 0.0,
                    "end": 0.0,
                    "text": text.strip()
                })

        full_text = " ".join(full_text_parts).strip()
        return {
            "has_speech": bool(full_text),
            "text": full_text,
            "transcriptions": transcriptions
        }
