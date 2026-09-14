# -*- coding: utf-8 -*-
"""参考图预分析器

上传/变更参考图后（或 Phase 1 启动时兜底），用本地 VLM 对每张参考图做一次
短输出推理，提取结构化关键词：
  entity_type（人物/动物/怪物/道具/场景/其他）
  gender（男性/女性/雄性/雌性/无）
  appearance（外貌特征关键词列表）
  entity_name（名字，看不出填未知）
合并用户输入的名字与说明，写 ref_profiles.json（幂等：图片未变化则复用）。

产物供身份确认调用注入——用关键词档案而非光一个名字做对照。
"""
import os
import json
import time
from typing import Dict, Any, Optional

from src.utils import logger

PROFILE_VERSION = 3  # v3：删身份字段、性别填无/雄性/雌性、空模板防照抄；旧档案一律重做
PROFILE_FILE = "ref_profiles.json"

PROFILE_PROMPT = (
    "你是一位影视角色设定分析师。请仔细观察这张参考图，提取图中主要对象的关键设定，"
    "严格以 JSON 输出（不要输出任何其他文字，JSON 的值必须根据图片实际内容填写，禁止照抄字段说明）：\n\n"
    "- entity_type：对象类型，填 人物/动物/怪物/道具/场景/其他 之一\n"
    "- gender：男性/女性；动物填 雄性/雌性；看不出或本无性别填 无\n"
    "- appearance：外貌特征关键词数组，3-6 个（发型/服装/颜色/配饰/体态/毛色等）\n"
    "- entity_name：图中对象的名字；看不出填 未知\n\n"
    "输出格式（值为空，请按图填写）：\n"
    "{\n"
    '  "entity_type": "",\n'
    '  "gender": "",\n'
    '  "appearance": [],\n'
    '  "entity_name": ""\n'
    "}"
)


def _image_fingerprint(path: str) -> str:
    """图片轻量指纹（大小+mtime），用于幂等判断"""
    try:
        st = os.stat(path)
        return f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return ""


def profile_to_desc(profile: Dict[str, Any]) -> str:
    """把档案转成注入 prompt 的一行关键词描述"""
    bits = []
    if profile.get("entity_type"):
        bits.append(f"类型:{profile['entity_type']}")
    gender = profile.get("gender")
    if gender and gender != "无":
        bits.append(f"性别:{gender}")
    app = profile.get("appearance") or []
    if app:
        bits.append("特征:" + "/".join(str(a) for a in app[:6]))
    if profile.get("entity_name") and profile["entity_name"] != "未知":
        bits.append(f"名字:{profile['entity_name']}")
    return "，".join(bits)


def ensure_ref_profiles(refs_dir: str, engine, force: bool = False) -> Dict[str, Dict[str, Any]]:
    """确保 refs 目录下每张图都有分析档案，返回 {name: profile}。

    - 清单 refs.json（WebUI 维护：file/name/description）
    - 产物 ref_profiles.json，含用户备注；图片变化或 force 时重新分析
    """
    manifest_path = os.path.join(refs_dir, "refs.json")
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    profiles_path = os.path.join(refs_dir, PROFILE_FILE)
    existing: Dict[str, Any] = {}
    if os.path.exists(profiles_path) and not force:
        try:
            with open(profiles_path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") == PROFILE_VERSION:
                existing = data.get("references", {}) or {}
        except Exception as e:
            logger.warning(f"[RefProfiler] 读取旧档案失败，全部重做: {e}")

    references: Dict[str, Dict[str, Any]] = {}
    changed = False
    for item in manifest.get("images", []):
        name = item.get("name") or os.path.splitext(item.get("file", ""))[0]
        img_path = os.path.join(refs_dir, item.get("file", ""))
        user_note = item.get("description", "")
        fp = _image_fingerprint(img_path)

        old = existing.get(name)
        if old and old.get("_fingerprint") == fp and os.path.exists(img_path):
            old["user_note"] = user_note  # 备注允许随时更新，不触发重分析
            references[name] = old
            continue

        if not os.path.exists(img_path):
            logger.warning(f"[RefProfiler] 图片缺失，跳过: {img_path}")
            continue

        try:
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            result = engine.analyze_reference_image(img)
        except Exception as e:
            logger.warning(f"[RefProfiler] 分析失败 {name}: {e}")
            continue

        profile = {
            "entity_type": result.get("entity_type", ""),
            "gender": result.get("gender", ""),
            "appearance": result.get("appearance", []) or [],
            "entity_name": result.get("entity_name", ""),
            "user_note": user_note,
            "_fingerprint": fp,
        }
        references[name] = profile
        changed = True
        logger.info(
            f"[RefProfiler] {name}: {profile_to_desc(profile)}"
            + (f"（备注: {user_note}）" if user_note else "")
        )

    if changed or not os.path.exists(profiles_path):
        payload = {
            "version": PROFILE_VERSION,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "references": references,
        }
        with open(profiles_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(f"[RefProfiler] 已写出 {len(references)} 份参考图档案 -> {profiles_path}")

    return references
