# LLM-AutoCut: 基于大语言模型的影视后期智能剪辑系统

> **定位**：AI 数字剪辑助理 —— 负责理解素材、语义级切分、分类归档、去重排序、提出剪辑方案；人类负责审美判断与最终精修。
>
> 设计理念：**先理解，再切分；状态决定处理深度。**

---

## 目录

1. [项目概述](#1-项目概述)
2. [核心设计思想](#2-核心设计思想)
3. [核心工作流](#3-核心工作流)
4. [技术栈](#4-技术栈)
5. [快速开始](#5-快速开始)
6. [剧本上传与预处理](#6-剧本上传与预处理)
7. [阶段详解](#7-阶段详解)
8. [素材状态机](#8-素材状态机)
9. [配置说明](#9-配置说明)
10. [输出规范](#10-输出规范)
11. [开发路线](#11-开发路线)

---

## 1. 项目概述

片场倒片式拍摄产生的素材具有**时间线杂乱、同场多 take、文件命名无规律**三大特征。传统人工粗剪需要剪辑师逐条浏览、记忆内容、手动分类，效率极低。

LLM-AutoCut 通过多模态大模型对素材进行**语义级理解**，结合剧本大纲自动完成：

- **语义级镜头拆分**：不是按像素变化切分，而是按机位、主体、情绪、剧情、动作、对话的综合连贯性判断。
- **故事线锚定**：将素材映射到剧本精确节点。
- **智能去重**：三层去重机制筛选最优 take。
- **剪辑语法决策**：自动推荐升格降格、转场、蒙太奇方案。
- **时间线输出**：生成标准 EDL/FCPXML 导入专业剪辑软件。

同时，系统支持三种素材状态：原始素材、人工已处理素材、已有分析结果素材，避免对已经做好的工作进行重复或错误处理。

---

## 2. 核心设计思想

### 2.1 先分析，再切分

传统做法是先用 FFmpeg 检测场景变化，然后逐个镜头做分析。这样容易误切连贯镜头，或漏掉同机位跳切。

LLM-AutoCut 采用**先理解视频语义，再决定是否切分**：

1. 用传统 CV 提取画质、分辨率、帧率、运动稳定性等固有属性。
2. 用多模态模型以抽帧序列理解镜头内容：机位、主体、情绪、动作、台词。
3. 按优先级计算切分积分，决定是否切分。
4. 对一镜到底 / 长镜头默认保护，避免过度拆分。

### 2.2 切分积分

对每个候选切分点按以下优先级加权：

```
cut_score =
    0.35 × 机位切换
  + 0.25 × 主体/角色变化
  + 0.15 × 情绪断裂
  + 0.10 × 剧情节点偏移
  + 0.10 × 动作不连续
  + 0.05 × 对话/音频断裂
```

- `> 0.55`：确定切分
- `0.40 ~ 0.55`：候选切分，建议复核
- `≤ 0.40`：不切分

长镜头/一镜到底的切分阈值自动上调，默认不拆。

### 2.3 素材状态决定处理深度

| 状态 | 含义 | 系统行为 |
|---|---|---|
| `RAW` | 原始片场素材 | 完整流程：分析 + 切分 |
| `PROCESSED` | 人工已处理好的视频 | 只分析内容，**不切分** |
| `ANALYZED` | 已有分析结果，格式不统一 | 只做格式转换/校验，**不分析、不切分** |

---

## 3. 核心工作流

```
┌─────────────────────────────────────────────────────────────────────┐
│  0. 素材状态识别                                                       │
│  根据配置/目录/文件名识别每段素材: RAW / PROCESSED / ANALYZED           │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 0: 纯 CV 粗剪（v0.2 新增）                                     │
│  Input:  RAW 原始素材                                                  │
│  Output: 粗剪片段 + 粗配置文件（不调用 AI，只拆硬切镜头）              │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 1: 素材结构化分析 + 语义级镜头拆分                               │
│  Input:  剧本大纲 + 原始素材 + 角色参考图                              │
│  Output: 结构化分析表（每镜头的多维元数据 + 前后关系 + 切分溯源）        │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 2: 重复检测 + 时间线去重与合并                                  │
│  Input:  Phase 1 输出                                                │
│  Output: 去重后的最优素材集 + 缺失项标记 + 备选池                       │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 3: 剪辑语法决策（精剪方案生成）                                  │
│  Input:  去重素材 + 故事线 + 目标时长/风格                            │
│  Output: 剪辑决策文档（升格降格 / 转场 / 蒙太奇 / 反应镜头等）           │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 4: 二次匹配与执行输出                                           │
│  Input:  剪辑方案 + 原始素材                                           │
│  Output: EDL / FCPXML / CSV / JSON → 导入 PR / FCPX / DaVinci 精修     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. 技术栈

| 层级 | 工具/模型 | 用途 |
|-----|----------|------|
| **视觉理解** | Qwen2.5-VL (本地) / GPT-4o / Gemini / 豆包 | 镜头内容分析、角色识别、场景描述、机位判断 |
| **语音处理** | SenseVoice (本地) / Whisper (OpenAI) | ASR 转录、台词提取、时间戳对齐 |
| **人脸识别** | InsightFace `buffalo_l` (本地) | 角色身份识别、参考图建库 |
| **推理规划** | DeepSeek-V3 / Claude 3.5 / GPT-4o | 故事线锚定、剪辑语法决策 |
| **视觉特征** | CLIP / ResNet-50 | 语义特征提取、场景相似度 |
| **去重检测** | pHash / aHash (`imagehash`) | 感知哈希重复检测 |
| **视频处理** | FFmpeg | 关键帧提取、场景检测、变速、转码、无损切分 |
| **时间线生成** | Python + `lxml` | EDL / FCPXML 生成 |
| **格式适配** | Python Adapter 模式 | 外部分析结果转换 |
| **工作流编排** | Python 脚本 | 四阶段 Pipeline 串联 |

---

## 5. 快速开始

### 5.1 环境准备

```bash
# 克隆项目
git clone https://github.com/yourname/llm-autocut.git
cd llm-autocut

# 方式 A：使用项目内置的 Python 3.11 + CUDA PyTorch 虚拟环境（推荐，已集成本地模型）
# 第一次会自动创建 .venv311 并安装依赖
uv python install 3.11
uv venv --python 3.11 .venv311
uv pip install --python .venv311/Scripts/python.exe torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv311/Scripts/python.exe -r requirements.txt

# 方式 B：自行创建虚拟环境（需自行处理 CUDA PyTorch 与本地模型依赖）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 安装 FFmpeg（系统依赖）
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
# Windows: https://ffmpeg.org/download.html
```

> **本地模型说明**：v0.3 起默认启用本地视觉（Qwen2.5-VL-3B-Instruct）、本地 ASR（SenseVoice）和本地人脸识别（InsightFace）。首次运行会从 ModelScope 自动下载模型（总计约 6-8GB）。如需回退到在线模型，将 `config.yaml` 中 `models.vlm.provider` / `models.asr.provider` 改为 `openai` / `whisper` 并填写对应 API Key。

### 5.2 配置文件

复制模板并编辑：

```bash
cp config/config.example.yaml config/config.yaml
```

```yaml
# config/config.yaml
project:
  name: "短片项目_雨夜告白"
  target_duration: "5m"        # 目标成片时长
  style: "紧张中带温情"         # 风格描述
  genre: "剧情短片"             # 类型

paths:
  raw_materials: "./materials/"   # 根目录，下面按状态分子目录
  reference_images: "./refs/"
  output: "./output/"
  script_outline: "./script.md"
  temp: "./temp/"

# 素材状态配置（核心新增）
materials:
  - path: "./materials/raw/"
    state: "RAW"
    split: true
    analyze: true

  - path: "./materials/processed/"
    state: "PROCESSED"
    split: false
    analyze: true

  - path: "./materials/analyzed/"
    state: "ANALYZED"
    split: false
    analyze: false
    meta_format: "autocut_v1"    # 可选 autocut_v1 / custom_v1 / csv

# 按文件名正则覆盖上述规则
materials_overrides:
  - match: ".*Opening.*\\.mp4"
    state: "PROCESSED"
    split: false
    analyze: true

# 切分积分权重
split_scoring:
  weights:
    camera_change: 0.35
    subject_change: 0.25
    emotion_break: 0.15
    plot_shift: 0.10
    action_break: 0.10
    dialogue_break: 0.05
  thresholds:
    high: 0.55
    medium: 0.40
  long_take_protection:
    enabled: true
    min_duration: 30.0
    threshold_boost: 0.15

models:
  vlm:
    provider: "openai"
    model: "gpt-4o"
    api_key: "${OPENAI_API_KEY}"
    frame_sample_rate: 1
    max_frames: 60
  llm:
    provider: "deepseek"
    model: "deepseek-chat"
    api_key: "${DEEPSEEK_API_KEY}"
  asr:
    provider: "whisper"
    model: "large-v3"

processing:
  scene_threshold: 0.3
  min_shot_duration: 1.0
  duplicate_similarity: 0.92
  slow_motion_min_fps: 60
  export_edl: true
  export_fcpxml: true
  export_csv: true
  export_json: true
```

### 5.3 准备输入文件

#### 素材目录结构

```text
materials/
├── raw/              # 原始片场素材，会被分析并切分
├── processed/        # 人工已处理好的镜头/段落，只分析不切分
└── analyzed/         # 已有分析结果，只做格式转换
    ├── scene_01.mp4
    ├── scene_01_config.json   # LLM-AutoCut 自身生成的配置（autocut_v1）
    └── scene_01_meta.json     # 旧版/外部配置（custom_v1 / csv）
```

#### 剧本大纲格式 (`script.md`)

```markdown
# 雨夜告白

## 第一幕：等待
### 场1-情节点A
- 地点：咖啡馆内
- 时间：傍晚
- 内容：男主独自等待，表现焦虑，反复看表
- 情绪：焦虑、压抑
- 关键动作：看表、深呼吸、望向门口
- 关键台词：""

### 场1-情节点B
- 地点：咖啡馆门口
- 时间：雨夜
- 内容：女主推门而入，两人对视
- 情绪：紧张 → 期待
- 关键动作：推门、对视、停顿
- 关键台词：""
```

#### 角色参考图 (`refs/`)

- 命名规则：`角色名_角度.jpg`，如 `男主_正面.jpg`、`女主_侧面.jpg`
- 建议每个角色提供 2-3 个不同角度的参考图

### 5.4 运行 Pipeline

```bash
# 一键运行完整工作流
python main.py --config config/config.yaml --all

# 或分阶段运行
python main.py --phase 0  # 仅执行 RAW 素材 CV 粗剪
python main.py --phase 1  # 仅执行素材结构化分析
python main.py --phase 2  # 仅执行去重
python main.py --phase 3  # 仅生成剪辑方案
python main.py --phase 4  # 仅输出时间线
```

### 5.5 Web UI 可视化运行

项目内置基于 Gradio 的可视化工作台，支持配置编辑、素材上传、剧本编写、分阶段运行和结果下载。

#### 方式一：双击启动（推荐）

```text
Windows:   双击 start.bat
Linux/Mac: ./start.sh
```

启动器会自动：
1. 检查 Python 环境
2. 检查并安装依赖
3. 启动 Web UI 服务
4. 自动打开浏览器访问 `http://localhost:7860`

#### 方式二：命令行启动

```bash
# 安装依赖（首次）
python -m pip install -r requirements.txt

# 启动 Web UI（默认 http://localhost:7860）
python webui.py

# 指定端口或预加载配置
python webui.py --config config/config.yaml --port 7860

# 生成公开分享链接（临时公网访问）
python webui.py --share
```

界面功能：
- **配置**：在线编辑 `config.yaml`，调整模型、切分权重、输出选项。
- **剧本**：
  - 直接上传 `.md` / `.txt` / `.docx` / `.pdf` 剧本文档。
  - 点击「自动解析为台本」，由 DeepSeek 把原始剧本转换为标准 `script.md` 大纲。
  - 支持在编辑器中人工微调后再运行 Pipeline。
- **素材**：
  - **RAW**：拖拽上传原始片场素材，自动放入 `materials/raw/`。
  - **PROCESSED**：上传人工已处理好的镜头/段落，自动放入 `materials/processed/`，只分析不切分。
  - **ANALYZED**：上传视频 + 对应的 JSON/CSV 分析文件，自动放入 `materials/analyzed/`，只做格式转译。
- **运行**：选择完整流程或单独运行 Phase 1~4，实时查看日志。
- **结果**：刷新输出文件列表，预览 JSON / CSV / EDL，下载生成的时间线文件。

---

## 6. 剧本上传与预处理

### 6.1 支持的剧本格式

| 格式 | 说明 |
|-----|------|
| `.md` / `.txt` | 直接读取纯文本 |
| `.docx` | 需要 `python-docx` |
| `.pdf` | 需要 `pypdf` |

### 6.2 自动解析流程

原始编剧稿/分场稿通常不是项目需要的「幕 → 场 → 情节点」结构。点击「自动解析为台本」后，系统会：

1. 读取上传的剧本文档。
2. 判断内容是否已是标准大纲；如果是，直接透传。
3. 如果不是，按常见场景标题（如「第一场」「内景/外景」）把剧本切成场景块。
4. 对每个场景块调用 DeepSeek，输出结构化 JSON：幕、场号、情节点、地点、时间、内容、情绪、关键动作、关键台词、建议镜头。
5. 合并为项目标准的 `script.md` Markdown 大纲，回填编辑器。

### 6.3 命令行预处理

```bash
# 仅把原始剧本转换为 script.md，不运行后续 Phase
python main.py --config config/config.yaml --preprocess-script /path/to/剧本.docx
```

### 6.4 配置项

```yaml
script_preprocessing:
  enabled: true                # 是否启用自动解析
  provider: "deepseek"         # 复用 models.llm
  max_chunk_chars: 4000        # 场景/段落最大字符数
  output_shot_requirements: true  # 是否输出建议镜头
```

### 6.5 注意事项

- 自动解析结果**必须人工复核**。LLM 对场次、角色、情绪的判断可能出错，尤其是人物关系复杂或场景跳跃的剧本。
- 建议先上传 → 自动解析 → 在编辑器里微调 → 再运行 Pipeline。

---

## 7. 阶段详解

### Phase 0: 纯 CV 粗剪（RAW 素材）

**目标**：用传统 CV 先把原始素材中的硬切镜头拆开，输出物理片段和粗配置文件，为后续 Phase 1 的多模态精细分析做准备。

**处理流程**：

1. **CV 预扫描**：分辨率、帧率、码率、时长、画幅比例、视觉质量。
2. **逐帧灰度直方图差异**：计算相邻帧颜色分布差异，得到 cut_score。
3. **孤立峰值检测**：在 ±0.1 秒窗口内为局部最大值，且上下文 prominence ≥ 0.25 的帧判为硬切候选。
4. **持续高活动保护**：连续高分帧（≥ 全局基线 + 0.05）持续 ≥ 0.6 秒的区域，视为运动/叠化/淡入淡出/高速镜头，记录但不切分。
5. **邻近切点合并**：≤ 0.3 秒的邻近硬切点合并为一个。
6. **长片段二次低阈值检测**：超过 3.5 秒的片段再用 prominence ≥ 0.10 检测一次，避免漏拆弱切镜。
7. **FFmpeg 无损切分**：按最终切分点时间码输出 `phase0_rough_clips/S###.mp4` 和 `phase0_rough_config.json`。

**状态差异**：

- 只有 `RAW` 素材进入 Phase 0。
- `PROCESSED` / `ANALYZED` 跳过 Phase 0，直接进入 Phase 1 的对应分支。

**可调参数**（`config.yaml` → `phase0`）：

```yaml
phase0:
  enabled: true
  prominence_min: 0.25
  long_segment_threshold: 3.5
  long_segment_prominence: 0.10
  hard_cut_window: 0.1
  hard_cut_merge_window: 0.3
  baseline_delta: 0.05
  soft_transition_gap_tol: 0.15
  soft_transition_min_duration: 0.6
```

---

### Phase 1: 素材结构化分析 + 语义级镜头拆分

**目标**：先理解每个视频的内容和连贯性，再决定是否拆分，将每个镜头映射到剧本精确节点。

**处理流程**：

1. **状态识别**：根据配置判断素材是 `RAW` / `PROCESSED` / `ANALYZED`。
2. **CV 预扫描**：提取画质、分辨率、帧率、码率、运动稳定性、候选切分点。
3. **视频原生理解（Video-Native VLM）**：
   - 默认以 `vlm_fallback` 模式运行：把视频抽帧为时序序列送入多模态模型，让模型像看电影一样理解内容。
   - 输出：段落级描述、机位、主体、情绪、动作、台词，以及**段落内部连贯性评分**和**建议切分点**。
   - 未来可替换为直接上传视频文件的 provider（如 Gemini 1.5 Pro / Qwen2.5-VL）。
4. **ASR 转录**：提取带时间戳的台词。
5. **切分积分计算**：对 VLM 建议切分点和 CV 候选点按六维优先级（机位 > 主体 > 情绪 > 剧情 > 动作 > 对白）计算切分积分。
6. **长镜头/高连贯保护**：
   - 一镜到底 / 连续长镜头 / 段落连贯性评分 ≥ 0.85 时默认不拆分。
   - 切分阈值自动上调，避免过度拆分。
7. **FFmpeg 拆分**：按最终切分点时间码无损拆分（`-c copy`），生成独立物理片段，记录 `split_clip_path`。
8. **LLM 故事线锚定**：将镜头描述映射到剧本情节点。
9. **生成关系图**：记录每个 Shot 与前后镜头的连贯性类型和分数。
10. **镜头配置信息**：为每个 Shot 填充完整配置（时长、分辨率、帧率、画幅比例、镜头类型、机位、运镜、主体、情绪、动作、对白、前后关联等）。

**状态差异**：

- `RAW`：先经 Phase 0 CV 粗剪，再对粗剪片段做完整分析（CV 预扫描 → 视频原生理解 → 切分积分 → FFmpeg 拆分 → 关系图 → 剧本锚定）。每个 Shot 输出 `{shot_id}.mp4` + `{shot_id}_config.json`。
- `PROCESSED`：只分析整体内容，**不切分**；把原视频拷贝为 `{shot_id}.mp4`，并生成统一 `{shot_id}_config.json`，标记 `do_not_split=true`。
- `ANALYZED`：通过 Adapter 读取已有分析文件，**不分析、不切分**；同样拷贝原视频并输出统一配置。若关键字段缺失则标记 `needs_review=true`。

**输出示例** (`output/phase1_analysis.json`)：

```json
{
  "shots": [
    {
      "shot_id": "S001",
      "state": "RAW",
      "source_file": "Camera_A_Take1.mp4",
      "tc_in": "00:01:23:10",
      "tc_out": "00:01:28:05",
      "vlm_description": {
        "location": "室内咖啡馆",
        "characters": ["男主"],
        "shot_size": "中景",
        "camera_position": "柜台正面固定机位",
        "action": "坐立不安，看表",
        "emotion": "焦虑",
        "dialogue": null,
        "visual_quality": 4.2,
        "stability": 4.0
      },
      "script_anchor": {
        "act": "第一幕",
        "scene": "场1",
        "beat": "情节点A",
        "function": "主镜头-焦虑等待",
        "confidence": 0.94
      },
      "relationships": {
        "prev": null,
        "next": {
          "shot_id": "S002",
          "relationship_type": "情绪延续",
          "coherence_score": 0.88
        }
      },
      "provenance": {
        "state": "RAW",
        "generated_by": "vlm_analysis",
        "split_decision": {
          "score": 0.72,
          "reason": "机位切换 + 情绪转折",
          "protected": false
        }
      }
    }
  ]
}
```

### Phase 2: 剧本-镜头匹配 + 去重 + take 选择

**目标**：先由 DeepSeek 文本模型把每个 Shot 的配置信息匹配到剧本情节点，再在同一情节点内进行去重，最终选出核心 take 与备选 take。

**处理流程**：

1. **剧本-镜头语义匹配**（新增核心步骤）
   - 输入：`script.md` 解析出的情节点 + Phase 1 产出的 Shot 配置。
   - 调用 `LLMService.anchor_shots_to_script()`，由 DeepSeek 比较情节点需求（地点、时间、内容、情绪、关键动作、台词）与每个 Shot 的配置字段（场景、时间、角色、景别、机位、运镜、动作、情绪、风格、标签、台词）。
   - 输出每个 Shot 的 `script_anchor`：
     - `beat`: 匹配的情节点 ID
     - `act`: 幕
     - `function`: 功能标签（主镜头 / 反应镜头 / 插入镜头 / 过渡 / 空镜 / B-roll）
     - `confidence`: 匹配置信度 0.0-1.0
     - `reasoning`: 匹配理由

2. **质量分计算**
   - 基于 `quality_scoring.weights` 计算每个 Shot 的综合质量分：
     - `visual_quality`：视觉质量
     - `stability`：稳定性
     - `script_confidence`：剧本匹配置信度（来自 DeepSeek）
     - `dialogue`：台词信息
     - `duration`：时长适中度
     - `metadata_complete`：元数据完整度

3. **状态标记**
   - `RAW`：候选，参与去重与核心选择。
   - `PROCESSED`：强制保留，不参与去重。
   - `ANALYZED`：待复核，不参与去重。
   - 无法匹配任何情节点的镜头标记为「未匹配」。

4. **三层去重机制**（仅作用于同组 RAW 候选）

| 层级 | 方法 | 处理对象 |
|-----|------|---------|
| L1 文件级 | MD5 哈希 | 完全相同的文件 |
| L2 视觉级 | pHash + CLIP 特征 | 几乎相同的画面（同 take 细微差异） |
| L3 语义级 | VLM 描述 + 台词匹配 | 同一场戏的不同 take |

5. **核心 take 选择**
   - 每个情节点按质量分排序，最高分 RAW 素材成为「核心」。
   - 其余同组素材标记为「备选」或「废弃」（受关系图保护的镜头不会被废弃）。
   - 没有被核心 Shot 覆盖的情节点会进入 `missing_beats` 报告。

**配置示例**：

```yaml
quality_scoring:
  weights:
    visual_quality: 0.25
    stability: 0.20
    script_confidence: 0.25
    dialogue: 0.10
    duration: 0.10
    metadata_complete: 0.10
```

**输出示例** (`output/phase2_deduplication.json`)：

```json
{
  "duplicate_groups": [
    {
      "group_id": "G001",
      "script_beat": "第一幕-场1-情节点A",
      "shots": [
        {"shot_id": "S001", "score": 8.5, "status": "保留", "reason": "表演最自然"},
        {"shot_id": "S015", "score": 7.2, "status": "备选", "reason": "曝光稍过"},
        {"shot_id": "S032", "score": 6.8, "status": "废弃", "reason": "画面抖动"}
      ]
    }
  ],
  "missing_beats": [
    {"beat": "第二幕-场3-情节点E", "severity": "高", "note": "无对应素材，需补拍"}
  ]
}
```

### Phase 3: 剪辑语法决策

**目标**：为每个镜头决策速度、剪辑手法、转场、音频。

**决策维度**：

| 维度 | 选项 | 决策依据 |
|-----|------|---------|
| 速度 | 1x / 升格(40-80%) / 降格(200-600%) | 情绪标签、叙事功能、素材帧率 |
| 剪辑手法 | 连续/J-Cut/L-Cut/交叉/跳切/匹配剪辑 | 上下文叙事需求、前后镜头关系 |
| 转场 | 硬切/叠化/闪白/黑场 | 时间/空间/情绪过渡 |
| 插入镜头 | 反应镜头/Cutaway | 对话场景、情感补充 |
| 音频 | 保留/配乐/音效/J-Cut/L-Cut | 叙事节奏 |

### Phase 4: 二次匹配与执行输出

**目标**：将逻辑时间线映射回物理素材，生成可导入剪辑软件的格式。

**支持输出格式**：

- **EDL** (CMX3600)：通用性最强，所有 NLE 支持
- **FCPXML**：Final Cut Pro / DaVinci Resolve
- **XML**：Premiere Pro
- **CSV**：供人工审核用
- **JSON**：完整数据与溯源信息

---

## 7. 素材状态机

### 7.1 RAW：原始素材

完整处理流程：

```
CV 预扫描 → 多模态分析 → 切分积分 → 长镜头保护 → FFmpeg 拆分 → 元数据 + 关系图
```

### 7.2 PROCESSED：人工已处理素材

- 只分析内容，**不切分**。
- 整个视频作为一个 Shot 输出。
- 标记 `do_not_split: true`。
- 可包含内部子段落标记，但不生成独立 Shot。

典型场景：

- 导演已经剪好的开场段落
- 调色/ stabilized 后的完成镜头
- 纪录片中已剪辑好的采访段落

### 7.3 ANALYZED：已有分析结果

- 只做格式转换与校验，**不分析、不切分**。
- 通过 Adapter 读取不同来源的分析结果。
- 输出统一的标准 Shot 结构。
- 标记 `needs_review: true`，提醒人工复核。

支持的来源格式（逐步扩展）：

- `custom_v1`：项目自定义旧格式
- `csv_meta`：CSV 表格
- `manual_json`：人工填写的 JSON
- `fcpxml_meta`：从 FCPXML 导入的元数据

---

## 8. 配置说明

### 8.1 模型选择建议

| 场景 | 推荐模型 | 备注 |
|-----|---------|------|
| 预算充足、追求精度 | GPT-4o (VLM+LLM) | 视觉理解最强，但成本高 |
| 国内性价比优先 | 豆包 Doubao-vision-lite / Doubao-vision-pro + DeepSeek-V3 | 火山方舟/OpenAI 兼容接口，速度快 |
| 性价比优先 | Qwen2.5-VL + DeepSeek-V3 | 开源可本地部署，VLM 显存 5.7GB |
| 完全本地离线 | MiniCPM-o + Qwen2.5-72B | 适合保密项目 |
| 快速原型验证 | Gemini 1.5 Pro | 上下文长，适合长视频分析 |

### 8.2 关键参数调优

```yaml
# 切分积分阈值
split_scoring:
  thresholds:
    high: 0.55       # 高于此值确定切分
    medium: 0.40     # 候选切分，建议复核
  long_take_protection:
    enabled: true
    min_duration: 30.0
    threshold_boost: 0.15

# 语义去重阈值
# 值越高，越不容易被判为重复
duplicate_similarity: 0.92

# 升格最低帧率要求
slow_motion_min_fps: 60

# VLM 采样控制，防止长视频成本爆炸
models:
  vlm:
    frame_sample_rate: 1   # 每秒 1 帧
    max_frames: 60         # 单视频最多分析 60 帧
```

### 8.3 使用豆包（Doubao）视觉模型

豆包视觉模型（如 `doubao-seed-2-0-pro-260215`、`doubao-vision-lite-32k`、`doubao-vision-pro-32k`）通过火山方舟提供 OpenAI 兼容接口，配置方式如下：

```yaml
models:
  vlm:
    provider: "doubao"                          # 或 "openai"
    model: "doubao-seed-2-0-pro-260215"     # 用户实测可用的豆包视觉模型
    api_key: "your-doubao-api-key"          # 火山方舟 API Key
    base_url: "https://ark.cn-beijing.volces.com/api/v3"
    max_tokens: 4096
    temperature: 0.3
    frame_sample_rate: 5
    max_frames: 150

  video_vlm:
    provider: "vlm_fallback"
    model: "doubao-seed-2-0-pro-260215"
    max_tokens: 4096
    temperature: 0.3
    frame_sample_rate: 5
    max_frames: 150

  llm:
    provider: "doubao"                          # 或 "openai"
    model: "doubao-pro-32k"                     # 文本模型
    api_key: "your-doubao-api-key"
    base_url: "https://ark.cn-beijing.volces.com/api/v3"
    max_tokens: 8192
    temperature: 0.5
```

说明：
- `provider` 填 `doubao` 即可，系统内部会优先使用火山方舟 **Responses API** 格式调用，失败时自动 fallback 到 `chat.completions`。
- 只需修改 `vlm` 和 `llm` 的 `model` / `api_key` / `base_url`，其余流程不变。
- 豆包视觉模型支持抽帧图像输入，与当前 `vlm_fallback` 视频理解流程完全兼容。
- **请勿将真实 API Key 上传到公开仓库**，建议使用本地配置文件或环境变量 `ARK_API_KEY`。`load_config` 会自动从环境变量读取。

---

## 9. 输出规范

### 9.1 目录结构

```text
output/
├── phase0_rough_config.json   # 粗剪结果总表（含每个片段的时间码、来源文件）
├── phase0_rough_clips/        # CV 粗剪后的物理片段
├── phase1_analysis.json       # 素材分析结果
├── phase1_keyframes/            # 提取的关键帧
├── phase2_deduplication.json    # 去重结果
├── phase2_deduplication.csv   # 去重审核表
├── phase3_edit_decision.json   # 剪辑方案
├── phase3_edit_decision.csv    # 剪辑方案审核表
├── timeline.edl                # EDL 时间线
├── timeline.fcpxml             # FCPXML 时间线
├── timeline_final.csv          # 最终审核表
├── timeline.json               # 完整时间线数据
└── logs/
    └── pipeline.log            # 完整执行日志
```

### 9.2 人工审核检查清单

在导入剪辑软件前，建议人工确认：

- [ ] **故事线完整性**：所有关键情节点是否都有素材覆盖？
- [ ] **长镜头保护**：一镜到底/长镜头是否被误切？
- [ ] **切分置信度**：候选切分点（medium）是否需要调整？
- [ ] **角色识别准确性**：VLM 是否认错人？（尤其侧面/逆光/夜景）
- [ ] **去重合理性**：最优 take 选择是否符合表演意图？
- [ ] **升格可行性**：升格镜头素材帧率是否 ≥60fps？
- [ ] **180度规则**：相邻镜头角色左右位置是否一致？
- [ ] **节奏呼吸感**：关键情节点后是否有足够停顿？
- [ ] **ANALYZED 数据**：转换而来的 Shot 是否已复核？

---

## 10. 开发路线

### v0.1 MVP（旧版）

- [x] Phase 1：基于 FFmpeg 场景检测的素材分析
- [x] Phase 2：三层去重机制
- [x] Phase 3：基础剪辑语法决策
- [x] Phase 4：EDL/FCPXML 输出

### v0.2 当前重构目标

- [x] Phase 0：纯 CV 粗剪，只拆硬切镜头，保护运动/叠化/淡入淡出
- [ ] 素材状态机：支持 RAW / PROCESSED / ANALYZED（Phase 0/1 落地）
- [ ] 先分析再切分：多维切分积分 + 长镜头保护
- [ ]  richer 镜头元数据：机位、主体、情绪、动作、台词、前后关系
- [ ] Provenance 溯源：每个 Shot 记录来源和切分理由
- [ ] Adapter 格式适配：接入外部/人工分析结果
- [ ] 配置化切分权重与阈值

### v0.3 优化

- [ ] 支持多机位同步对齐（时间码/音频波形）
- [ ] 智能 B-roll 推荐（从备选池自动插入 Cutaway）
- [ ] 音乐情绪匹配（根据情绪标签自动推荐 BGM）
- [ ] 180度规则自动检测与标记
- [ ] 交互式审核界面（Web UI 替代 JSON/CSV）

---

## 许可证

MIT License

---

> **最后提醒**：AI 是剪辑助理，不是剪辑师。它擅长处理信息，你擅长创作判断。最终的艺术决策，永远在你手中。
