# LLM-AutoCut 重构计划书

> 版本：v0.2 执行计划  
> 目标：新增并定版 Phase 0 纯 CV 粗剪；重构 Phase 1，实现"先分析再切分"的多模态视频原生理解流程；同时把 RAW / PROCESSED / ANALYZED 三个状态开关彻底落地。Phase 2/3/4 当前接口已能衔接，本轮不改。  

---

## 1. 当前问题

1. **缺少前置粗剪步骤**：原始素材直接进 Phase 1，VLM 同时承担"切镜头"和"分析内容"，成本高且容易漏拆/误拆。
2. **Phase 1 分析维度仍偏静态**：VLM 基于抽帧图像做分析，缺少对视频时序连续性的原生理解，容易把动作连贯、情绪延续的长段落误判为可拆分。
3. **切分与分析的割裂**：CV 场景检测先给出候选切分点，VLM 再给出语义 segment，SplitScorer 只算 candidate 点，没有真正让"多模态理解结论"驱动切分。
4. **缺少连贯性判断**：没有显式计算"段落内部关联性强弱"，导致一镜到底/长镜头可能被误切。
5. **缺少镜头配置信息输出**：拆分后没有为每个镜头生成标准配置信息（时长、画质、画幅比例、镜头类型、分镜内容、前后关联等）。
6. **状态开关未彻底落地**：`RAW` / `PROCESSED` / `ANALYZED` 的分发逻辑已存在，但 `RAW` 流程仍依赖抽帧图像模型，未接入真正的视频原生理解模型；`ANALYZED` 的 adapter 覆盖度不足。

---

## 2. 本轮目标

### 2.1 Phase 0 定版目标（已完成）

- 用传统 CV（OpenCV 灰度直方图差异）对 RAW 素材做**纯硬切粗剪**。
- 不调用 VLM/LLM/ASR，不建立关系图，只输出物理片段 + 粗配置文件。
- 通过孤立峰值检测识别硬切点，通过持续高活动保护运动/叠化/淡入淡出/高速镜头。
- 对超过 3.5 秒的片段做第二轮低阈值 prominence 检测，减少漏拆。
- 参数全部配置化，写入 `config.yaml` → `phase0`。

### 2.2 Phase 1 重构目标

- 引入**低成本多模态视频原生理解模型**（video-native VLM），对 Phase 0 产出的粗剪片段或 PROCESSED/ANALYZED 素材做连续理解。
- 输出**段落级内容描述** + **段落内连贯性评分**。
- 基于连贯性评分 + 切分积分，决定哪些段落应该保留为一个镜头，哪些应该拆分。
- 对强连贯段落（一镜到底 / 长镜头）默认保护，不拆分。
- 拆分后，为每个镜头生成**标准配置信息**（时长、分辨率、帧率、画幅比例、镜头类型、机位、主体、情绪、动作、对白、前后关联等）。
- 调用 FFmpeg 按精确时间码拆分素材，并记录拆分后的物理片段路径。

### 2.3 状态开关落地目标

| 状态 | 行为 | 当前完成度 | 本轮动作 |
|---|---|---|---|
| `RAW` | Phase 0 CV 粗剪 → CV 预扫描 + 视频原生理解 + 切分积分 + FFmpeg 拆分 | 70% | 接入 video-native VLM，完善连贯性判断和配置输出 |
| `PROCESSED` | 只分析内容，不切分，整体作为一个 Shot | 80% | 补充关键帧提取和内部子段落标记 |
| `ANALYZED` | 只做格式转换/校验，不分析不切分 | 60% | 完善 adapter 错误处理，缺字段时 fallback 并标记复核 |

---

## 3. 核心设计

### 3.1 RAW 素材完整流程（Phase 0 → Phase 1）

```
输入视频
   │
   ▼
Phase 0: 纯 CV 粗剪
  ├─ 逐帧灰度直方图差异
  ├─ 孤立峰值硬切检测
  ├─ 持续高活动保护（运动/叠化/淡入淡出）
  ├─ 长片段低阈值二次检测
  └─ 输出粗剪片段 + phase0_rough_config.json
   │
   ▼
Phase 1: 视频原生理解
   │
   ▼
CV 预扫描
  ├─ 分辨率、帧率、码率、时长、画幅比例
  ├─ 视觉质量粗评
  └─ 输出 CVMetadata
   │
   ▼
Video-Native VLM 分析
  ├─ 输入：视频文件（或压缩后的代理文件）
  ├─ 输出：
  │   ├─ segments[]: 每个段落的内容描述
  │   ├─ 每个 segment 的连贯性评分 coherence_score (0-1)
  │   ├─ 每个 segment 的镜头类型、机位、主体、情绪、动作、对白
  │   └─ 建议切分点（弱关联边界）
   │
   ▼
切分积分决策
  ├─ 对弱关联边界计算六维切分积分
  ├─ 长镜头/高连贯 segment 提升阈值保护
  ├─ 合并过短片段
  └─ 输出最终切分点时间码
   │
   ▼
FFmpeg 拆分
  ├─ 按时间码生成独立片段
  ├─ 默认 -c copy，关键帧不对齐时 brief re-encode
  └─ 记录 split_clip_path
   │
   ▼
生成 Shot + 配置信息 + 关系图
```

### 3.2 切分优先级（六维积分）

```
cut_score =
    0.35 × camera_change      (机位切换)
  + 0.25 × subject_change     (主体/角色变化)
  + 0.15 × emotion_break      (情绪断裂)
  + 0.10 × plot_shift         (剧情节点偏移)
  + 0.10 × action_break       (动作不连续)
  + 0.05 × dialogue_break     (对话/音频断裂)
```

- 边界 `coherence_score < 0.4` 才进入切分积分计算。
- `score > 0.55`：确定切分
- `0.40 < score ≤ 0.55`：候选切分，建议复核
- 一镜到底 / 高连贯 segment：阈值上调 0.15

### 3.3 镜头配置信息（Shot Config）

每个 Shot 拆分后生成统一配置：

```yaml
shot_config:
  duration_sec: float           # 时长
  resolution: [w, h]            # 分辨率
  fps: float                    # 帧率
  aspect_ratio: str             # 画幅比例
  bitrate: str                  # 码率
  codec: str                    # 编码
  visual_quality: float         # 画质评分

  shot_type: str                # 镜头类型：特写/近景/中景/全景/大全景
  camera_position: str         # 机位
  camera_movement: str          # 运镜
  framing: str                  # 构图
  lighting: str                 # 光效
  color_tone: str               # 色调

  location: str                 # 场景地点
  time_of_day: str              # 时间
  characters: [str]             # 主体人物
  action: str                  # 动作
  emotion: str                  # 情绪
  dialogue: str                 # 对白
  key_objects: [str]            # 关键道具

  split_clip_path: str         # 拆分后的物理片段路径
  source_range: {tc_in, tc_out} # 在源文件中的时间码范围
  coherence_with_previous: float  # 与前一 Shot 的连贯性
  coherence_with_next: float       # 与后一 Shot 的连贯性
```

### 3.4 状态机行为

#### RAW
- 完整流程：Phase 0 纯 CV 粗剪 → CV 预扫描 → video-native VLM → 切分积分 → FFmpeg 拆分 → 生成 Shot + 配置 + 关系图。

#### PROCESSED
- 只分析整体内容，不切分。
- 整体作为一个 Shot，标记 `do_not_split=true`。
- 可包含 `internal_segments` 描述内部段落，但不生成独立 Shot。
- 提取关键帧供后续阶段使用。

#### ANALYZED
- 只读取已有分析文件，通过 Adapter 转换为标准 Shot。
- 不分析、不切分、不调用 VLM/ASR。
- 缺少必要字段时 fallback 为最小 Shot，并标记 `needs_review=true`。
- 支持的格式：custom_v1、csv_meta、manual_json。

---

## 4. 实施阶段

### 阶段 0：Phase 0 纯 CV 粗剪（已完成）

- [x] 实现 `RoughCutAnalyzer`（`src/phase0_rough_cut.py`），逐帧灰度直方图差异 + 孤立峰值检测。
- [x] 实现持续高活动保护，识别运动/叠化/淡入淡出/高速镜头，避免误切。
- [x] 实现长片段二次低阈值 prominence 检测，减少漏拆。
- [x] 用 FFmpeg 无损切分并输出 `phase0_rough_clips/` + `phase0_rough_config.json`。
- [x] 将 Phase 0 参数配置化，写入 `config.yaml` → `phase0`。
- [x] 端到端验证：多组 RAW 素材粗剪效果符合预期。

### 阶段 1：基础设施准备

- [ ] 新增 `VideoVLMService` 服务层（`src/services/video_vlm_service.py`），支持 video-native VLM 调用。
- [ ] 设计兼容抽帧 VLM 的 fallback：当 video-native 不可用时，可降级为现有抽帧方案。
- [ ] 更新 `config.yaml`：新增 `models.video_vlm` 配置块。

### 阶段 2：Phase 1 核心重构

- [ ] 在 `RawProcessor` 中接入 `VideoVLMService`，获取带连贯性评分的段落描述。
- [ ] 扩展 `Segment` 模型：增加 `coherence_score`、`is_long_take`、`shot_type` 等字段。
- [ ] 重写 `SplitScorer`：
  - 同时消费 VLM 段落边界和 CV 候选切分点；
  - 对弱关联边界计算六维积分；
  - 应用长镜头/高连贯保护；
  - 合并过短片段。
- [ ] 在 `RawProcessor` 中调用 FFmpeg 拆分，记录 `split_clip_path`。
- [ ] 为每个 Shot 填充完整 `shot_config` 信息。
- [ ] 更新 `RelationshipBuilder`，在拆分后的 Shot 之间建立前后关系。

### 阶段 3：PROCESSED / ANALYZED 开关完善

- [ ] `ProcessedProcessor`：补充关键帧提取，支持 `internal_segments` 记录。
- [ ] `AnalyzedProcessor`：完善 adapter 错误处理，缺字段时 fallback 并标记 `needs_review`。
- [ ] 增加 `AdapterRegistry` 便于扩展新格式。
- [ ] 补充 `tests/test_phase1.py` 单元测试，覆盖三种状态。

### 阶段 4：端到端验证与文档

- [ ] 用 mock / 本地视频跑一次 `main.py --phase 1`，验证输出 JSON 结构完整。
- [ ] 确保 Phase 1 输出能被 Phase 2 正常消费（`status` 字段默认值、关系图、script_anchor）。
- [ ] 更新 `PLAN.md` 和 `README.md` 中 Phase 1 相关描述。
- [ ] 全量单元测试通过。

---

## 5. 风险与注意事项

1. **Video-Native VLM 成本**：按视频时长计费，需限制单次分析时长或降采样到代理文件。
2. **FFmpeg 关键帧切分**：`-c copy` 切分点不在关键帧时可能花屏，需要检测并在必要时 brief re-encode。
3. **连贯性评分稳定性**：不同 VLM 对同一段落的连贯性评分可能波动，需要 prompt 约束和时序平滑。
4. **ANALYZED 数据质量**：转换而来的 Shot 必须明确标记 `needs_review`，避免直接进入剪辑决策。
5. **Phase 2/3/4 兼容**：本轮不改动它们，但要确保 Phase 1 输出字段与它们的读取逻辑对齐。

---

## 6. 下一步

Phase 0 已定版。下一步进入**阶段 1：基础设施准备**，开始实现 `VideoVLMService` 和在 `RawProcessor` 中接入视频原生理解流程。
