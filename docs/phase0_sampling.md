# Phase 0 自动抽帧约定

每次通过 Phase 0 生成镜头时，`RoughCutAnalyzer._enrich_clip()` 自动调用抽帧器，
无需单独运行迁移脚本。当前版本为 `adaptive_v4`。

## 自动生效机制

- 缓存包含算法版本、完整有效配置、源文件路径、大小、修改时间指纹；任一变化会重抽。
- 缓存缺帧或缺 motion.json 时重抽；音频文件保留。
- 速度换算到 320 像素宽、30fps 基准，平滑窗口随帧率换算。
- 周期抽帧由局部档位控制。连续事件仅在起始沿补帧，事件间至少间隔 0.5 秒。
- 静止档的运动事件使用折算速度，避免原始光流绕过景别校正而连续抽帧。
- 首帧和尾帧强制覆盖；首尾约束允许最后两张的间距小于常规间隔。
- frames.json 中的 capture_reason 说明首尾、档位上升、周期、事件起始或内容变化。
- motion.json 的 sampling_audit 记录抽帧数量、事件数量和首尾覆盖；速度与档位冲突写入日志。

## 验证

运行 `.venv311/Scripts/python.exe scripts/audit_phase0_speed.py`，对现有镜头整批验证，
结果写入 `temp/phase0_speed_audit_v4/report.json`，不覆盖正式镜头。
运行 `.venv311/Scripts/python.exe -m unittest tests.test_adaptive_frames tests.test_phase0 -q`
检查回归约束。

## 适用范围

这是参考帧采样策略，不是播放倍速识别或可靠的景别分类器。低频能量仍是受纹理、
虚焦和画风影响的启发式指标。整批结构检查通过不能证明所有镜头的“普通/慢/快”
语义标签准确；目前 S036 仍是 slow，而用户标注为普通。不要用 subject_speed
单一均值标签直接决定成片播放倍速；混合速度镜头应参考逐帧档位。
