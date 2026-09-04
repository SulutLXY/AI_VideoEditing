#!/usr/bin/env python3
"""
LLM-AutoCut: 基于大语言模型的影视后期智能剪辑系统

使用方法:
    python main.py --config config/config.yaml --all
    python main.py --config config/config.yaml --phase 1
    python main.py --config config/config.yaml --phase 2
    python main.py --config config/config.yaml --phase 3
    python main.py --config config/config.yaml --phase 4
"""
import sys
import os

# Windows 控制台默认使用 GBK，强制 UTF-8 避免中文日志乱码
if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import argparse
import json
import shutil
import time
import yaml

# 确保 src 在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import (
    load_config, parse_script_outline, logger, ensure_dir,
    init_logging, Shot, get_video_files, save_json, load_json,
    parse_duration_string, save_beats_review_md,
    fallback_vlm_to_local,
)
from src.models import ScriptBeat
from src.phase0_rough_cut import RoughCutAnalyzer
from src.phase1_analyzer import Phase1Analyzer
from src.phase2_dedup import Phase2TakeSelector
from src.phase2_inventory import MaterialInventoryBuilder
from src.dialogue_planner import DialoguePlanner
from src.phase3_subbeat import SubBeatSplitter
from src.services.llm_service import LLMService
from src.services.script_service import ScriptPreprocessor, read_script_file, ScriptReadError, ScriptParseError


def _import_phase3_editor():
    """延迟导入 Phase 3 编辑器，避免未安装本地模型依赖时无法运行 Phase 0/1/2"""
    from src.phase3_editor import Phase3Editor, EditDecision
    return Phase3Editor, EditDecision


def _import_phase4_exporter():
    """延迟导入 Phase 4 导出器"""
    from src.phase4_exporter import Phase4Exporter
    return Phase4Exporter


def _import_phase4_dubbing():
    """延迟导入 Phase 4 配音模块"""
    from src.phase4_dubbing import Phase4Dubbing
    return Phase4Dubbing


def clean_phase_outputs(output_dir: str, phase: int | None, all_phases: bool = False):
    """
    按阶段清理历史输出文件/目录，避免旧数据干扰。
    --all 时清理整个 output_dir；--phase N 时只清理 Phase N 的产物。
    注意：此方法不删除日志目录本身，仅在清理前把 pipeline.log 重命名为备份。
    """
    if not output_dir or not os.path.isdir(output_dir):
        return

    # 日志文件共享所有阶段，先备份再清空，避免旧阶段日志混入当前阶段
    log_file = os.path.join(output_dir, "logs", "pipeline.log")
    if os.path.exists(log_file):
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        bak_path = os.path.join(output_dir, "logs", f"pipeline_{timestamp}.log")
        try:
            shutil.move(log_file, bak_path)
        except Exception:
            try:
                os.remove(log_file)
            except Exception:
                pass

    if all_phases:
        # 清理整个输出目录，但保留 logs 目录（已备份主日志）
        for name in os.listdir(output_dir):
            target = os.path.join(output_dir, name)
            if os.path.basename(target) == "logs":
                continue
            try:
                if os.path.isdir(target):
                    shutil.rmtree(target, ignore_errors=True)
                else:
                    os.remove(target)
            except Exception:
                pass
        return

    # 阶段 -> 产物映射
    phase_artifacts = {
        0: [
            "phase0_rough_config.json",
            "phase0_rough_clips",
        ],
        1: [
            "phase1_analysis.json",
            "phase1_keyframes",
            "phase1_split_clips",
        ],
        2: [
            "script_beats_analysis.json",
            "dialogue_plan.json",
            "phase2_beats_review.md",
            "phase2_deduplication.json",
            "phase2_deduplication.csv",
            "phase2_selected_shots.json",
            "phase2_material_inventory.json",
            "phase2_anchor_corrections.csv",
        ],
        3: [
            "phase3_edit_decision.json",
            "phase3_edit_decision.csv",
            "phase3_missing_subbeats.json",
            "phase3_attempts",
            "timeline.json",
        ],
        4: [
            "timeline.edl",
            "timeline.fcpxml",
            "timeline_final.csv",
            "final_with_dubbing.mp4",
            "mixed_audio.wav",
            "dub_info.json",
        ],
    }

    for p in ([phase] if phase is not None else []):
        for name in phase_artifacts.get(p, []):
            target = os.path.join(output_dir, name)
            if not os.path.exists(target):
                continue
            try:
                if os.path.isdir(target):
                    shutil.rmtree(target, ignore_errors=True)
                else:
                    os.remove(target)
            except Exception as e:
                print(f"警告: 清理 {target} 失败: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="LLM-AutoCut: AI 辅助影视后期剪辑系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行完整工作流
  python main.py --config config/config.yaml --all

  # 仅运行素材分析（Phase 1）
  python main.py --config config/config.yaml --phase 1

  # 从已有分析结果继续去重（Phase 2）
  python main.py --config config/config.yaml --phase 2

  # 从已有去重结果生成剪辑方案（Phase 3）
  python main.py --config config/config.yaml --phase 3

  # 从已有剪辑方案导出时间线（Phase 4）
  python main.py --config config/config.yaml --phase 4
        """
    )

    parser.add_argument(
        '--config', '-c',
        default='config/config.yaml',
        help='配置文件路径 (默认: config/config.yaml)'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='运行完整四阶段工作流'
    )
    parser.add_argument(
        '--phase', '-p',
        type=int,
        choices=[0, 1, 2, 3, 4],
        help='仅运行指定阶段 (0=粗剪, 1=分析, 2=去重, 3=剪辑决策, 4=导出)'
    )
    parser.add_argument(
        '--input-json', '-i',
        help='指定上一阶段的 JSON 输入文件（用于从中间阶段开始）'
    )
    parser.add_argument(
        '--materials-dir', '-m',
        help='Phase 2 专用：指定素材库文件夹，未提供 phase1_analysis.json 时做 CV 轻量清点'
    )
    parser.add_argument(
        '--preprocess-script',
        help='把原始剧本文件解析为结构化 script.md 后退出'
    )
    parser.add_argument(
        '--export-anchor-corrections',
        action='store_true',
        help='导出 phase2_selected_shots.json 中每个镜头的锚定信息为可编辑 CSV，供用户手动校正'
    )
    parser.add_argument(
        '--apply-anchor-corrections',
        metavar='CSV_PATH',
        help='读取用户编辑后的锚定校正 CSV，将修正写回 phase2_selected_shots.json'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='输出详细日志'
    )
    parser.add_argument(
        '--clean', '-cl',
        action='store_true',
        help='运行前先清理本阶段生成的旧文件（--all 时清理全部输出，--phase N 时只清理 Phase N 的产出）'
    )

    args = parser.parse_args()

    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        print("请复制 config/config.example.yaml 为 config/config.yaml 并填入 API Key")
        sys.exit(1)

    # 加载配置
    config = load_config(args.config)

    # 未配置 VLM API Key 时自动回退到本地 VLM，并把更新写回配置文件
    if fallback_vlm_to_local(config):
        with open(args.config, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)

    # 确保输出目录
    output_dir = config['paths']['output']
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, 'logs'))

    # 如果用户要求清理，先按阶段清理历史产物，再初始化日志
    if args.clean:
        clean_phase_outputs(output_dir, args.phase, all_phases=args.all)
        print(f"已清理阶段输出: {'全部' if args.all else 'Phase ' + str(args.phase)}")

    # 初始化日志
    init_logging(output_dir)
    if args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)

    # 单独处理锚定校正导出/应用
    from src.phase2_anchor_editor import export_anchor_corrections, apply_anchor_corrections
    if args.export_anchor_corrections:
        csv_path = os.path.join(output_dir, "phase2_anchor_corrections.csv")
        try:
            export_anchor_corrections(output_dir, csv_path)
            print(f"已导出锚定校正表: {csv_path}")
            print("请用 Excel/WPS 编辑 corrected_* 列后，运行:")
            print(f"  python main.py --config {args.config} --apply-anchor-corrections {csv_path}")
            sys.exit(0)
        except Exception as e:
            print(f"导出锚定校正表失败: {e}")
            sys.exit(1)

    if args.apply_anchor_corrections:
        try:
            apply_anchor_corrections(output_dir, args.apply_anchor_corrections)
            print(f"锚定校正已应用: {args.apply_anchor_corrections}")
            print(f"已更新: {os.path.join(output_dir, 'phase2_selected_shots.json')}")
            print("建议重新运行 Phase 3/4 查看最新效果")
            sys.exit(0)
        except Exception as e:
            print(f"应用锚定校正失败: {e}")
            sys.exit(1)

    logger.info("=" * 70)
    logger.info(f"LLM-AutoCut 启动")
    logger.info(f"项目: {config['project']['name']}")
    logger.info(f"配置: {args.config}")
    logger.info("=" * 70)

    # 单独执行剧本预处理
    if args.preprocess_script:
        try:
            raw_text = read_script_file(args.preprocess_script)
        except ScriptReadError as e:
            print(f"错误: {e}")
            sys.exit(1)

        llm_service = LLMService(config)
        preprocessor_config = config.get("script_preprocessing", {})
        preprocessor = ScriptPreprocessor(llm_service, preprocessor_config)
        try:
            parsed = preprocessor.preprocess(raw_text)
        except ScriptParseError as e:
            print(f"剧本解析失败: {e}")
            sys.exit(1)

        script_path = config['paths']['script_outline']
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(parsed)
        print(f"剧本预处理完成，已保存至: {script_path}")
        sys.exit(0)

    # 剧本大纲只在 Phase 2 及以后使用
    script_beats = None

    # 确定运行阶段
    run_all = args.all
    run_phase = args.phase

    if not run_all and run_phase is None:
        print("错误: 请指定 --all 或 --phase")
        parser.print_help()
        sys.exit(1)

    phases_to_run = [0, 1, 2, 3, 4] if run_all else [run_phase]

    # 若后续阶段依赖剧本，提前统一加载，避免单独跑 Phase 3/4 时丢失剧本信息
    if any(p >= 2 for p in phases_to_run):
        # 优先使用已分析的剧本文件（含 dialogue_entries / 节奏分析），否则回退解析 script.md
        analyzed_script_path = os.path.join(output_dir, 'script_beats_analysis.json')
        if os.path.exists(analyzed_script_path) and script_beats is None:
            logger.info(f"预加载已分析的剧本文件: {analyzed_script_path}")
            try:
                analyzed_data = load_json(analyzed_script_path)
                script_beats = [ScriptBeat.from_dict(b) for b in analyzed_data.get("beats", [])]
                if script_beats:
                    logger.info(f"剧本分析数据加载完成: {len(script_beats)} 个情节点")
                    for beat in script_beats:
                        logger.info(f"  - {beat.act} / {beat.beat_id}: {beat.content[:40]}...")
                else:
                    logger.warning("剧本分析文件为空，回退解析 script.md")
            except Exception as e:
                logger.warning(f"加载剧本分析文件失败: {e}，回退解析 script.md")
                script_beats = None

        if script_beats is None:
            script_path = config['paths']['script_outline']
            if os.path.exists(script_path) and script_beats is None:
                logger.info(f"预加载剧本大纲: {script_path}")
                script_beats = parse_script_outline(script_path)
                if script_beats:
                    logger.info(f"剧本解析完成: {len(script_beats)} 个情节点")
                    for beat in script_beats:
                        logger.info(f"  - {beat.act} / {beat.beat_id}: {beat.content[:40]}...")
                else:
                    logger.warning("未能从剧本大纲解析出任何情节点，后续阶段可能无法生成剧本驱动配音")

    # 执行各阶段
    shots = None
    decisions = None

    for phase in phases_to_run:
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"执行 Phase {phase}")
        logger.info("=" * 70)

        if phase == 0:
            # Phase 0: 纯 CV 粗剪，只处理 RAW 素材
            raw_dir = config['paths'].get('raw_materials')
            if not raw_dir or not os.path.exists(raw_dir):
                print(f"错误: Phase 0 需要 RAW 素材目录: {raw_dir}")
                sys.exit(1)

            video_paths = get_video_files(raw_dir)
            if not video_paths:
                print(f"错误: RAW 素材目录中未找到视频文件: {raw_dir}")
                sys.exit(1)

            analyzer = RoughCutAnalyzer(config)
            shots = analyzer.run(video_paths)
            if not shots:
                logger.error("Phase 0 未产生任何粗剪片段")
                sys.exit(1)

        elif phase == 1:
            analyzer = Phase1Analyzer(config)
            shots = analyzer.run()

            if not shots:
                logger.error("Phase 1 未产生任何镜头分析结果，终止")
                sys.exit(1)

        elif phase == 2:
            # Phase 2: 剧本分析 + 节奏分析 + sub-beat 拆解
            if not script_beats:
                script_path = config['paths']['script_outline']
                if not os.path.exists(script_path):
                    print(f"错误: 剧本大纲不存在: {script_path}")
                    print("请创建剧本大纲文件（参考 README.md 格式）")
                    sys.exit(1)

                logger.info(f"解析剧本大纲: {script_path}")
                script_beats = parse_script_outline(script_path)
                if not script_beats:
                    print("错误: 未能从剧本大纲解析出任何情节点")
                    print("请检查剧本格式是否符合 README.md 中的示例")
                    print("如果是小说/分镜稿等非结构化文本，请先运行剧本预处理：python main.py --config <config> --preprocess-script <剧本文件>")
                    sys.exit(1)

                logger.info(f"剧本解析完成: {len(script_beats)} 个情节点")
                for beat in script_beats:
                    logger.info(f"  - {beat.act} / {beat.beat_id}: {beat.content[:40]}...")

            # 用 LLM 分析剧本节奏，为每个 beat 分配目标时长
            beat_analysis = {}
            try:
                llm_service = LLMService(config)
                target_duration = parse_duration_string(config['project'].get('target_duration', 0))
                beat_analysis = llm_service.analyze_script_beats(script_beats, target_duration)
                if beat_analysis:
                    for beat in script_beats:
                        info = beat_analysis.get(beat.beat_id)
                        if info:
                            beat.estimated_duration = info.get("estimated_duration", 0.0)
                            beat.pace = info.get("pace", "正常")
                            beat.emotion_intensity = info.get("emotion_intensity", 0.0)
                            beat.priority = info.get("priority", 3)
                            beat.required_shots_count = info.get("required_shots_count", 1)
                            beat.gender_state = info.get("gender_state", beat.gender_state or "")
                            beat.gender_transition = info.get("gender_transition", beat.gender_transition or "")
                            if info.get("key_actions"):
                                beat.key_actions = info.get("key_actions")
                    # 解析对白，生成 voice_cast 和 dialogue_entries
                    try:
                        planner = DialoguePlanner(config)
                        script_beats, _ = planner.plan(script_beats)
                    except Exception as e:
                        logger.warning(f"对白规划失败，将使用原始 key_dialogue: {e}")
            except Exception as e:
                logger.warning(f"剧本节奏分析失败，将使用平均时长分配: {e}")

            # 拆解 sub-beat，并附加到每个 beat 上
            try:
                splitter = SubBeatSplitter({"beats": [b.to_dict() for b in script_beats]})
                sub_beats = splitter.split()
                logger.info(f"sub-beat 拆解完成: {len(script_beats)} 个 beat -> {len(sub_beats)} 个 sub-beat")
                # 按 parent_beat_id 分组附加，转为 dict 以便序列化
                sub_beat_map = {}
                for sb in sub_beats:
                    sub_beat_map.setdefault(sb.parent_beat_id, []).append(sb.to_dict())
                for beat in script_beats:
                    beat.sub_beats = sub_beat_map.get(beat.beat_id, [])
            except Exception as e:
                logger.warning(f"sub-beat 拆解失败: {e}")

            save_json(
                {"beats": [b.to_dict() for b in script_beats], "analysis": beat_analysis},
                os.path.join(output_dir, 'script_beats_analysis.json')
            )
            logger.info(f"剧本节奏与分镜点分析完成，已保存: {os.path.join(output_dir, 'script_beats_analysis.json')}")

            # 生成用户可二次编辑的 beat 审核表
            review_path = os.path.join(output_dir, 'phase2_beats_review.md')
            save_beats_review_md(script_beats, review_path, beat_analysis)
            logger.info("各 beat 目标时长分配:")
            for beat in script_beats:
                logger.info(
                    f"  - {beat.beat_id}: {beat.estimated_duration:.1f}s, "
                    f"节奏={beat.pace}, 优先级={beat.priority}, 情绪={beat.emotion_intensity:.1f}, "
                    f"sub-beat={len(beat.sub_beats)} 个"
                )

        elif phase == 3:
            # Phase 3: 镜头筛选 + 排序 + 去重（原 Phase2TakeSelector）
            if shots is None:
                # 优先从 Phase 2 选择后的结果加载，回退到 Phase 1
                phase2_path = os.path.join(output_dir, 'phase2_selected_shots.json')
                phase1_path = os.path.join(output_dir, 'phase1_analysis.json')

                if os.path.exists(phase2_path):
                    with open(phase2_path, 'r', encoding='utf-8') as f:
                        shots = [Shot.from_dict(s) for s in json.load(f).get('shots', [])]
                    logger.info(f"从 Phase 2 加载 {len(shots)} 个带选择状态的镜头用于 Phase 3")
                elif os.path.exists(phase1_path):
                    with open(phase1_path, 'r', encoding='utf-8') as f:
                        shots = [Shot.from_dict(s) for s in json.load(f).get('shots', [])]
                    logger.warning(f"未找到 Phase 2 选择结果，从 Phase 1 回退加载 {len(shots)} 个镜头（状态均为候选）")
                else:
                    # 从素材库目录做 CV 轻量清点
                    materials_dir = args.materials_dir or config['paths'].get('raw_materials')
                    if materials_dir and os.path.isdir(materials_dir):
                        logger.info(f"Phase 3 从素材库启动: {materials_dir}")
                        inventory = MaterialInventoryBuilder()
                        shots = inventory.build_shots_from_directory(materials_dir)
                        if shots:
                            inventory_path = os.path.join(output_dir, 'phase2_material_inventory.json')
                            save_json({"shots": [s.to_dict() for s in shots]}, inventory_path)
                            logger.info(f"已保存素材库清点结果: {inventory_path}")

                    if not shots:
                        print("错误: Phase 3 无法获取镜头列表，素材缺失。请提供以下至少一项：")
                        print("  - --input-json 或已存在的 phase1_analysis.json / phase2_selected_shots.json（推荐）")
                        print("  - --materials-dir 或配置中的 paths.raw_materials 指向有效视频目录")
                        sys.exit(1)

            # 加载剧本节点（含 sub-beat）
            if not script_beats:
                script_path = config['paths']['script_outline']
                if os.path.exists(script_path):
                    script_beats = parse_script_outline(script_path)
                analysis_path = os.path.join(output_dir, 'script_beats_analysis.json')
                if os.path.exists(analysis_path):
                    try:
                        data = load_json(analysis_path)
                        script_beats = [ScriptBeat.from_dict(b) for b in data.get('beats', [])]
                        logger.info(f"从 Phase 2 加载 {len(script_beats)} 个剧本节点用于 Phase 3")
                    except Exception as e:
                        logger.warning(f"加载 script_beats_analysis.json 失败: {e}")

            if not script_beats:
                print("错误: Phase 3 需要 Phase 2 的剧本分析结果")
                sys.exit(1)

            beat_analysis = {}
            analysis_path = os.path.join(output_dir, 'script_beats_analysis.json')
            if os.path.exists(analysis_path):
                try:
                    beat_analysis = load_json(analysis_path).get('analysis', {})
                except Exception:
                    pass

            selector = Phase2TakeSelector(config)
            shots, report = selector.run(shots, script_beats, beat_analysis)

            # 把剧本节奏分析结果注入每个镜头的 script_anchor，供 Phase 4 使用
            if script_beats:
                beat_info = {
                    b.beat_id: {
                        "estimated_duration": b.estimated_duration,
                        "pace": b.pace,
                        "emotion_intensity": b.emotion_intensity,
                        "priority": b.priority,
                        "required_shots_count": b.required_shots_count,
                    }
                    for b in script_beats
                }
                for shot in shots:
                    if not shot.script_anchor:
                        continue
                    beat_id = shot.script_anchor.get("beat", "UNMATCHED")
                    if beat_id in beat_info:
                        shot.script_anchor.update(beat_info[beat_id])
                # 更新 phase2_selected_shots.json
                selected_shots_path = os.path.join(output_dir, 'phase2_selected_shots.json')
                save_json({"shots": [s.to_dict() for s in shots]}, selected_shots_path)

        elif phase == 4:
            # Phase 4: 剪辑决策 + 导出（原 Phase3Editor + Phase4Exporter + Phase4Dubbing）
            if shots is None:
                phase2_path = os.path.join(output_dir, 'phase2_selected_shots.json')
                phase1_path = os.path.join(output_dir, 'phase1_analysis.json')

                if os.path.exists(phase2_path):
                    with open(phase2_path, 'r', encoding='utf-8') as f:
                        shots = [Shot.from_dict(s) for s in json.load(f).get('shots', [])]
                    logger.info(f"从 Phase 3 加载 {len(shots)} 个带选择状态的镜头用于剪辑决策")
                elif os.path.exists(phase1_path):
                    with open(phase1_path, 'r', encoding='utf-8') as f:
                        shots = [Shot.from_dict(s) for s in json.load(f).get('shots', [])]
                    logger.warning(f"未找到 Phase 3 选择结果，从 Phase 1 回退加载 {len(shots)} 个镜头（状态均为候选）")
                else:
                    print(f"错误: Phase 4 需要 Phase 1/3 的分析结果: {phase1_path}")
                    sys.exit(1)

            if decisions is None:
                # 优先从 timeline.json 读取（含 passed 终审结果）
                _, EditDecision = _import_phase3_editor()
                phase3_path = args.input_json or os.path.join(output_dir, 'timeline.json')
                fallback_phase3_path = os.path.join(output_dir, 'phase3_edit_decision.json')

                if os.path.exists(phase3_path):
                    with open(phase3_path, 'r', encoding='utf-8') as f:
                        timeline_data = json.load(f)
                    decisions = [EditDecision(**d) for d in timeline_data.get('timeline', [])]
                    passed = timeline_data.get('passed', False)
                    total_score = timeline_data.get('total_score', 0.0)
                    logger.info(f"从 timeline.json 加载 {len(decisions)} 个剪辑决策")
                elif os.path.exists(fallback_phase3_path):
                    with open(fallback_phase3_path, 'r', encoding='utf-8') as f:
                        decisions = [EditDecision(**d) for d in json.load(f).get('timeline', [])]
                    passed = False
                    total_score = 0.0
                    logger.info(f"从 phase3_edit_decision.json 加载 {len(decisions)} 个剪辑决策")
                else:
                    # 没有缓存的剪辑决策，现场运行 Phase 3 编辑器
                    Phase3Editor, _ = _import_phase3_editor()
                    editor = Phase3Editor(config)
                    decisions = editor.run(shots)

            # Final Gate: 检查 Phase 3 是否通过 VLM 终审
            timeline_json_path = os.path.join(output_dir, 'timeline.json')
            if os.path.exists(timeline_json_path):
                with open(timeline_json_path, 'r', encoding='utf-8') as f:
                    timeline_data = json.load(f)
                passed = timeline_data.get('passed', False)
                total_score = timeline_data.get('total_score', 0.0)
                if not passed:
                    if not config.get('phase3', {}).get('enable_vlm_review', False):
                        logger.warning("[Phase4 Final Gate] VLM 终审未启用，跳过评分检查")
                    else:
                        logger.error(f"[Phase4 Final Gate] Phase 3 未通过终审 (score={total_score:.2f})，停止 Phase 4")
                        print(f"错误: Phase 3 未通过 VLM 终审 (score={total_score:.2f})，请在 output/phase3_attempts 中检查各轮尝试")
                        sys.exit(1)
            else:
                logger.warning("[Phase4 Final Gate] 未找到 timeline.json，跳过终审检查")

            Phase4Exporter = _import_phase4_exporter()
            exporter = Phase4Exporter(config)
            exporter.run(decisions, shots)

            # Phase 4 扩展：配音配乐合成
            audio_cfg = config.get("audio", {})
            if audio_cfg.get("enabled", True):
                try:
                    Phase4Dubbing = _import_phase4_dubbing()
                    dubbing = Phase4Dubbing(config)
                    dubbing.run(shots, decisions, script_beats=script_beats)
                except Exception as e:
                    logger.error(f"Phase 4 配音配乐失败: {e}")
                    logger.warning("已跳过配音配乐，基础导出文件仍然可用。")
            else:
                logger.info("Phase 4 配音配乐已禁用（config.audio.enabled=false）")

    logger.info("")
    logger.info("=" * 70)
    logger.info("LLM-AutoCut 全部完成!")
    logger.info(f"输出目录: {os.path.abspath(output_dir)}")
    logger.info("=" * 70)

    # 打印输出文件清单
    if os.path.exists(output_dir):
        print("\n输出文件:")
        for root, dirs, files in os.walk(output_dir):
            level = root.replace(output_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in sorted(files):
                if not file.endswith('.log'):
                    print(f"{subindent}{file}")


if __name__ == '__main__':
    main()
