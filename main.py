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
import argparse
import os
import sys
import json

# 确保 src 在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import (
    load_config, parse_script_outline, logger, ensure_dir,
    init_logging, Shot, get_video_files, save_json,
)
from src.phase0_rough_cut import RoughCutAnalyzer
from src.phase1_analyzer import Phase1Analyzer
from src.phase2_dedup import Phase2TakeSelector
from src.phase2_inventory import MaterialInventoryBuilder
from src.phase3_editor import Phase3Editor, EditDecision
from src.phase4_exporter import Phase4Exporter
from src.services.llm_service import LLMService
from src.services.script_service import ScriptPreprocessor, read_script_file, ScriptReadError, ScriptParseError

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
        '--verbose', '-v',
        action='store_true',
        help='输出详细日志'
    )

    args = parser.parse_args()

    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        print("请复制 config/config.example.yaml 为 config/config.yaml 并填入 API Key")
        sys.exit(1)

    # 加载配置
    config = load_config(args.config)

    # 确保输出目录
    output_dir = config['paths']['output']
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, 'logs'))

    # 初始化日志
    init_logging(output_dir)
    if args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)

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
            # Phase 2 首次需要剧本大纲
            if script_beats is None:
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
                    sys.exit(1)

                logger.info(f"剧本解析完成: {len(script_beats)} 个情节点")
                for beat in script_beats:
                    logger.info(f"  - {beat.act} / {beat.beat_id}: {beat.content[:40]}...")

            if shots is None:
                # 1) 优先从 --input-json 或 phase1_analysis.json 加载
                input_path = args.input_json or os.path.join(output_dir, 'phase1_analysis.json')
                if os.path.exists(input_path):
                    with open(input_path, 'r') as f:
                        data = json.load(f)
                    shots = [Shot.from_dict(s) for s in data.get('shots', [])]
                    logger.info(f"从文件加载 {len(shots)} 个镜头: {input_path}")
                else:
                    # 2) 否则从素材库目录做 CV 轻量清点
                    materials_dir = args.materials_dir or config['paths'].get('raw_materials')
                    if materials_dir and os.path.isdir(materials_dir):
                        logger.warning(
                            "Phase 2 未找到 Phase 1 分析结果，将使用 CV 轻量清点作为 fallback。"
                            "此时镜头缺少 VLM 语义信息，剧本锚定精度会显著下降。"
                        )
                        logger.info(f"Phase 2 从素材库启动: {materials_dir}")
                        inventory = MaterialInventoryBuilder()
                        shots = inventory.build_shots_from_directory(materials_dir)
                        if shots:
                            inventory_path = os.path.join(output_dir, 'phase2_material_inventory.json')
                            save_json({"shots": [s.to_dict() for s in shots]}, inventory_path)
                            logger.info(f"已保存素材库清点结果: {inventory_path}")

                    if not shots:
                        print("错误: Phase 2 无法获取镜头列表，素材缺失。请提供以下至少一项：")
                        print("  - --input-json 或已存在的 phase1_analysis.json（推荐）")
                        print("  - --materials-dir 或配置中的 paths.raw_materials 指向有效视频目录")
                        sys.exit(1)

            dedup = Phase2TakeSelector(config)
            shots, report = dedup.run(shots, script_beats)

        elif phase == 3:
            if shots is None:
                # Phase 3 依赖 Phase 1 的完整 Shot 对象
                phase1_path = os.path.join(output_dir, 'phase1_analysis.json')
                if not os.path.exists(phase1_path):
                    print(f"错误: Phase 3 需要 Phase 1 的分析结果: {phase1_path}")
                    sys.exit(1)

                with open(phase1_path, 'r') as f:
                    shots = [Shot.from_dict(s) for s in json.load(f).get('shots', [])]
                logger.info(f"从 Phase 1 加载 {len(shots)} 个镜头用于剪辑决策")

            editor = Phase3Editor(config)
            decisions = editor.run(shots)

        elif phase == 4:
            if shots is None or decisions is None:
                # 尝试从文件加载
                phase1_path = os.path.join(output_dir, 'phase1_analysis.json')
                phase3_path = args.input_json or os.path.join(output_dir, 'phase3_edit_decision.json')

                if not os.path.exists(phase1_path) or not os.path.exists(phase3_path):
                    print("错误: Phase 4 需要 Phase 1 和 Phase 3 的结果，请按顺序运行")
                    sys.exit(1)

                with open(phase1_path, 'r') as f:
                    shots = [Shot.from_dict(s) for s in json.load(f).get('shots', [])]

                with open(phase3_path, 'r') as f:
                    decisions = [EditDecision(**d) for d in json.load(f).get('timeline', [])]

                logger.info(f"从文件加载 {len(shots)} 个镜头和 {len(decisions)} 个决策")

            exporter = Phase4Exporter(config)
            exporter.run(decisions, shots)

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
