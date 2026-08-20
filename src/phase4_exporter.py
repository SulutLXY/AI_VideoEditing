"""
Phase 4: 二次匹配与执行输出

将逻辑剪辑决策映射回物理素材，生成:
- EDL (CMX3600) 时间线
- FCPXML (Final Cut Pro / DaVinci Resolve)
- CSV 审核表
"""
import os
import csv
from typing import List, Dict
from datetime import datetime

from src.utils import Shot, sec_to_tc, tc_to_sec, run_ffmpeg, save_json, logger
from src.phase3_editor import EditDecision


class Phase4Exporter:
    """阶段4导出器"""

    def __init__(self, config: Dict):
        self.config = config
        self.project = config['project']
        self.output_dir = config['paths']['output']
        self.paths = config['paths']

    def run(self, decisions: List[EditDecision], shots: List[Shot]):
        """执行导出"""
        logger.info("=" * 60)
        logger.info("Phase 4: 二次匹配与执行输出")
        logger.info("=" * 60)

        shot_map = {s.shot_id: s for s in shots}

        # 二次匹配: 确认物理素材路径
        matched = []
        for d in decisions:
            shot = shot_map.get(d.shot_id)
            if not shot:
                logger.warning(f"找不到镜头 {d.shot_id}，跳过")
                continue

            # 验证物理文件存在
            if not os.path.exists(shot.source_path):
                logger.warning(f"素材文件不存在: {shot.source_path}")
                continue

            matched.append((d, shot))

        logger.info(f"二次匹配成功: {len(matched)}/{len(decisions)} 个决策")

        # 生成各种输出格式
        exports = []

        if self.config['processing'].get('export_edl', True):
            edl_path = self._export_edl(matched)
            exports.append(('EDL', edl_path))

        if self.config['processing'].get('export_fcpxml', True):
            fcpxml_path = self._export_fcpxml(matched)
            exports.append(('FCPXML', fcpxml_path))

        if self.config['processing'].get('export_csv', True):
            csv_path = self._export_final_csv(matched)
            exports.append(('CSV', csv_path))

        if self.config['processing'].get('export_json', True):
            json_path = self._export_json(matched)
            exports.append(('JSON', json_path))

        # 可选: 直接渲染代理文件
        # self._render_proxy(matched)

        logger.info("Phase 4 完成，导出文件:")
        for fmt, path in exports:
            logger.info(f"  [{fmt}] {path}")

    def _export_edl(self, matched: List[tuple]) -> str:
        """导出 CMX3600 EDL 格式"""
        edl_path = os.path.join(self.output_dir, 'timeline.edl')

        lines = [
            f"TITLE: {self.project['name']}",
            "FCM: NON-DROP FRAME",
            ""
        ]

        timeline_tc = 0.0  # 累积时间线时间（秒）
        fps = 24.0

        for idx, (decision, shot) in enumerate(matched, 1):
            # 解析源时间码
            src_in_sec = tc_to_sec(decision.tc_in, shot.fps)
            src_out_sec = tc_to_sec(decision.tc_out, shot.fps)

            # 应用速度调整
            speed = self._parse_speed(decision.speed)
            duration = (src_out_sec - src_in_sec) / speed

            # 时间线出点
            timeline_out = timeline_tc + duration

            # EDL 格式: 序号  源文件名  轨道  剪辑类型  源入点  源出点  时间线入点  时间线出点
            # 注意: EDL 文件名限制 8 字符，这里使用简化名
            reel_name = self._sanitize_reel_name(shot.source_file)

            lines.append(
                f"{idx:03d}  {reel_name:8s}  V     C        "
                f"{decision.tc_in} {decision.tc_out} "
                f"{sec_to_tc(timeline_tc, fps)} {sec_to_tc(timeline_out, fps)}"
            )

            # 速度注释
            if speed != 1.0:
                lines.append(f"* SPEED: {speed*100:.1f}%")

            # 源文件注释
            lines.append(f"* FROM CLIP NAME: {shot.source_file}")

            # 其他注释
            if decision.technique != "连续剪辑":
                lines.append(f"* TECHNIQUE: {decision.technique}")
            if decision.transition != "硬切":
                lines.append(f"* TRANSITION: {decision.transition}")
            if decision.audio not in ["保留原声", "保留"]:
                lines.append(f"* AUDIO: {decision.audio}")
            if decision.notes:
                lines.append(f"* NOTES: {decision.notes}")

            lines.append("")
            timeline_tc = timeline_out

        with open(edl_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        logger.info(f"EDL 已导出: {edl_path}")
        return edl_path

    def _sanitize_reel_name(self, filename: str) -> str:
        """将文件名转换为 EDL 兼容的 reel 名（8字符大写）"""
        name = os.path.splitext(filename)[0]
        # 移除非法字符，取前8位
        name = ''.join(c for c in name if c.isalnum()).upper()
        return name[:8]

    def _parse_speed(self, speed_str: str) -> float:
        """解析速度字符串为倍率"""
        speed_str = speed_str.strip().lower()
        if speed_str == '1x' or speed_str == '100%':
            return 1.0
        if '%' in speed_str:
            return float(speed_str.replace('%', '')) / 100.0
        if 'x' in speed_str:
            return float(speed_str.replace('x', ''))
        try:
            return float(speed_str)
        except:
            return 1.0

    def _export_fcpxml(self, matched: List[tuple]) -> str:
        """导出 FCPXML 格式（Final Cut Pro / DaVinci Resolve 兼容）"""
        fcpxml_path = os.path.join(self.output_dir, 'timeline.fcpxml')

        from lxml import etree

        # FCPXML 基础结构
        root = etree.Element('fcpxml', version='1.9')

        # 资源定义
        resources = etree.SubElement(root, 'resources')
        format_elem = etree.SubElement(resources, 'format', {
            'id': 'r1',
            'name': 'FFVideoFormat1080p24',
            'frameDuration': '1/24s',
            'width': '1920',
            'height': '1080'
        })

        # 为每个源文件创建资源引用
        file_resources = {}
        for idx, (_, shot) in enumerate(matched, 2):
            if shot.source_file not in file_resources:
                file_id = f"r{idx}"
                file_elem = etree.SubElement(resources, 'asset', {
                    'id': file_id,
                    'name': shot.source_file,
                    'src': f"file://{shot.source_path}",
                    'hasVideo': '1',
                    'hasAudio': '1',
                    'duration': f"{shot.duration_sec}s"
                })
                file_resources[shot.source_file] = file_id

        # 时间线
        library = etree.SubElement(root, 'library')
        event = etree.SubElement(library, 'event', name=self.project['name'])
        project_elem = etree.SubElement(event, 'project', name=self.project['name'])
        sequence = etree.SubElement(project_elem, 'sequence', {
            'duration': '0s',  # 稍后计算
            'format': 'r1',
            'tcStart': '0s',
            'tcFormat': 'NDF'
        })
        spine = etree.SubElement(sequence, 'spine')

        # 添加每个剪辑片段
        timeline_tc = 0.0
        fps = 24.0

        for decision, shot in matched:
            src_in_sec = tc_to_sec(decision.tc_in, shot.fps)
            src_out_sec = tc_to_sec(decision.tc_out, shot.fps)
            speed = self._parse_speed(decision.speed)
            duration = (src_out_sec - src_in_sec) / speed

            file_id = file_resources[shot.source_file]

            # 创建 clip 元素
            clip = etree.SubElement(spine, 'clip', {
                'name': decision.shot_id,
                'offset': f"{timeline_tc}s",
                'duration': f"{duration}s",
                'start': f"{src_in_sec}s",
                'tcStart': f"{src_in_sec}s"
            })

            # 视频引用
            video = etree.SubElement(clip, 'video')
            etree.SubElement(video, 'asset-clip', {
                'ref': file_id,
                'offset': f"{timeline_tc}s",
                'duration': f"{duration}s",
                'start': f"{src_in_sec}s"
            })

            # 速度效果
            if speed != 1.0:
                etree.SubElement(clip, 'speed', {
                    'rate': f"{speed}",
                    'type': 'conform' if speed > 1.0 else 'conform'
                })

            # 转场
            if decision.transition == "叠化":
                transition = etree.SubElement(spine, 'transition', {
                    'name': 'Cross Dissolve',
                    'offset': f"{timeline_tc}s",
                    'duration': '1s'
                })

            timeline_tc += duration

        # 写入文件
        tree = etree.ElementTree(root)
        tree.write(fcpxml_path, pretty_print=True, xml_declaration=True, encoding='UTF-8')

        logger.info(f"FCPXML 已导出: {fcpxml_path}")
        return fcpxml_path

    def _export_final_csv(self, matched: List[tuple]) -> str:
        """导出最终 CSV 审核表"""
        csv_path = os.path.join(self.output_dir, 'timeline_final.csv')

        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                '序列', '镜头ID', '源文件', '物理路径', '源入点', '源出点',
                '速度', '剪辑手法', '转场', '音频', '叙事目的', '备注', '审核状态'
            ])

            for idx, (decision, shot) in enumerate(matched, 1):
                writer.writerow([
                    idx,
                    decision.shot_id,
                    shot.source_file,
                    shot.source_path,
                    decision.tc_in,
                    decision.tc_out,
                    decision.speed,
                    decision.technique,
                    decision.transition,
                    decision.audio,
                    decision.purpose,
                    decision.notes,
                    ''  # 审核状态，人工填写
                ])

        logger.info(f"最终 CSV 已导出: {csv_path}")
        return csv_path

    def _export_json(self, matched: List[tuple]) -> str:
        """导出完整 JSON 数据"""
        json_path = os.path.join(self.output_dir, 'timeline.json')

        data = {
            'project': self.project,
            'export_time': datetime.now().isoformat(),
            'total_clips': len(matched),
            'timeline': []
        }

        timeline_tc = 0.0
        for decision, shot in matched:
            src_in_sec = tc_to_sec(decision.tc_in, shot.fps)
            src_out_sec = tc_to_sec(decision.tc_out, shot.fps)
            speed = self._parse_speed(decision.speed)
            duration = (src_out_sec - src_in_sec) / speed

            data['timeline'].append({
                'sequence': decision.sequence,
                'shot_id': decision.shot_id,
                'source_file': shot.source_file,
                'source_path': shot.source_path,
                'source_in': decision.tc_in,
                'source_out': decision.tc_out,
                'timeline_in': sec_to_tc(timeline_tc, 24.0),
                'timeline_out': sec_to_tc(timeline_tc + duration, 24.0),
                'speed': speed,
                'technique': decision.technique,
                'transition': decision.transition,
                'audio': decision.audio,
                'purpose': decision.purpose,
                'notes': decision.notes,
                'vlm_description': shot.vlm_description,
                'script_anchor': shot.script_anchor
            })

            timeline_tc += duration

        save_json(data, json_path)
        return json_path

    def _render_proxy(self, matched: List[tuple], resolution: str = "480p"):
        """渲染低分辨率代理文件用于快速预览（可选）"""
        proxy_dir = os.path.join(self.output_dir, 'proxy')
        os.makedirs(proxy_dir, exist_ok=True)

        logger.info("开始渲染代理文件...")
        for idx, (decision, shot) in enumerate(matched, 1):
            src_in_sec = tc_to_sec(decision.tc_in, shot.fps)
            src_out_sec = tc_to_sec(decision.tc_out, shot.fps)
            duration = src_out_sec - src_in_sec

            proxy_path = os.path.join(proxy_dir, f"{idx:03d}_{decision.shot_id}.mp4")

            try:
                run_ffmpeg([
                    "-ss", str(src_in_sec),
                    "-t", str(duration),
                    "-i", shot.source_path,
                    "-vf", f"scale=-2:{resolution.replace('p', '')}",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "28",
                    "-c:a", "aac", "-b:a", "128k",
                    proxy_path
                ])
                logger.info(f"代理文件: {proxy_path}")
            except Exception as e:
                logger.error(f"代理渲染失败 {decision.shot_id}: {e}")
