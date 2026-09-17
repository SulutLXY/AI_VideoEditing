"""批量验证抽帧行为；独立输出，保留正式产物。"""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.adaptive_frame_extractor import extract_adaptive_frames


def main():
    cfg = yaml.safe_load((ROOT / 'config/config.yaml').read_text(encoding='utf-8'))['phase0']['adaptive_frames']
    base = ROOT / 'workspace/output/phase0_rough_clips'
    output = ROOT / 'temp/phase0_speed_audit_v4'
    results = []
    for video in sorted(base.glob('S*/S*.mp4')):
        try:
            result = extract_adaptive_frames(str(video), str(output / video.stem), cfg)
            assert result, '未生成结果'
            frames = result['frames']
            errors = []
            for a, b in zip(frames, frames[1:]):
                gap = b['t'] - a['t']
                if gap <= 0:
                    errors.append('时间戳非递增')
            events = [x['t'] for x in frames if x.get('capture_reason') == 'event_onset']
            if any(b - a < cfg.get('event_cooldown', 0.5) - 1e-6 for a, b in zip(events, events[1:])):
                errors.append('事件采样冷却失效')
            if any('capture_reason' not in x for x in frames):
                errors.append('缺抽帧原因')
            if not result['motion']['sampling_audit']['first_last_covered']:
                errors.append('缺首尾帧')
            results.append(dict(shot=video.stem, motion=result['motion'], errors=errors))
            print(video.stem, len(frames), result['motion']['subject_speed'], errors, flush=True)
        except Exception as exc:
            results.append(dict(shot=video.stem, errors=[str(exc)]))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print('TOTAL', len(results), 'ERRORS', sum(bool(x['errors']) for x in results), flush=True)
    return int(any(x['errors'] for x in results))


if __name__ == '__main__':
    raise SystemExit(main())
