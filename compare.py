"""results/ 아래 모든 실행 결과를 모아 비교표와 그래프를 만듭니다.

    python compare.py --results results

출력:
    results/comparison.md   시퀀스별 표 + 특징점별 평균 표 (README 에 그대로 붙여 넣기 좋음)
    results/comparison.png  특징점별 정확도-속도 막대 그래프
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from vo.plotting import plot_comparison

COLS = [
    ("t_rel_percent", "t_rel (%)", "{:.2f}"),
    ("r_rel_deg_per_100m", "r_rel (°/100m)", "{:.2f}"),
    ("ate_rmse_m", "ATE (m)", "{:.2f}"),
    ("frontend_ms", "검출+매칭 (ms)", "{:.1f}"),
    ("fps_total", "전체 FPS", "{:.1f}"),
    ("mean_inliers", "평균 인라이어", "{:.0f}"),
    ("inlier_ratio", "인라이어 비율", "{:.2f}"),
    ("fallback_frames", "fallback", "{:d}"),
]


def fmt(v, f):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f.format(int(v)) if f == "{:d}" else f.format(v)


def table(rows, first_cols):
    head = "| " + " | ".join([c for c, _ in first_cols] + [c[1] for c in COLS]) + " |"
    sep = "|" + "---|" * (len(first_cols) + len(COLS))
    lines = [head, sep]
    for r in rows:
        cells = [str(r[k]) for _, k in first_cols] + [fmt(r.get(k), f) for k, _, f in COLS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    args = p.parse_args()
    root = Path(args.results)

    runs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(root.glob("*/*/metrics.json"))]
    if not runs:
        raise SystemExit(f"{root} 에 metrics.json 이 없습니다. run_vo.py 를 먼저 실행하세요.")

    runs.sort(key=lambda r: (r["sequence"], r["feature"]))

    # 특징점별 평균 (여러 시퀀스에 걸쳐)
    by_feat = defaultdict(list)
    for r in runs:
        by_feat[r["feature"]].append(r)
    order = [f for f in ("orb", "sift", "superpoint") if f in by_feat] + \
            [f for f in by_feat if f not in ("orb", "sift", "superpoint")]
    avg_rows = []
    for f in order:
        rs = by_feat[f]
        row = {"feature": f, "n_seq": len(rs)}
        for k, _, _ in COLS:
            vals = [r.get(k) for r in rs if r.get(k) is not None]
            vals = [v for v in vals if np.isfinite(v)]
            row[k] = float(np.mean(vals)) if vals else float("nan")
        row["fallback_frames"] = int(sum(r.get("fallback_frames", 0) for r in rs))
        avg_rows.append(row)

    seqs = sorted({r["sequence"] for r in runs})
    md = ["# 특징점별 스테레오 VO 비교", "",
          f"시퀀스: {', '.join(seqs)}  ",
          f"특징점 최대 개수: {sorted({r['max_kp'] for r in runs})}  ",
          f"장치: {sorted({r.get('device', 'cpu') for r in runs})}", "",
          "## 특징점별 평균", "",
          table(avg_rows, [("특징점", "feature"), ("시퀀스 수", "n_seq")]), "",
          "## 시퀀스별", "",
          table(runs, [("시퀀스", "sequence"), ("특징점", "feature")]), "",
          "- t_rel, r_rel: KITTI 공식 지표 (100~800 m 구간 평균). 낮을수록 좋음",
          "- 검출+매칭: 특징점 종류에 따라 달라지는 부분의 프레임당 시간",
          "- 전체 FPS: SGBM 깊이 계산까지 포함한 VO 한 스텝 속도",
          "- fallback: PnP 가 실패해 직전 움직임으로 대체한 프레임 수"]
    maps = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(root.glob("*/*/map*/map_metrics.json"))]
    maps = [m for m in maps if "chamfer_m" in m]
    if maps:
        md += ["", "## 3D 맵 품질 (build_map.py, GT 자세 맵 기준)", "",
               "| 시퀀스 | 특징점 | 프레임 | accuracy (m) | completeness (m) | chamfer (m) | F-score |",
               "|---|---|---|---|---|---|---|"]
        for m in sorted(maps, key=lambda m: (m["sequence"], m["feature"])):
            md.append(f"| {m['sequence']} | {m['feature']} | {m['start']}~{m['end'] - 1} | "
                      f"{m['accuracy_mean_m']:.3f} | {m['completeness_mean_m']:.3f} | "
                      f"{m['chamfer_m']:.3f} | {m['f_score']:.3f} (τ={m['tau_m']} m) |")
        md += ["", "- 두 맵은 같은 깊이 지도를 쓰고 자세만 달라서, 차이는 VO 자세 오차가 맵에 미친 영향입니다"]

    out_md = root / "comparison.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    plot_comparison(root / "comparison.png", avg_rows)

    print("\n".join(md[6:9]))
    print(f"\n저장: {out_md}, {root / 'comparison.png'}")


if __name__ == "__main__":
    main()
