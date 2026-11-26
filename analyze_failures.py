"""VO 가 어디서, 왜 틀리는지 분석합니다.

    python analyze_failures.py --data-root D:/KITTI/dataset --seq 00 --feature orb --top-k 10

run_vo.py 결과(results/<seq>/<feature>/)를 읽어서:
  1) 오차가 큰 프레임 상위 K개를 뽑고, 자동으로 원인 후보를 붙임
  2) 회전 속도 / 인라이어 수 구간별 평균 오차를 표로 정리
  3) 상위 K개 프레임의 매칭 결과를 이미지로 저장 (초록: 인라이어, 빨강: 아웃라이어)
     -> 이미지를 직접 보고 '움직이는 차', '그늘', '텍스처 없는 벽' 같은 원인을 확인하세요.

출력: results/<seq>/<feature>/failures/
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from vo.dataset import KittiSequence, imwrite, read_poses
from vo.features import FEATURES, build_features
from vo.metrics import per_frame_errors
from vo.odometry import StereoVO, VOConfig

SHARP_TURN_DEG = 1.5   # GT 회전 속도가 이 이상이면 급회전 (deg/frame, 10 Hz 기준 15 deg/s)
LOW_INLIERS = 50


def read_stats(path):
    with open(path, encoding="utf-8") as f:
        return {int(r["frame"]): r for r in csv.DictReader(f)}


def cause(err, st):
    if int(st["fallback"]):
        return "추정 실패 (인라이어 부족 → 직전 움직임으로 대체)"
    if err["gt_rot_deg"] >= SHARP_TURN_DEG:
        return "급회전"
    if int(st["n_inliers"]) < LOW_INLIERS:
        return "매칭 부족 (저텍스처·모션 블러·역광 의심)"
    return "확인 필요 (움직이는 물체·반복 패턴 등, 이미지로 확인)"


def binned(rows, key, edges, labels):
    lines = []
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        sel = [r for r in rows if lo <= r[key] < hi]
        if sel:
            lines.append(f"| {lab} | {len(sel)} | {np.mean([r['trans_err_m'] for r in sel]):.3f} | "
                         f"{np.mean([r['rot_err_deg'] for r in sel]):.3f} |")
    return lines


def draw_matches(img0, img1, info, title):
    """두 프레임을 위아래로 붙이고 매칭선을 그림."""
    h = img0.shape[0]
    canvas = cv2.cvtColor(np.vstack([img0, img1]), cv2.COLOR_GRAY2BGR)
    pts0, pts1 = info["pts0"], info["pts1"]
    for k in range(len(pts0)):
        inl = info["inlier_mask"][k]
        color = (60, 180, 60) if inl else (40, 40, 220)
        p0 = tuple(np.round(pts0[k]).astype(int))
        p1 = (int(round(pts1[k][0])), int(round(pts1[k][1])) + h)
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 2, color, -1)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(canvas, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--seq", default="00")
    p.add_argument("--feature", default="orb", choices=FEATURES)
    p.add_argument("--results", default="results")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-kp", type=int, default=2000, help="run_vo.py 와 같은 값으로")
    p.add_argument("--device", default=None)
    p.add_argument("--no-images", action="store_true")
    args = p.parse_args()

    seq = KittiSequence(args.data_root, args.seq)
    if seq.gt_poses is None:
        raise SystemExit("GT 포즈가 있는 시퀀스(00~10)에서만 실패 분석이 가능합니다.")
    run_dir = Path(args.results) / seq.seq / args.feature
    if not (run_dir / "poses.txt").exists():
        raise SystemExit(f"{run_dir} 에 결과가 없습니다. run_vo.py 를 먼저 실행하세요.")

    est = read_poses(run_dir / "poses.txt")
    gt = seq.gt_poses[: len(est)]
    errs = per_frame_errors(gt, est)
    stats = read_stats(run_dir / "stats.csv")
    for e in errs:
        e["n_inliers"] = int(stats[e["frame"]]["n_inliers"])

    top = sorted(errs, key=lambda r: r["trans_err_m"], reverse=True)[: args.top_k]
    out = run_dir / "failures"
    out.mkdir(exist_ok=True)

    md = [f"# 실패 분석: 시퀀스 {seq.seq} / {args.feature}", "",
          f"## 오차 상위 {len(top)} 프레임", "",
          "| 프레임 | 이동 오차 (m) | 회전 오차 (°) | GT 회전 속도 (°/frame) | 인라이어 | 원인 후보 |",
          "|---|---|---|---|---|---|"]
    for e in top:
        st = stats[e["frame"]]
        md.append(f"| {e['frame']} | {e['trans_err_m']:.3f} | {e['rot_err_deg']:.3f} | "
                  f"{e['gt_rot_deg']:.2f} | {st['n_inliers']} | {cause(e, st)} |")

    md += ["", "## 회전 속도별 평균 오차", "",
           "| GT 회전 속도 (°/frame) | 프레임 수 | 평균 이동 오차 (m) | 평균 회전 오차 (°) |",
           "|---|---|---|---|"]
    md += binned(errs, "gt_rot_deg", [0, 0.5, SHARP_TURN_DEG, 1e9],
                 ["직진 (< 0.5)", f"완만한 회전 (0.5 ~ {SHARP_TURN_DEG})", f"급회전 (≥ {SHARP_TURN_DEG})"])

    md += ["", "## 인라이어 수별 평균 오차", "",
           "| PnP 인라이어 | 프레임 수 | 평균 이동 오차 (m) | 평균 회전 오차 (°) |",
           "|---|---|---|---|"]
    md += binned(errs, "n_inliers", [0, LOW_INLIERS, 150, 1e9],
                 [f"< {LOW_INLIERS}", f"{LOW_INLIERS} ~ 150", "≥ 150"])

    if not args.no_images:
        print(f"상위 {len(top)} 프레임 매칭 이미지를 만드는 중...")
        fm = build_features(args.feature, max_kp=args.max_kp, device=args.device)
        vo = StereoVO(seq, fm, VOConfig())
        md += ["", "## 매칭 이미지", "", "초록: PnP 인라이어, 빨강: 아웃라이어 또는 깊이 없음", ""]
        for e in top:
            i = e["frame"] - 1  # frame k 의 오차 = 스텝 (k-1 -> k)
            img0, img1, _, info = vo.inspect_pair(i)
            title = (f"frame {i}->{i + 1} | trans err {e['trans_err_m']:.2f} m | "
                     f"inliers {info['n_inliers']}/{info['n_matches']}")
            name = f"frame_{e['frame']:06d}.png"
            imwrite(out / name, draw_matches(img0, img1, info, title))
            md.append(f"![frame {e['frame']}]({name})")

    (out / "failures.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[: 6 + len(top)]))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
