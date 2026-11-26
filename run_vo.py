"""스테레오 VO 실행 + 평가.

예시:
    python run_vo.py --data-root D:/KITTI/dataset --seq 00 --feature orb
    python run_vo.py --data-root D:/KITTI/dataset --seq 00 05 07 --feature orb sift superpoint
    python run_vo.py --data-root D:/KITTI/dataset --seq 00 --feature orb --max-frames 300   # 빠른 확인용

결과: results/<seq>/<feature>/
    poses.txt          추정 궤적 (KITTI 형식, evo 로 바로 읽힘)
    stats.csv          프레임별 특징점·매칭·인라이어 수, 단계별 처리 시간
    frame_errors.csv   프레임별 상대 움직임 오차 (GT 있을 때)
    metrics.json       t_rel, r_rel, ATE, RPE, FPS 요약
    trajectory.png     GT vs 추정 궤적
    frame_errors.png   프레임별 오차 + 인라이어 수
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from vo.dataset import KittiSequence, write_poses
from vo.features import FEATURES, build_features
from vo.metrics import evaluate, per_frame_errors
from vo.odometry import StereoVO, VOConfig
from vo.plotting import plot_frame_errors, plot_trajectory


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarize_timing(stats):
    def mean(key):
        return float(np.mean([s[key] for s in stats]))

    total = mean("t_total_ms")
    frontend = mean("t_detect_ms") + mean("t_match_ms")
    return {
        "t_depth_ms": mean("t_depth_ms"),
        "t_detect_ms": mean("t_detect_ms"),
        "t_match_ms": mean("t_match_ms"),
        "t_pose_ms": mean("t_pose_ms"),
        "t_total_ms": total,
        "frontend_ms": frontend,                 # 특징점 검출 + 매칭 (특징점 비교의 핵심)
        "fps_total": 1000.0 / total,             # SGBM 포함 전체
        "fps_frontend": 1000.0 / frontend if frontend > 0 else float("nan"),
        "mean_keypoints": mean("n_keypoints"),
        "mean_matches": mean("n_matches"),
        "mean_inliers": mean("n_inliers"),
        "inlier_ratio": float(np.sum([s["n_inliers"] for s in stats]) /
                              max(1, np.sum([s["n_matches"] for s in stats]))),
        "fallback_frames": int(np.sum([s["fallback"] for s in stats])),
    }


def run_one(args, seq_id, feature):
    seq = KittiSequence(args.data_root, seq_id)
    fm = build_features(feature, max_kp=args.max_kp, device=args.device)
    cfg = VOConfig(max_depth=args.max_depth, min_inliers=args.min_inliers)
    out = Path(args.out) / seq.seq / feature
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n=== 시퀀스 {seq.seq} | {feature} | {len(seq)} 프레임 | baseline {seq.baseline:.3f} m ===")
    poses, stats = StereoVO(seq, fm, cfg).run(max_frames=args.max_frames)

    write_poses(out / "poses.txt", poses)
    write_csv(out / "stats.csv", stats)

    metrics = {"sequence": seq.seq, "feature": feature, "max_kp": args.max_kp,
               "n_frames": len(poses), "device": str(getattr(fm, "device", "cpu")),
               **summarize_timing(stats)}

    gt = None
    if seq.gt_poses is not None:
        gt = seq.gt_poses[: len(poses)]
        metrics.update(evaluate(gt, poses))
        fe = per_frame_errors(gt, poses)
        write_csv(out / "frame_errors.csv", fe)
        if not args.no_plot:
            plot_frame_errors(out / "frame_errors.png", fe, stats, title=f"seq {seq.seq} | {feature}")
    else:
        print("  (GT 포즈가 없는 시퀀스라 정확도 평가는 건너뜁니다. GT는 00~10에만 있습니다.)")

    if not args.no_plot:
        plot_trajectory(out / "trajectory.png", poses, gt, title=f"seq {seq.seq} | {feature}")

    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    msg = (f"  -> 전체 {metrics['fps_total']:.1f} fps | 검출+매칭 {metrics['frontend_ms']:.1f} ms | "
           f"평균 인라이어 {metrics['mean_inliers']:.0f} | fallback {metrics['fallback_frames']}")
    if gt is not None:
        msg += (f"\n  -> t_rel {metrics['t_rel_percent']:.2f} % | r_rel {metrics['r_rel_deg_per_100m']:.2f} deg/100m"
                f" | ATE {metrics['ate_rmse_m']:.2f} m")
    print(msg)
    print(f"  저장: {out}")


def main():
    p = argparse.ArgumentParser(description="KITTI 스테레오 VO 특징점 비교 실험")
    p.add_argument("--data-root", required=True, help="'sequences' 와 'poses' 가 들어 있는 폴더")
    p.add_argument("--seq", nargs="+", default=["00"], help="시퀀스 번호들 (예: 00 05 07)")
    p.add_argument("--feature", nargs="+", default=["orb"], choices=FEATURES)
    p.add_argument("--max-kp", type=int, default=2000, help="특징점 최대 개수 (공정 비교를 위해 모두 동일)")
    p.add_argument("--max-frames", type=int, default=None, help="앞에서부터 이 프레임까지만 (빠른 확인용)")
    p.add_argument("--max-depth", type=float, default=80.0)
    p.add_argument("--min-inliers", type=int, default=15)
    p.add_argument("--device", default=None, help="superpoint 용: cuda 또는 cpu (기본: 자동)")
    p.add_argument("--out", default="results")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    for s in args.seq:
        for f in args.feature:
            run_one(args, s, f)


if __name__ == "__main__":
    main()
