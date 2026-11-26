"""스테레오 깊이 + VO 자세로 3D 점군 맵 복원, GT 자세 맵과 비교.

    python build_map.py --data-root D:/KITTI/dataset --seq 08 --feature orb
    python build_map.py --data-root D:/KITTI/dataset --seq 08 --feature orb --start 0 --end 1000   # 구간만

run_vo.py 결과(results/<seq>/<feature>/poses.txt)가 먼저 있어야 합니다.

하는 일:
  1) --every 프레임마다 SGBM 밀집 깊이 지도를 3D 점군으로 바꿈
  2) 같은 점군을 (a) VO 추정 자세, (b) GT 자세로 각각 월드 좌표계에 누적
     -> 두 맵의 차이는 오직 '자세 오차' 때문 (깊이는 똑같이 씀)
  3) voxel 로 정리 후 PLY 저장, Chamfer distance·F-score 로 비교

출력: results/<seq>/<feature>/map/   (--start/--end 로 구간을 주면 map_<start>_<end>/)
    map_est.ply        VO 자세로 만든 맵 (밝기 색)
    map_gt.ply         GT 자세로 만든 맵 (기준)
    map_error.ply      추정 맵을 오차로 색칠 (파랑: 정확, 빨강: 1 m 이상 어긋남)
    map_topdown.png    위에서 본 그림 3장 (GT 맵 / 추정 맵 / 오차)
    map_metrics.json   accuracy, completeness, chamfer, precision, recall, F-score
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from vo.dataset import KittiSequence, read_poses
from vo.mapping import (backproject, error_colors, gray_colors, make_dense_sgbm, map_quality,
                        transform, voxel_downsample, write_ply)
from vo.metrics import align_rigid


def topdown_png(path, gt_pts, est_pts, est_err, gt_traj, est_traj, title, vmax=1.0, max_plot=400_000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)

    def sub(p, *extra):
        if len(p) <= max_plot:
            return (p, *extra)
        i = rng.choice(len(p), max_plot, replace=False)
        return (p[i], *[e[i] for e in extra])

    g, = sub(gt_pts)
    e, err = sub(est_pts, est_err)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    # KITTI 카메라 좌표: y 가 아래쪽이므로 높이 = -y
    axes[0].scatter(g[:, 0], g[:, 2], c=-g[:, 1], s=0.2, cmap="viridis", linewidths=0)
    axes[0].plot(gt_traj[:, 0, 3], gt_traj[:, 2, 3], "k-", lw=1)
    axes[0].set_title("GT poses map (height color)")
    axes[1].scatter(e[:, 0], e[:, 2], c=-e[:, 1], s=0.2, cmap="viridis", linewidths=0)
    axes[1].plot(est_traj[:, 0, 3], est_traj[:, 2, 3], "k-", lw=1)
    axes[1].set_title("VO poses map (height color)")
    sc = axes[2].scatter(e[:, 0], e[:, 2], c=err, s=0.2, cmap="jet", vmin=0, vmax=vmax, linewidths=0)
    axes[2].set_title("VO map: distance to GT map (m)")
    fig.colorbar(sc, ax=axes[2], fraction=0.046)
    allp = np.concatenate([g[:, [0, 2]], e[:, [0, 2]]])
    lo, hi = allp.min(0), allp.max(0)
    c, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.05
    for ax in axes:
        ax.set_xlim(c[0] - half, c[0] + half)
        ax.set_ylim(c[1] - half, c[1] + half)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("z (m)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="스테레오 깊이 + VO 자세로 3D 맵 복원 및 평가")
    p.add_argument("--data-root", required=True)
    p.add_argument("--seq", default="00")
    p.add_argument("--feature", default="orb", help="results/<seq>/<feature>/poses.txt 를 사용")
    p.add_argument("--results", default="results")
    p.add_argument("--start", type=int, default=0, help="맵에 넣을 첫 프레임")
    p.add_argument("--end", type=int, default=None, help="맵에 넣을 마지막 프레임 (포함 안 함)")
    p.add_argument("--every", type=int, default=5, help="몇 프레임마다 깊이를 넣을지")
    p.add_argument("--max-depth", type=float, default=30.0, help="밀집 깊이는 멀수록 부정확해서 짧게")
    p.add_argument("--pixel-step", type=int, default=2, help="깊이 지도에서 몇 픽셀마다 점을 뽑을지")
    p.add_argument("--voxel", type=float, default=0.2, help="voxel 크기 (m)")
    p.add_argument("--tau", type=float, default=0.5, help="precision/recall 거리 기준 (m)")
    p.add_argument("--no-align", action="store_true",
                   help="추정 궤적을 GT 에 강체 정렬하지 않음 (기본은 ATE 와 같은 방식으로 정렬)")
    p.add_argument("--view", action="store_true", help="Open3D 가 설치돼 있으면 3D 창으로 보기")
    args = p.parse_args()

    seq = KittiSequence(args.data_root, args.seq)
    run_dir = Path(args.results) / seq.seq / args.feature
    if not (run_dir / "poses.txt").exists():
        raise SystemExit(f"{run_dir / 'poses.txt'} 가 없습니다. run_vo.py 를 먼저 실행하세요.")
    est = read_poses(run_dir / "poses.txt")
    end = min(args.end or len(est), len(est))
    if args.start >= end - 1:
        raise SystemExit(f"구간이 비었습니다 (start={args.start}, end={end}, 추정 궤적 길이={len(est)})")
    frames = list(range(args.start, end, args.every))
    gt = seq.gt_poses
    has_gt = gt is not None

    # 추정 궤적을 GT 에 강체 정렬 (ATE 와 같은 방식). 맵 비교가 '시작점 하나의 차이' 에 좌우되지 않게.
    est_used = est.copy()
    if has_gt and not args.no_align:
        R, t = align_rigid(est[args.start:end, :3, 3], gt[args.start:end, :3, 3])
        A = np.eye(4)
        A[:3, :3], A[:3, 3] = R, t
        est_used = A @ est

    full = args.start == 0 and end == len(est)
    out = run_dir / ("map" if full else f"map_{args.start}_{end}")
    out.mkdir(exist_ok=True)
    sgbm = make_dense_sgbm()
    nd = 96

    est_chunks, gt_chunks = [], []
    buf_est, buf_gt = [], []
    t0 = time.perf_counter()
    print(f"시퀀스 {seq.seq} / {args.feature}: 프레임 {args.start}~{end - 1}, {args.every} 프레임 간격 "
          f"({len(frames)} 장) 로 맵 생성")
    for k, i in enumerate(frames):
        left, right = seq.left(i), seq.right(i)
        disp = sgbm.compute(left, right).astype(np.float32) / 16.0
        depth = np.zeros_like(disp)
        valid = disp > 0.5
        depth[valid] = seq.K[0, 0] * seq.baseline / disp[valid]
        pts, vals = backproject(depth, left, seq.K, max_depth=args.max_depth,
                                pixel_step=args.pixel_step, crop_left=nd)
        buf_est.append((transform(est_used[i], pts), vals))
        if has_gt:
            buf_gt.append((transform(gt[i], pts), vals))

        # 메모리 절약: 50 장마다 voxel 로 압축
        if len(buf_est) == 50 or k == len(frames) - 1:
            for buf, chunks in ((buf_est, est_chunks), (buf_gt, gt_chunks)):
                if buf:
                    P = np.concatenate([b[0] for b in buf])
                    V = np.concatenate([b[1] for b in buf]).astype(np.float32)
                    chunks.append(voxel_downsample(P, V, args.voxel))
                    buf.clear()
            print(f"  {k + 1}/{len(frames)} 장 ({time.perf_counter() - t0:.0f} s)")

    def merge(chunks):
        P = np.concatenate([c[0] for c in chunks])
        V = np.concatenate([c[1] for c in chunks])
        return voxel_downsample(P, V, args.voxel)

    est_pts, est_vals = merge(est_chunks)
    write_ply(out / "map_est.ply", est_pts, gray_colors(est_vals))
    print(f"추정 맵: {len(est_pts):,} 점 -> {out / 'map_est.ply'}")

    metrics = {"sequence": seq.seq, "feature": args.feature, "start": args.start, "end": end,
               "every": args.every, "voxel_m": args.voxel, "max_depth_m": args.max_depth,
               "aligned_to_gt": bool(has_gt and not args.no_align), "n_points_est": int(len(est_pts))}

    if has_gt:
        gt_pts, gt_vals = merge(gt_chunks)
        write_ply(out / "map_gt.ply", gt_pts, gray_colors(gt_vals))
        q, d_est = map_quality(est_pts, gt_pts, tau=args.tau)
        write_ply(out / "map_error.ply", est_pts, error_colors(d_est, vmax=1.0))
        metrics.update({"n_points_gt": int(len(gt_pts)), **q})
        topdown_png(out / "map_topdown.png", gt_pts, est_pts, d_est,
                    gt[args.start:end], est_used[args.start:end],
                    title=f"seq {seq.seq} | {args.feature} | frames {args.start}-{end - 1}")
        print(f"GT 맵: {len(gt_pts):,} 점")
        print(f"  accuracy {q['accuracy_mean_m']:.3f} m | completeness {q['completeness_mean_m']:.3f} m | "
              f"chamfer {q['chamfer_m']:.3f} m | F@{args.tau}m {q['f_score']:.3f}")
    else:
        print("GT 포즈가 없어 비교는 건너뜁니다.")

    with open(out / "map_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"저장: {out}")

    if args.view:
        try:
            import open3d as o3d
        except ImportError:
            print("Open3D 가 없습니다: pip install open3d  (또는 .ply 를 MeshLab / CloudCompare 로 여세요)")
            return
        name = "map_error.ply" if has_gt else "map_est.ply"
        o3d.visualization.draw_geometries([o3d.io.read_point_cloud(str(out / name))])


if __name__ == "__main__":
    main()
