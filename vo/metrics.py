"""평가 지표.

- kitti_metrics: KITTI 공식 devkit(evaluate_odometry.cpp)과 같은 방식.
  100, 200, ..., 800 m 구간마다 상대 오차를 구해 평균.
  t_rel = 이동 오차 (%), r_rel = 회전 오차 (deg / 100 m)
- ate_rmse: 궤적 전체를 강체 정렬(Umeyama, 스케일 없음)한 뒤 위치 오차 RMSE (m)
- per_frame_errors: 매 프레임 상대 움직임 오차. 실패 분석에 사용
"""
import numpy as np

KITTI_LENGTHS = (100, 200, 300, 400, 500, 600, 700, 800)


def rotation_angle(R):
    """회전 행렬의 회전각 (rad)."""
    c = (np.trace(R[:3, :3]) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def trajectory_distances(poses):
    steps = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def kitti_metrics(gt, est, lengths=KITTI_LENGTHS, step=10):
    assert len(gt) == len(est), "GT와 추정 궤적 길이가 다릅니다"
    dist = trajectory_distances(gt)
    errs = []  # (length, t_err_per_m, r_err_per_m)
    for first in range(0, len(gt), step):
        for L in lengths:
            last = int(np.searchsorted(dist, dist[first] + L, side="right"))
            if last >= len(gt):
                continue
            d_gt = np.linalg.inv(gt[first]) @ gt[last]
            d_est = np.linalg.inv(est[first]) @ est[last]
            e = np.linalg.inv(d_est) @ d_gt
            errs.append((L, np.linalg.norm(e[:3, 3]) / L, rotation_angle(e) / L))

    if not errs:
        # 궤적이 100 m 보다 짧으면 KITTI 지표를 계산할 수 없음 (--max-frames 가 작을 때)
        return {"t_rel": float("nan"), "r_rel": float("nan"), "n_segments": 0, "per_length": {}}

    errs = np.array(errs)
    per_length = {}
    for L in lengths:
        sel = errs[errs[:, 0] == L]
        if len(sel):
            per_length[int(L)] = {"t_rel": float(sel[:, 1].mean() * 100),
                                  "r_rel": float(np.degrees(sel[:, 2].mean()) * 100)}
    return {
        "t_rel": float(errs[:, 1].mean() * 100),               # %
        "r_rel": float(np.degrees(errs[:, 2].mean()) * 100),   # deg / 100 m
        "n_segments": int(len(errs)),
        "per_length": per_length,
    }


def align_rigid(src, dst):
    """src 점들을 dst 에 맞추는 회전 R, 이동 t (Umeyama, 스케일 없음)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    t = mu_d - R @ mu_s
    return R, t


def ate_rmse(gt, est):
    p_gt, p_est = gt[:, :3, 3], est[:, :3, 3]
    R, t = align_rigid(p_est, p_gt)
    aligned = (R @ p_est.T).T + t
    return float(np.sqrt(np.mean(np.sum((aligned - p_gt) ** 2, axis=1))))


def per_frame_errors(gt, est):
    """프레임 i-1 -> i 상대 움직임 오차 (i = 1..N-1)."""
    rows = []
    for i in range(1, len(gt)):
        d_gt = np.linalg.inv(gt[i - 1]) @ gt[i]
        d_est = np.linalg.inv(est[i - 1]) @ est[i]
        e = np.linalg.inv(d_est) @ d_gt
        rows.append({
            "frame": i,
            "trans_err_m": float(np.linalg.norm(e[:3, 3])),
            "rot_err_deg": float(np.degrees(rotation_angle(e))),
            "gt_step_m": float(np.linalg.norm(d_gt[:3, 3])),
            "gt_rot_deg": float(np.degrees(rotation_angle(d_gt))),  # 회전 속도 (deg/frame)
        })
    return rows


def evaluate(gt, est):
    km = kitti_metrics(gt, est)
    pf = per_frame_errors(gt, est)
    return {
        "t_rel_percent": km["t_rel"],
        "r_rel_deg_per_100m": km["r_rel"],
        "kitti_segments": km["n_segments"],
        "kitti_per_length": km["per_length"],
        "ate_rmse_m": ate_rmse(gt, est),
        "rpe_trans_mean_m": float(np.mean([r["trans_err_m"] for r in pf])),
        "rpe_rot_mean_deg": float(np.mean([r["rot_err_deg"] for r in pf])),
        "path_length_m": float(trajectory_distances(gt)[-1]),
    }
