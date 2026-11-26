"""스테레오 깊이 + 카메라 자세로 3D 점군 맵을 만들고 평가하는 함수들."""
import numpy as np
import cv2


def make_dense_sgbm(num_disparities=96, block_size=7):
    """맵용 SGBM. VO용보다 블록을 작게(디테일), speckle 필터로 잡음 제거."""
    bs = block_size
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=bs,
        P1=8 * bs * bs,
        P2=32 * bs * bs,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def backproject(depth, intensity, K, max_depth=30.0, min_depth=1.0, pixel_step=2, crop_left=96):
    """깊이 지도 -> 카메라 좌표계 3D 점 (N,3) 과 밝기 (N,).

    crop_left: SGBM 은 왼쪽 numDisparities 픽셀 열에서 시차를 구할 수 없어 잘라냄.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w = depth.shape
    v, u = np.mgrid[0:h:pixel_step, crop_left:w:pixel_step]
    z = depth[v, u]
    m = (z > min_depth) & (z < max_depth)
    u, v, z = u[m], v[m], z[m]
    pts = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1)
    return pts.astype(np.float32), intensity[v, u]


def transform(T, pts):
    return (pts @ T[:3, :3].T + T[:3, 3]).astype(np.float32)


def voxel_downsample(pts, vals, voxel):
    """같은 voxel 안의 점들을 평균 하나로 합침."""
    if len(pts) == 0:
        return pts, vals
    keys = np.floor(pts / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out = np.zeros((len(counts), 3), np.float64)
    np.add.at(out, inv, pts)
    out /= counts[:, None]
    v = np.bincount(inv, weights=vals.astype(np.float64)) / counts
    return out.astype(np.float32), v.astype(np.float32)


def nn_distances(src, dst):
    """src 각 점에서 dst 의 가장 가까운 점까지 거리."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(dst).query(src, k=1, workers=-1)
    return d


def map_quality(est_pts, gt_pts, tau=0.5):
    """복원 품질 지표 (Tanks and Temples 방식).

    accuracy     : 추정 맵의 점이 GT 맵에서 얼마나 떨어져 있나 (낮을수록 좋음)
    completeness : GT 맵의 점이 추정 맵으로 얼마나 덮이나 (낮을수록 좋음)
    chamfer      : 두 값의 평균
    precision/recall/F-score : 거리 tau 이내인 점의 비율
    """
    d_est = nn_distances(est_pts, gt_pts)
    d_gt = nn_distances(gt_pts, est_pts)
    precision = float(np.mean(d_est < tau))
    recall = float(np.mean(d_gt < tau))
    f = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "accuracy_mean_m": float(d_est.mean()),
        "accuracy_median_m": float(np.median(d_est)),
        "completeness_mean_m": float(d_gt.mean()),
        "completeness_median_m": float(np.median(d_gt)),
        "chamfer_m": float((d_est.mean() + d_gt.mean()) / 2),
        "tau_m": tau,
        "precision": precision,
        "recall": recall,
        "f_score": float(f),
    }, d_est


def write_ply(path, pts, colors):
    """바이너리 PLY 저장. colors: (N,3) uint8 RGB. MeshLab, CloudCompare, Open3D 에서 열림."""
    n = len(pts)
    data = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                              ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    data["x"], data["y"], data["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    data["red"], data["green"], data["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def gray_colors(vals):
    g = np.clip(vals, 0, 255).astype(np.uint8)
    return np.stack([g, g, g], axis=1)


def error_colors(d, vmax=1.0):
    """거리 오차 -> 파랑(정확) ~ 빨강(어긋남)."""
    t = np.clip(d / vmax, 0, 1)
    cmap = cv2.applyColorMap((t * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
    return cmap[:, ::-1].copy()  # BGR -> RGB
