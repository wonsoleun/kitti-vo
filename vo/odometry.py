"""스테레오 Visual Odometry.

프레임 i -> i+1 한 스텝:
  1) 왼쪽/오른쪽 영상으로 StereoSGBM 시차 -> 프레임 i 의 깊이 지도
  2) 왼쪽 영상 i, i+1 에서 특징점 검출 + 매칭 (여기만 특징점 종류에 따라 달라짐)
  3) 프레임 i 매칭점에 깊이를 붙여 3D 점으로 만들고,
     프레임 i+1 의 2D 점과 PnP + RANSAC 으로 상대 자세 추정
  4) 자세 누적:  T_world_{i+1} = T_world_i @ inv(T_{i+1 <- i})

추정이 실패하면(인라이어 부족) 직전 움직임을 그대로 쓰고(constant velocity) fallback 으로 기록합니다.
"""
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VOConfig:
    max_depth: float = 80.0          # 이보다 먼 점은 깊이 오차가 커서 버림 (m)
    min_inliers: int = 15            # PnP 인라이어가 이보다 적으면 실패로 처리
    reproj_error: float = 1.5        # RANSAC 재투영 오차 허용치 (px)
    num_disparities: int = 96        # SGBM 시차 탐색 범위 (16의 배수)
    block_size: int = 11             # SGBM 블록 크기 (홀수)


def _ms(t0, t1):
    return (t1 - t0) * 1000.0


class StereoVO:
    def __init__(self, seq, features, cfg=None):
        self.seq = seq
        self.fm = features
        self.cfg = cfg or VOConfig()
        bs = self.cfg.block_size
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.cfg.num_disparities,
            blockSize=bs,
            P1=8 * bs * bs,
            P2=32 * bs * bs,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        K = seq.K
        self.fx, self.fy, self.cx, self.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # ---- 단계별 함수 ----
    def depth_map(self, left, right):
        disp = self.sgbm.compute(left, right).astype(np.float32) / 16.0
        depth = np.zeros_like(disp)
        valid = disp > 0.5
        depth[valid] = self.fx * self.seq.baseline / disp[valid]
        return depth

    def estimate_motion(self, depth, pts0, pts1):
        """프레임 i 의 2D 점(pts0)+깊이 -> 3D, 프레임 i+1 의 2D 점(pts1) 으로 PnP.

        반환: (T_{i+1<-i} 4x4 또는 None, info dict)
        """
        info = {"n_matches": len(pts0), "n_depth": 0, "n_inliers": 0, "ok": False,
                "pts0": pts0, "pts1": pts1, "depth_mask": np.zeros(len(pts0), bool),
                "inlier_mask": np.zeros(len(pts0), bool)}
        if len(pts0) < 6:
            return None, info

        h, w = depth.shape
        u = np.clip(np.round(pts0[:, 0]).astype(int), 0, w - 1)
        v = np.clip(np.round(pts0[:, 1]).astype(int), 0, h - 1)
        z = depth[v, u]
        mask = (z > 0) & (z < self.cfg.max_depth)
        info["depth_mask"] = mask
        info["n_depth"] = int(mask.sum())
        if mask.sum() < 6:
            return None, info

        z = z[mask]
        obj = np.stack([(pts0[mask, 0] - self.cx) * z / self.fx,
                        (pts0[mask, 1] - self.cy) * z / self.fy,
                        z], axis=1).astype(np.float64)
        img = pts1[mask].astype(np.float64)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj, img, self.seq.K, None,
            reprojectionError=self.cfg.reproj_error,
            iterationsCount=300, confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        n_inl = 0 if inl is None else len(inl)
        info["n_inliers"] = n_inl
        if not ok or n_inl < self.cfg.min_inliers:
            return None, info

        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], self.seq.K, None, rvec, tvec)
        idx_depth = np.flatnonzero(mask)
        inlier_mask = np.zeros(len(pts0), bool)
        inlier_mask[idx_depth[inl]] = True
        info["inlier_mask"] = inlier_mask
        info["ok"] = True

        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(rvec)[0]
        T[:3, 3] = tvec.ravel()
        return T, info

    # ---- 시퀀스 전체 실행 ----
    def run(self, max_frames=None, log_every=100):
        n = len(self.seq) if max_frames is None else min(max_frames, len(self.seq))
        poses = [np.eye(4)]
        stats = []
        prev_motion = np.eye(4)

        left_i = self.seq.left(0)
        prev_feats = self.fm.detect(left_i)
        self.fm.sync()

        t_start = time.perf_counter()
        for i in range(n - 1):
            right_i = self.seq.right(i)
            left_next = self.seq.left(i + 1)  # 이미지 읽기 시간은 측정에서 제외

            t0 = time.perf_counter()
            depth = self.depth_map(left_i, right_i)
            t1 = time.perf_counter()
            feats_next = self.fm.detect(left_next)
            self.fm.sync()
            t2 = time.perf_counter()
            pts0, pts1 = self.fm.match(prev_feats, feats_next)
            self.fm.sync()
            t3 = time.perf_counter()
            T, info = self.estimate_motion(depth, pts0, pts1)
            t4 = time.perf_counter()

            fallback = T is None
            if fallback:
                T = prev_motion
            prev_motion = T
            poses.append(poses[-1] @ np.linalg.inv(T))

            stats.append({
                "frame": i + 1,
                "n_keypoints": self.fm.num_keypoints(feats_next),
                "n_matches": info["n_matches"],
                "n_with_depth": info["n_depth"],
                "n_inliers": info["n_inliers"],
                "fallback": int(fallback),
                "t_depth_ms": _ms(t0, t1),
                "t_detect_ms": _ms(t1, t2),
                "t_match_ms": _ms(t2, t3),
                "t_pose_ms": _ms(t3, t4),
                "t_total_ms": _ms(t0, t4),
            })
            prev_feats = feats_next
            left_i = left_next

            if log_every and (i + 1) % log_every == 0:
                el = time.perf_counter() - t_start
                print(f"  [{self.fm.name}] {i + 1}/{n - 1} 프레임 ({(i + 1) / el:.1f} fps)")

        return np.array(poses), stats

    def inspect_pair(self, i):
        """실패 분석용: 프레임 i -> i+1 한 스텝을 다시 계산하고 매칭/인라이어 정보를 돌려줌."""
        left_i, right_i, left_next = self.seq.left(i), self.seq.right(i), self.seq.left(i + 1)
        depth = self.depth_map(left_i, right_i)
        f0, f1 = self.fm.detect(left_i), self.fm.detect(left_next)
        pts0, pts1 = self.fm.match(f0, f1)
        T, info = self.estimate_motion(depth, pts0, pts1)
        return left_i, left_next, T, info
