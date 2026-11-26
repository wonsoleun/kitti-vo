"""KITTI Odometry 데이터 로더.

기대하는 폴더 구조 (KITTI Odometry 공식 압축을 풀면 이렇게 됩니다):

    <data_root>/
        sequences/00/image_0/000000.png   # 왼쪽 흑백 카메라
        sequences/00/image_1/000000.png   # 오른쪽 흑백 카메라
        sequences/00/calib.txt
        poses/00.txt                      # GT 궤적 (시퀀스 00~10만 존재)
"""
from pathlib import Path

import cv2
import numpy as np


def imread_gray(path):
    """한글/OneDrive 경로에서도 동작하는 흑백 이미지 읽기.

    Windows에서 cv2.imread는 경로에 한글이 있으면 None을 돌려주므로
    np.fromfile + cv2.imdecode를 사용합니다.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    return img


def imwrite(path, img):
    """한글 경로에서도 동작하는 이미지 저장."""
    path = Path(path)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise IOError(f"이미지를 인코딩할 수 없습니다: {path}")
    buf.tofile(str(path))


def read_calib(path):
    mats = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, vals = line.split(":", 1)
            nums = [float(v) for v in vals.split()]
            if len(nums) == 12:
                mats[key.strip()] = np.array(nums).reshape(3, 4)
    return mats


def read_poses(path):
    """KITTI 포즈 파일(한 줄에 3x4 행렬 12개 값) -> (N, 4, 4)."""
    data = np.loadtxt(path).reshape(-1, 3, 4)
    poses = np.tile(np.eye(4), (len(data), 1, 1))
    poses[:, :3, :] = data
    return poses


def write_poses(path, poses):
    """(N, 4, 4) -> KITTI 포즈 파일. evo_traj kitti 로 바로 읽을 수 있습니다."""
    np.savetxt(path, np.asarray(poses)[:, :3, :].reshape(len(poses), 12), fmt="%.6e")


class KittiSequence:
    def __init__(self, data_root, seq):
        self.root = Path(data_root)
        self.seq = f"{int(seq):02d}"
        self.seq_dir = self.root / "sequences" / self.seq
        if not self.seq_dir.exists():
            raise FileNotFoundError(
                f"{self.seq_dir} 가 없습니다. --data-root 는 'sequences' 와 'poses' 폴더가 "
                f"들어 있는 폴더(보통 .../dataset)여야 합니다."
            )
        self.left_files = sorted((self.seq_dir / "image_0").glob("*.png"))
        self.right_files = sorted((self.seq_dir / "image_1").glob("*.png"))
        if not self.left_files or len(self.left_files) != len(self.right_files):
            raise FileNotFoundError(
                f"{self.seq_dir} 의 image_0 / image_1 이미지가 없거나 개수가 다릅니다. "
                f"흑백(grayscale) 데이터셋을 받았는지 확인하세요."
            )

        calib_path = self.seq_dir / "calib.txt"
        if not calib_path.exists():
            raise FileNotFoundError(
                f"{calib_path} 가 없습니다. KITTI Odometry 페이지에서 calibration 파일을 받아 "
                f"같은 위치에 풀어 주세요."
            )
        calib = read_calib(calib_path)
        P0, P1 = calib["P0"], calib["P1"]
        self.K = P0[:, :3].copy()
        # P1 = K [I | -b*e_x]  =>  P1[0,3] = -fx * b
        self.baseline = -(P1[0, 3] - P0[0, 3]) / P1[0, 0]

        pose_path = self.root / "poses" / f"{self.seq}.txt"
        self.gt_poses = read_poses(pose_path) if pose_path.exists() else None

    def __len__(self):
        return len(self.left_files)

    def left(self, i):
        return imread_gray(self.left_files[i])

    def right(self, i):
        return imread_gray(self.right_files[i])
