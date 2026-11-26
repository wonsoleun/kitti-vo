"""교체 가능한 특징점 검출기 + 매칭기.

모든 클래스는 같은 인터페이스를 가집니다:
    feats = fm.detect(gray_image)          # 한 프레임에서 특징점 검출
    pts0, pts1 = fm.match(feats0, feats1)  # 두 프레임 매칭 결과 (N,2), (N,2) 픽셀 좌표
    fm.num_keypoints(feats)
    fm.sync()                              # GPU 사용 시 시간 측정을 위한 동기화
"""
import numpy as np
import cv2


class OpenCVFeatures:
    """ORB / SIFT + BFMatcher + Lowe ratio test."""

    def __init__(self, kind="orb", max_kp=2000, ratio=0.75):
        self.name = kind
        self.ratio = ratio
        if kind == "orb":
            self.detector = cv2.ORB_create(nfeatures=max_kp)
            norm = cv2.NORM_HAMMING
        elif kind == "sift":
            self.detector = cv2.SIFT_create(nfeatures=max_kp)
            norm = cv2.NORM_L2
        else:
            raise ValueError(f"알 수 없는 특징점 종류: {kind}")
        self.matcher = cv2.BFMatcher(norm)

    def detect(self, img):
        kps, desc = self.detector.detectAndCompute(img, None)
        pts = np.float32([k.pt for k in kps]).reshape(-1, 2)
        return {"pts": pts, "desc": desc}

    def num_keypoints(self, feats):
        return len(feats["pts"])

    def match(self, f0, f1):
        empty = np.zeros((0, 2), np.float32)
        if f0["desc"] is None or f1["desc"] is None or len(f0["pts"]) < 2 or len(f1["pts"]) < 2:
            return empty, empty
        knn = self.matcher.knnMatch(f0["desc"], f1["desc"], k=2)
        good = [p[0] for p in knn if len(p) == 2 and p[0].distance < self.ratio * p[1].distance]
        if not good:
            return empty, empty
        idx0 = np.array([m.queryIdx for m in good])
        idx1 = np.array([m.trainIdx for m in good])
        return f0["pts"][idx0], f1["pts"][idx1]

    def sync(self):
        pass


class SuperPointLightGlue:
    """학습 기반 특징점 SuperPoint + 학습 기반 매칭기 LightGlue.

    설치: pip install torch  (GPU가 있으면 CUDA 버전 권장)
          pip install git+https://github.com/cvg/LightGlue.git
    첫 실행 때 가중치를 자동으로 내려받습니다.
    """

    def __init__(self, max_kp=2000, device=None):
        try:
            import torch
            from lightglue import LightGlue, SuperPoint
            from lightglue.utils import rbd
        except ImportError as e:
            raise ImportError(
                "SuperPoint+LightGlue 를 쓰려면 다음을 설치하세요:\n"
                "  pip install torch\n"
                "  pip install git+https://github.com/cvg/LightGlue.git"
            ) from e
        self.name = "superpoint"
        self.torch = torch
        self.rbd = rbd
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.extractor = SuperPoint(max_num_keypoints=max_kp).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

    def detect(self, img):
        t = self.torch.from_numpy(img).float().div(255.0)[None].to(self.device)  # (1, H, W)
        with self.torch.inference_mode():
            # resize=None: 원본 해상도 그대로 (기본값은 긴 변 1024로 축소)
            return self.extractor.extract(t, resize=None)

    def num_keypoints(self, feats):
        return int(feats["keypoints"].shape[1])

    def match(self, f0, f1):
        with self.torch.inference_mode():
            out = self.matcher({"image0": f0, "image1": f1})
        out = self.rbd(out)
        m = out["matches"]
        if m.numel() == 0:
            empty = np.zeros((0, 2), np.float32)
            return empty, empty
        k0 = f0["keypoints"][0][m[:, 0]].cpu().numpy().astype(np.float32)
        k1 = f1["keypoints"][0][m[:, 1]].cpu().numpy().astype(np.float32)
        return k0, k1

    def sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()


FEATURES = ("orb", "sift", "superpoint")


def build_features(name, max_kp=2000, device=None):
    if name in ("orb", "sift"):
        return OpenCVFeatures(name, max_kp=max_kp)
    if name == "superpoint":
        return SuperPointLightGlue(max_kp=max_kp, device=device)
    raise ValueError(f"지원하는 특징점: {FEATURES}")
