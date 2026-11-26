# KITTI Stereo VO: 특징점별 정확도-속도 비교

KITTI Odometry 스테레오 흑백 영상으로 Visual Odometry를 돌리면서 **특징점만 바꿔** 비교하는 실험 코드입니다.

- 특징점: **ORB / SIFT / SuperPoint + LightGlue** (학습 기반)
- 평가: KITTI 공식 지표 **t_rel(%)·r_rel(°/100m)**, ATE, RPE
- 속도: 단계별 처리 시간 (깊이, 검출, 매칭, 자세 추정), FPS
- 실패 분석: 오차가 큰 프레임 자동 추출, 회전 속도·인라이어 구간별 오차, 매칭 시각화
- **3D 맵 복원**: 스테레오 밀집 깊이 + VO 자세로 도로 구간 점군 맵을 만들고, GT 자세 맵과 Chamfer distance·F-score 로 비교

## 1. 설치 (Windows PowerShell 기준)

```powershell
cd kitti_vo_benchmark
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# SuperPoint + LightGlue 까지 비교하려면 (선택)
pip install torch            # GPU가 있으면 pytorch.org 에서 CUDA 버전 명령 사용
pip install git+https://github.com/cvg/LightGlue.git
```

## 2. 데이터

KITTI Odometry 페이지에서 세 가지를 받아 **같은 폴더에** 풀면 됩니다.

| 파일 | 크기 | 용도 |
|---|---|---|
| grayscale (`data_odometry_gray`) | 약 22 GB | 입력 영상 |
| calibration (`data_odometry_calib`) | 약 1 MB | 카메라 내부 파라미터, 스테레오 기준선 |
| ground truth poses (`data_odometry_poses`) | 약 4 MB | 평가용 정답 궤적 (00~10) |

```
D:/KITTI/dataset/
├── sequences/00/image_0/000000.png   ← 왼쪽
├── sequences/00/image_1/000000.png   ← 오른쪽
├── sequences/00/calib.txt
└── poses/00.txt
```

`--data-root` 에는 `sequences` 와 `poses` 가 들어 있는 폴더(위 예시의 `D:/KITTI/dataset`)를 넣습니다.
OneDrive나 한글이 들어간 경로도 그대로 쓸 수 있습니다.

## 3. 실행 순서

```powershell
# (1) 빠른 동작 확인: 앞 300 프레임만 (약 1분)
python run_vo.py --data-root D:/KITTI/dataset --seq 00 --feature orb --max-frames 300

# (2) 본 실험: 시퀀스 여러 개 × 특징점 3종
python run_vo.py --data-root D:/KITTI/dataset --seq 00 05 07 --feature orb sift superpoint

# (3) 비교표 + 그래프
python compare.py --results results

# (4) 실패 분석 (특징점별로)
python analyze_failures.py --data-root D:/KITTI/dataset --seq 00 --feature orb --top-k 10

# (5) 3D 맵 복원 + GT 맵과 비교 (특징점별로). 끝나면 compare.py 를 다시 돌리면 표에 추가됨
python build_map.py --data-root D:/KITTI/dataset --seq 00 --feature orb
python build_map.py --data-root D:/KITTI/dataset --seq 00 --feature orb --start 0 --end 1000   # 구간만
```

- 시퀀스 추천: **00**(도심, 회전 많음, 4541 프레임), **05**(도심, 2761), **07**(짧은 도심, 1101).
  시간이 부족하면 07부터 하세요.
- `--max-frames` 로 자른 궤적이 100 m보다 짧으면 KITTI 지표(t_rel, r_rel)는 `nan` 이 됩니다. ATE와 RPE는 계산됩니다.
- 모든 특징점은 `--max-kp 2000` (기본값)으로 개수를 맞춰 공정하게 비교합니다.

## 4. 결과물

```
results/
├── comparison.md          특징점별 평균 / 시퀀스별 표
├── comparison.png         t_rel · r_rel · 검출+매칭 시간 막대 그래프
└── 00/orb/
    ├── poses.txt          추정 궤적 (KITTI 형식)
    ├── metrics.json       요약 지표
    ├── stats.csv          프레임별 특징점·매칭·인라이어 수, 단계별 시간
    ├── frame_errors.csv   프레임별 상대 움직임 오차
    ├── trajectory.png     GT vs 추정 궤적
    ├── frame_errors.png   프레임별 오차 + 인라이어 + GT 회전 속도
    ├── failures/          analyze_failures.py 결과 (표 + 매칭 이미지)
    └── map/               build_map.py 결과 (구간을 주면 map_<start>_<end>/)
        ├── map_est.ply      VO 자세로 만든 점군 맵
        ├── map_gt.ply       GT 자세로 만든 점군 맵 (기준)
        ├── map_error.ply    추정 맵을 GT 맵과의 거리로 색칠 (파랑 정확 ~ 빨강 1 m 이상)
        ├── map_topdown.png  위에서 본 GT 맵 / 추정 맵 / 오차
        └── map_metrics.json accuracy, completeness, chamfer, precision, recall, F-score
```

`.ply` 파일은 MeshLab, CloudCompare(무료)로 열거나 `pip install open3d` 후 `--view` 옵션으로 바로 볼 수 있습니다.

evo 로 한 번 더 확인하고 싶으면:

```powershell
pip install evo
evo_traj kitti results/00/orb/poses.txt results/00/sift/poses.txt --ref D:/KITTI/dataset/poses/00.txt -p --plot_mode xz
evo_ape kitti D:/KITTI/dataset/poses/00.txt results/00/orb/poses.txt -a --plot --plot_mode xz
```

## 5. 파이프라인

프레임 i → i+1 한 스텝마다:

1. **깊이**: 왼쪽/오른쪽 영상으로 StereoSGBM 시차 계산 → 깊이 = fx · baseline / 시차
2. **특징점 검출·매칭**: 왼쪽 영상 i, i+1 사이 (이 단계만 특징점 종류에 따라 달라짐)
3. **자세 추정**: 프레임 i 매칭점에 깊이를 붙여 3D로 만들고, 프레임 i+1 의 2D 점과 **PnP + RANSAC** → LM 정제
4. **누적**: `T_world(i+1) = T_world(i) · inv(T(i+1←i))`

인라이어가 15개보다 적으면 실패로 보고 직전 움직임을 그대로 씁니다(`fallback`). 몇 프레임에서 일어났는지 기록됩니다.

| 파일 | 내용 |
|---|---|
| `vo/dataset.py` | KITTI 로더 (calib, poses, 한글 경로 대응) |
| `vo/features.py` | ORB / SIFT / SuperPoint+LightGlue 공통 인터페이스 |
| `vo/odometry.py` | 스테레오 VO 본체 |
| `vo/metrics.py` | KITTI devkit 방식 t_rel·r_rel, ATE(강체 정렬), 프레임별 오차 |
| `vo/plotting.py` | 그래프 |
| `vo/mapping.py` | 깊이 → 점군 변환, voxel 정리, 맵 품질 지표, PLY 저장 |

### 3D 맵 복원 (`build_map.py`)

1. `--every` 프레임(기본 5)마다 SGBM **밀집** 깊이 지도의 픽셀들을 3D 점으로 바꿉니다 (30 m 이내, 잡음 필터 적용)
2. 같은 점들을 **VO 추정 자세**와 **GT 자세**로 각각 월드 좌표계에 누적합니다
3. 0.2 m voxel 로 겹치는 점을 정리하고 두 맵을 비교합니다
   - accuracy: 추정 맵의 점 → GT 맵까지 평균 거리
   - completeness: GT 맵의 점 → 추정 맵까지 평균 거리
   - chamfer: 둘의 평균, F-score: 0.5 m 이내 점의 비율로 계산

두 맵은 **같은 깊이 지도**를 쓰고 자세만 다르므로, 차이는 순수하게 VO 자세 오차가 맵 품질에 미친 영향입니다.
추정 궤적은 ATE 와 같은 방식(강체 정렬)으로 GT 에 맞춘 뒤 비교합니다 (`--no-align` 으로 끌 수 있음).
voxel 크기 때문에 자세가 거의 정확해도 chamfer 가 0 이 되지는 않으니, **특징점끼리 상대 비교**로 보세요.
이것은 스테레오 깊이 기반 맵 복원이며, 여러 시점의 자세와 구조를 함께 최적화하는 SfM 과는 다릅니다.

## 6. 실험 아이디어 (다음 단계)

- `--max-kp` 를 500 / 1000 / 2000 으로 바꿔 특징점 개수에 따른 정확도-속도 곡선 그리기
- ORB 특징점이 한곳에 몰리는 문제 → 격자(grid) 단위로 고르게 뽑기
- 최근 N 프레임 **Local Bundle Adjustment** 추가 → 누적 오차 감소량 측정
- 라즈베리파이 등 엣지 기기에서 FPS 실측 (SuperPoint는 ONNX 변환)
- 3D 복원: COLMAP 으로 짧은 구간 SfM → KITTI LiDAR 점군과 비교

## 검증 상태

- 합성 KITTI 형식 시퀀스(150 프레임)로 ORB·SIFT 전체 파이프라인, 비교표, 실패 분석, 3D 맵 복원까지 실행 확인
- 맵 지표 확인: 같은 점군 → chamfer 0 / F-score 1, 0.3 m 평행이동 → accuracy 0.29 m
- 평가 지표 확인: 같은 궤적 → 오차 0, 스케일 2% 오차 → t_rel 1.98%
- SuperPoint + LightGlue 경로는 LightGlue 공식 사용법에 맞춰 작성했지만, 실제 실행 확인은 아직 하지 않았습니다. 처음 돌릴 때 `--max-frames 50` 으로 먼저 확인하세요.
