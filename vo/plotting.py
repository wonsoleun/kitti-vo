import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_trajectory(path, est, gt=None, title=""):
    """위에서 내려다본 궤적 (KITTI 좌표: x 오른쪽, z 앞쪽)."""
    fig, ax = plt.subplots(figsize=(7, 7))
    if gt is not None:
        ax.plot(gt[:, 0, 3], gt[:, 2, 3], color="black", lw=2, label="Ground truth")
        ax.plot(gt[0, 0, 3], gt[0, 2, 3], "o", color="black")
    ax.plot(est[:, 0, 3], est[:, 2, 3], color="#2a7fd4", lw=1.5, label="Estimated")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_frame_errors(path, frame_err, stats, title=""):
    """프레임별 오차와 인라이어 수. 오차가 튀는 구간 = 실패 후보."""
    frames = [r["frame"] for r in frame_err]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(frames, [r["trans_err_m"] for r in frame_err], lw=0.8, color="#d4552a")
    axes[0].set_ylabel("trans err (m)")
    axes[1].plot(frames, [r["rot_err_deg"] for r in frame_err], lw=0.8, color="#7a3fbf")
    axes[1].set_ylabel("rot err (deg)")
    ax2 = axes[2]
    ax2.plot([s["frame"] for s in stats], [s["n_inliers"] for s in stats], lw=0.8, color="#2a7fd4",
             label="PnP inliers")
    fb = [s["frame"] for s in stats if s["fallback"]]
    if fb:
        ax2.plot(fb, [0] * len(fb), "x", color="red", label="fallback")
    ax2.set_ylabel("inliers")
    ax2.set_xlabel("frame")
    ax2.legend(loc="upper right")
    ax3 = ax2.twinx()
    ax3.plot(frames, [r["gt_rot_deg"] for r in frame_err], lw=0.6, color="gray", alpha=0.6)
    ax3.set_ylabel("GT rotation (deg/frame)", color="gray")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(path, rows):
    """특징점별 정확도-속도 비교 막대 그래프. rows: compare.py 의 특징점 평균 결과."""
    names = [r["feature"] for r in rows]
    colors = ["#2a7fd4", "#d4552a", "#3fa35a", "#7a3fbf"][: len(names)]
    panels = [("t_rel_percent", "t_rel (%)  ↓"),
              ("r_rel_deg_per_100m", "r_rel (deg/100m)  ↓"),
              ("frontend_ms", "detect + match (ms/frame)  ↓")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (key, label) in zip(axes, panels):
        vals = [r[key] for r in rows]
        bars = ax.bar(names, vals, color=colors)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}",
                        ha="center", va="bottom", fontsize=9)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
