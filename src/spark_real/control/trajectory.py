"""
Trajectory recording and visualization for score execution.
"""

import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

logger = logging.getLogger(__name__)

# Resolve at import time relative to this file so it works on any machine.
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output" / "trajectories"


class TrajectoryRecorder:
    """
    Records robot state during execution and generates plots.
    """

    def __init__(self, robot):
        self.robot = robot
        self.start_time = time.time()
        self.data = {
            "timestamps": [],
            "tcp_poses": [],
            "joint_positions": [],
            "gripper_positions": [],
            "action_labels": [],
            "action_transitions": [],
        }
        self._current_label = "idle"

    def set_action_label(self, label: str):
        self._current_label = label

    def mark_transition(self, label: str):
        t = time.time() - self.start_time
        self.data["action_transitions"].append((t, label))

    def record(self):
        """
        Sample current robot state (~10Hz from motion wait loops).
        """
        try:
            obs = self.robot.get_observation()
            tcp = np.array(obs["tcp_pose"][:3])
            joints = np.array(obs["joint_positions"][:6])
            gripper = float(obs.get("gripper_position", 0.0))
            t = time.time() - self.start_time

            self.data["timestamps"].append(t)
            self.data["tcp_poses"].append(tcp.tolist())
            self.data["joint_positions"].append(joints.tolist())
            self.data["gripper_positions"].append(gripper)
            self.data["action_labels"].append(self._current_label)
        except Exception as e:
            logger.debug("Trajectory record skipped: %s", e)

    def save(self):
        """
        Save trajectory data (.npz) and plot (.png).
        """
        traj = self.data
        if len(traj["timestamps"]) < 2:
            logger.info(
                "Trajectory too short to save (%d samples)", len(traj["timestamps"])
            )
            return

        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = OUTPUT_DIR / ts_str
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamps = np.array(traj["timestamps"])
        tcp_poses = np.array(traj["tcp_poses"])
        joint_positions = np.array(traj["joint_positions"])
        gripper_positions = np.array(traj["gripper_positions"])
        action_labels = np.array(traj["action_labels"], dtype=object)

        npz_path = out_dir / "trajectory.npz"
        np.savez(
            npz_path,
            timestamps=timestamps,
            tcp_poses=tcp_poses,
            joint_positions=joint_positions,
            gripper_positions=gripper_positions,
            action_labels=action_labels,
        )
        logger.info(
            "Trajectory saved: %s (%d samples, %.1fs)",
            npz_path,
            len(timestamps),
            timestamps[-1],
        )

        try:
            _save_plot(out_dir, traj)
        except Exception as e:
            logger.warning("Failed to save trajectory plot: %s", e)


def _ema_smooth(data: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Apply exponential moving average smoothing.
    """
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def _padded_limits(data: np.ndarray, min_range: float = 0.05) -> tuple:
    """
    Axis limits with minimum range to avoid over-zooming on noise.
    """
    dmin, dmax = float(data.min()), float(data.max())
    drange = dmax - dmin
    if drange < min_range:
        center = (dmin + dmax) / 2
        return center - min_range / 2, center + min_range / 2
    padding = drange * 0.1
    return dmin - padding, dmax + padding


def _save_plot(out_dir: Path, trajectory: dict):
    """
    Save 4-subplot trajectory visualization.
    """
    tcp_poses = np.array(trajectory["tcp_poses"])
    timestamps = np.array(trajectory["timestamps"])
    gripper_positions = np.array(trajectory["gripper_positions"])
    transitions = trajectory.get("action_transitions", [])

    if len(tcp_poses) < 2:
        return

    x_raw, y_raw, z_raw = tcp_poses[:, 0], tcp_poses[:, 1], tcp_poses[:, 2]
    x = _ema_smooth(x_raw)
    y = _ema_smooth(y_raw)
    z = _ema_smooth(z_raw)

    x_lim = _padded_limits(x)
    y_lim = _padded_limits(y)
    z_lim = _padded_limits(z)

    trans_idx = [
        (int(np.argmin(np.abs(timestamps - t))), label) for t, label in transitions
    ]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(trans_idx), 1)))

    fig = plt.figure(figsize=(16, 10))

    # 3D trajectory
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.plot(x_raw, y_raw, z_raw, "gray", alpha=0.2, linewidth=0.5)
    scatter = ax1.scatter(x, y, z, c=timestamps, cmap="viridis", s=20)
    ax1.plot(x, y, z, "b-", alpha=0.5, linewidth=1.5)
    ax1.scatter([x[0]], [y[0]], [z[0]], c="green", s=100, marker="o", label="Start")
    ax1.scatter([x[-1]], [y[-1]], [z[-1]], c="red", s=100, marker="x", label="End")
    for ci, (idx, label) in enumerate(trans_idx):
        ax1.scatter(
            [x[idx]],
            [y[idx]],
            [z[idx]],
            color=colors[ci],
            s=120,
            marker="D",
            edgecolors="black",
            linewidths=0.5,
        )
        ax1.text(x[idx], y[idx], z[idx], f" {label}", fontsize=6, color=colors[ci])
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)
    ax1.set_zlim(z_lim)
    ax1.set_title("3D TCP Trajectory (EMA smoothed)")
    ax1.legend(loc="upper left", fontsize=7)
    plt.colorbar(scatter, ax=ax1, label="Time (s)", shrink=0.6)

    # XY top-down
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(x_raw, y_raw, "gray", alpha=0.3, linewidth=0.5)
    ax2.scatter(x, y, c=timestamps, cmap="viridis", s=20)
    ax2.plot(x, y, "b-", alpha=0.5, linewidth=1.5)
    ax2.scatter([x[0]], [y[0]], c="green", s=100, marker="o", label="Start")
    ax2.scatter([x[-1]], [y[-1]], c="red", s=100, marker="x", label="End")
    for ci, (idx, label) in enumerate(trans_idx):
        ax2.scatter(
            [x[idx]],
            [y[idx]],
            color=colors[ci],
            s=80,
            marker="D",
            edgecolors="black",
            linewidths=0.5,
        )
        ax2.annotate(
            label,
            (x[idx], y[idx]),
            fontsize=6,
            color=colors[ci],
            textcoords="offset points",
            xytext=(5, 5),
        )
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_xlim(x_lim)
    ax2.set_ylim(y_lim)
    ax2.set_title("Top View (XY)")
    ax2.legend(fontsize=7)
    ax2.ticklabel_format(useOffset=False, style="plain")

    # XZ side view
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(x_raw, z_raw, "gray", alpha=0.3, linewidth=0.5)
    ax3.scatter(x, z, c=timestamps, cmap="viridis", s=20)
    ax3.plot(x, z, "b-", alpha=0.5, linewidth=1.5)
    ax3.scatter([x[0]], [z[0]], c="green", s=100, marker="o", label="Start")
    ax3.scatter([x[-1]], [z[-1]], c="red", s=100, marker="x", label="End")
    for ci, (idx, label) in enumerate(trans_idx):
        ax3.scatter(
            [x[idx]],
            [z[idx]],
            color=colors[ci],
            s=80,
            marker="D",
            edgecolors="black",
            linewidths=0.5,
        )
        ax3.annotate(
            label,
            (x[idx], z[idx]),
            fontsize=6,
            color=colors[ci],
            textcoords="offset points",
            xytext=(5, 5),
        )
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Z (m)")
    ax3.set_xlim(x_lim)
    ax3.set_ylim(z_lim)
    ax3.set_title("Side View (XZ)")
    ax3.legend(fontsize=7)
    ax3.ticklabel_format(useOffset=False, style="plain")

    # Gripper timeline
    ax4 = fig.add_subplot(2, 2, 4)
    grip_pct = np.array(gripper_positions) * 100
    ax4.plot(timestamps, grip_pct, "g-", linewidth=2)
    ax4.fill_between(timestamps, 0, grip_pct, alpha=0.3)
    for ci, (t_trans, label) in enumerate(transitions):
        ax4.axvline(x=t_trans, color=colors[ci], linestyle="--", alpha=0.7, linewidth=1)
        ax4.text(
            t_trans,
            102,
            label,
            fontsize=5,
            rotation=45,
            ha="left",
            va="bottom",
            color=colors[ci],
        )
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Gripper (%)")
    ax4.set_title("Gripper Position Over Time")
    ax4.set_ylim(-5, 115)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = out_dir / "trajectory_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Trajectory plot saved: %s", plot_path)
