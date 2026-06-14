import os
from pathlib import Path
from typing import cast

from matplotlib.figure import Figure
import numpy as np
import os
from pathlib import Path
from typing import cast

from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

import gymnasium as gym

from coco_rocket_lander.algs.controller import Controller
from coco_rocket_lander.env.env_cfg import UserArgs
from coco_rocket_lander.env.rocketlander import RocketLander


def plot_trajectory_sets(
        trajectories: list[pd.DataFrame],
        title: str | None = None,
        ax: Axes | None = None,
        arrow_count: int = 20,
        arrow_scale: float = 1.5,
        trajectory_labels: list[str] | None = None,
        save_dir: str | None = None,
) -> Figure | None:
    if not trajectories:
        raise ValueError("trajectories must contain at least one DataFrame")

    if trajectory_labels is not None and len(trajectory_labels) != len(trajectories):
        raise ValueError("trajectory_labels must match the number of flight data sets")

    def get_n_distinct_colors(n: int) -> list[tuple[float, float, float]]:
        if n <= 0:
            return []
        colors = plt.cm.get_cmap("tab10", n)
        return [colors(i)[:3] for i in range(n)]

    trajectory_colors = get_n_distinct_colors(len(trajectories))

    x_values: list[np.ndarray] = []
    y_values: list[np.ndarray] = []
    for flight_data in trajectories:
        valid_positions = flight_data.dropna(subset=["x", "y"])
        if valid_positions.empty:
            continue
        x_values.append(valid_positions["x"].to_numpy(dtype=float))
        y_values.append(valid_positions["y"].to_numpy(dtype=float))
        if "goal_x" in flight_data.columns and "goal_y" in flight_data.columns:
            x_values.append(np.asarray([float(flight_data["goal_x"].iloc[0])], dtype=float))
            y_values.append(np.asarray([float(flight_data["goal_y"].iloc[0])], dtype=float))

    created_fig = False
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7))
        created_fig = True

    for index, flight_data in enumerate(trajectories):
        required_columns = {"x", "y", "theta", "main_engine_thrust", "nozzle_angle"}
        missing_columns = required_columns.difference(flight_data.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns in flight data set {index}: {sorted(missing_columns)}"
            )

        valid_positions = flight_data.dropna(subset=["x", "y"])
        if valid_positions.empty:
            continue

        trajectory_color = trajectory_colors[index]
        trajectory_label = (
            trajectory_labels[index] if trajectory_labels is not None else f"Trajectory {index + 1}"
        )

        ax.plot(
            valid_positions["x"],
            valid_positions["y"],
            color=trajectory_color,
            linewidth=2.0,
            alpha=0.9,
            label=trajectory_label,
        )

        start_row = valid_positions.iloc[0]
        end_row = valid_positions.iloc[-1]
        goal_x = float(flight_data["goal_x"].iloc[0]) if "goal_x" in flight_data.columns else np.nan
        goal_y = float(flight_data["goal_y"].iloc[0]) if "goal_y" in flight_data.columns else np.nan

        ax.scatter(
            start_row["x"],
            start_row["y"],
            s=90,
            marker="o",
            color=trajectory_color,
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
            label="Start position" if index == 0 else None,
        )
        ax.scatter(
            end_row["x"],
            end_row["y"],
            s=100,
            marker="X",
            color=trajectory_color,
            edgecolors="black",
            linewidths=1.0,
            zorder=6,
            label="End position" if index == 0 else None,
        )
        if np.isfinite(goal_x) and np.isfinite(goal_y):
            ax.scatter(
                goal_x,
                goal_y,
                s=130,
                marker="*",
                color="gold",
                edgecolors="black",
                linewidths=1.0,
                zorder=7,
                label="Goal position" if index == 0 else None,
            )

        control_rows = flight_data.dropna(subset=["main_engine_thrust", "nozzle_angle", "theta"])
        control_rows = control_rows.copy()
        control_rows["nozzle_angle"] = control_rows["theta"] + control_rows["nozzle_angle"]
        if control_rows.empty:
            continue

        sample_count = min(arrow_count, len(control_rows))
        sample_indices = np.unique(np.linspace(0, len(control_rows) - 1, sample_count, dtype=int))
        sampled_rows = control_rows.iloc[sample_indices]

        for row_number, (_, row) in enumerate(sampled_rows.iterrows()):
            thrust_strength = float(np.clip(row["main_engine_thrust"], 0.0, 1.0))
            nozzle_deflection = float(np.clip(row["nozzle_angle"], -1.0, 1.0))
            world_nozzle_angle = float(row["theta"] + nozzle_deflection * (15 * np.pi / 180))

            thrust_vector_scale = 0.15 + arrow_scale * thrust_strength
            arrow_dx = np.sin(world_nozzle_angle) * thrust_vector_scale
            arrow_dy = -np.cos(world_nozzle_angle) * thrust_vector_scale

            ax.quiver(
                row["x"],
                row["y"],
                arrow_dx,
                arrow_dy,
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.04,
                color=trajectory_color,
                alpha=0.65,
                zorder=4,
                label="Nozzle orientation" if index == 0 and row_number == 0 else None,
            )

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel("x position [m]")
    ax.set_ylabel("y position [m]")
    ax.grid(True, alpha=0.25)

    if x_values and y_values:
        x_all = np.concatenate(x_values)
        y_all = np.concatenate(y_values)
        x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
        y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
        x_span = max(x_max - x_min, 1e-6)
        y_span = max(y_max - y_min, 1e-6)
        padding = 0.08 * max(x_span, y_span)
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)

    ax.set_aspect("equal", adjustable="box")

    handles, labels = ax.get_legend_handles_labels()
    ordered_handles = []
    ordered_labels = []
    seen_labels: set[str] = set()
    for handle, label in zip(handles, labels):
        if not label or label in seen_labels:
            continue
        seen_labels.add(label)
        ordered_handles.append(handle)
        ordered_labels.append(label)

    legend = None
    if ordered_handles:
        legend = ax.legend(
            ordered_handles,
            ordered_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
        )

    if save_dir is not None and created_fig and fig is not None:
        filename = Path(save_dir) / "trajectory.png"
        fig.tight_layout(rect=(0, 0, 0.8, 1))
        fig.savefig(
            filename,
            bbox_inches="tight",
            pad_inches=0.06,
            bbox_extra_artists=(legend,) if legend is not None else None,
        )

    if created_fig and fig is not None:
        fig.tight_layout(rect=(0, 0, 0.8, 1))
    plt.close(fig)
    return fig


def plot_flight_history(
        flight_data: pd.DataFrame,
        save_dir: str,
        video_data: dict[int, np.ndarray] | None = None,
        video_timestamps_s: list[float] | tuple[float, ...] | None = None,
        fps: float | int | None = None,
        metrics_to_plot=(
            "Main thrust",
            "Nozzle angle",
            "Side thrust",
            "Normalized y",
            "Normalized theta",
            "Relative impulse",
            "Max g",
        ),
        plot_raw: bool = False,
) -> Figure | None:
    required_columns = [
        "time_s",
        "main_engine_thrust",
        "main_engine_thrust_raw",
        "nozzle_angle",
        "nozzle_angle_raw",
        "side_engine_thrust",
        "side_engine_thrust_raw",
        "max_g",
        "y",
        "theta",
    ]
    missing_columns = [column for column in required_columns if column not in flight_data.columns]
    if missing_columns:
        raise ValueError(f"flight_data is missing required columns: {missing_columns}")

    plot_data = flight_data.dropna(subset=["time_s"]).copy().sort_values("time_s").reset_index(drop=True)
    if plot_data.empty:
        raise ValueError("flight_data does not contain any valid time samples")

    def _safe_min_max(series: pd.Series, fallback: tuple[float, float] = (0.0, 1.0)) -> tuple[float, float]:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            return fallback
        lower = float(numeric.min())
        upper = float(numeric.max())
        if np.isclose(lower, upper):
            padding = 1.0 if lower == 0 else abs(lower) * 0.1
            return lower - padding, upper + padding
        padding = 0.05 * (upper - lower)
        return lower - padding, upper + padding

    def _normalize(series: pd.Series, lower: float, upper: float) -> pd.Series:
        if np.isclose(upper, lower):
            return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
        return (series - lower) / (upper - lower)

    time_values = plot_data["time_s"].to_numpy(dtype=float)
    t_min = float(time_values.min())
    t_max = float(time_values.max())
    if np.isclose(t_min, t_max):
        t_max = t_min + 1.0

    has_video_frames = video_data is not None and len(video_data) > 0
    if has_video_frames and video_timestamps_s is not None:
        frame_count = len(cast(dict[int, np.ndarray], video_data))
        if frame_count != len(video_timestamps_s):
            raise ValueError(
                "video_data and video_timestamps_s must have the same length if both are provided. "
                f"Got video_data length {frame_count}, video_timestamps_s length {len(video_timestamps_s)}"
            )

    frame_count = len(cast(dict[int, np.ndarray], video_data)) if has_video_frames else 0
    if has_video_frames and video_timestamps_s is not None:
        sample_times_s = list(video_timestamps_s)
    elif has_video_frames:
        if fps is None:
            raise ValueError("fps must be provided when video_data is supplied without video_timestamps_s")
        sample_times_s = [index / float(fps) for index in range(frame_count)]
    else:
        sample_times_s = []

    available_metrics = [
        {
            "name": "Max g",
            "column": "max_g",
            "raw_column": None,
            "scale": _safe_min_max(plot_data["max_g"]),
            "color": "tab:gray",
        },
        {
            "name": "Main thrust",
            "column": "main_engine_thrust",
            "raw_column": "main_engine_thrust_raw",
            "scale": (0.0, 1.0),
            "color": "tab:blue",
        },
        {
            "name": "Nozzle angle",
            "column": "nozzle_angle",
            "raw_column": "nozzle_angle_raw",
            "scale": (-1.0, 1.0),
            "color": "tab:orange",
        },
        {
            "name": "Side thrust",
            "column": "side_engine_thrust",
            "raw_column": "side_engine_thrust_raw",
            "scale": (-1.0, 1.0),
            "color": "tab:green",
        },
        {
            "name": "Normalized y",
            "column": "y",
            "raw_column": None,
            "scale": _safe_min_max(plot_data["y"]),
            "color": "tab:red",
        },
        {
            "name": "Normalized theta",
            "column": "theta",
            "raw_column": None,
            "scale": _safe_min_max(plot_data["theta"]),
            "color": "tab:purple",
        },
        {
            "name": "Relative impulse",
            "column": "relative_total_impulse",
            "raw_column": None,
            "scale": (0.0, 1.0),
            "color": "tab:brown",
        },
    ]

    available_metric_names = {metric["name"] for metric in available_metrics}
    selected_metric_names = tuple(
        metric["name"] if isinstance(metric, dict) else metric
        for metric in metrics_to_plot
    )
    metrics = [metric for metric in available_metrics if metric["name"] in selected_metric_names]

    fig = plt.figure(figsize=(12, 8))
    video_rows = 1 if has_video_frames else 0
    columns = max(1, frame_count if has_video_frames else 1)
    grid = GridSpec(
        video_rows + len(metrics),
        columns,
        figure=fig,
        height_ratios=([1.35] if has_video_frames else []) + [0.72] * len(metrics),
        hspace=0.22,
        wspace=0.05,
    )

    # Display Picture Roll
    if has_video_frames:
        frame_items = sorted(cast(dict[int, np.ndarray], video_data).items())
        for pos_index, (index, frame) in enumerate(frame_items):
            frame_axis = fig.add_subplot(grid[0, pos_index])
            frame_axis.imshow(frame)
            frame_axis.set_axis_off()
            if video_timestamps_s is not None:
                frame_time = float(video_timestamps_s[pos_index])
            elif fps is not None:
                frame_time = float(index) / float(fps)
            else:
                frame_time = float(index)
            frame_axis.set_title(f"t = {frame_time:.2f} s", fontsize=10)

    # Display metrics
    for metric_index, metric in enumerate(metrics, start=video_rows):
        axis = fig.add_subplot(grid[metric_index, :])
        column = metric["column"]
        values = plot_data[column] if column == "relative_total_impulse" else pd.to_numeric(plot_data[column], errors="coerce")

        lower, upper = metric["scale"]
        axis.plot(plot_data["time_s"], values, color=metric["color"], linewidth=2.0)
        axis.set_ylabel(metric["name"])
        axis.set_xlim(t_min, t_max)
        axis.set_ylim(lower, upper)
        axis.grid(True, alpha=0.25)

        twin_axis = axis.twinx()
        twin_axis.patch.set_visible(False)
        twin_axis.set_ylim(lower, upper)
        twin_axis.set_yticks([lower, (lower + upper) / 2.0, upper])
        twin_axis.tick_params(axis="y", labelsize=8)
        twin_axis.grid(False)

        for sample_time in sample_times_s:
            axis.axvline(
                sample_time,
                color="red",
                linestyle="-",
                linewidth=1.5,
                alpha=0.9,
                zorder=20,
            )

        if metric["raw_column"] is not None and metric["raw_column"] in plot_data.columns and plot_raw:
            raw_series = pd.to_numeric(plot_data[metric["raw_column"]], errors="coerce")
            if raw_series.notna().any():
                raw_lower, raw_upper = _safe_min_max(raw_series)
                if raw_upper > raw_lower:
                    normalized_raw = _normalize(raw_series, raw_lower, raw_upper)
                    axis.plot(
                        plot_data["time_s"],
                        normalized_raw * (upper - lower) + lower,
                        color=metric["color"],
                        alpha=0.18,
                        linewidth=1.0,
                        linestyle="--",
                    )

        if metric["name"] == "Relative impulse":
            axis.set_xlabel("Time [s]")

    fig_path = Path(save_dir) / "flight_history.png"
    fig.savefig(fig_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return fig


def plot_trajectory_set_realistic(
        trajectories: list[pd.DataFrame],
        title: str | None = None,
        ax: Axes | None = None,
        trajectory_labels: list[str] | None = None,
        trajectory_colors: list[tuple[float, float, float]] | None = None,
        save_dir: str | None = None,
        user_args: dict | UserArgs | None = None,
) -> Figure | None:
    if not trajectories:
        raise ValueError("trajectories must contain at least one DataFrame")

    if trajectory_labels is not None and len(trajectory_labels) != len(trajectories):
        raise ValueError("trajectory_labels must match the number of flight data sets")

    def get_n_distinct_colors(n: int) -> list[tuple[float, float, float]]:
        if n <= 0:
            return []
        colors = plt.cm.get_cmap("tab10", n)
        return [colors(i)[:3] for i in range(n)]

    if trajectory_colors is None:
        trajectory_colors = get_n_distinct_colors(len(trajectories))

    env = gym.make(
        "coco_rocket_lander/RocketLander-v0", 
        render_mode="rgb_array",
        args= {} if user_args is None else user_args
    )

    created_render_env = env is None
    env_for_render = env if env is not None else gym.make(
        "coco_rocket_lander/RocketLander-v0",
        render_mode="rgb_array",
        args={} if user_args is None else user_args,
    )
    env_for_render.reset()
    bg_frame = env_for_render.render()
    if bg_frame is None:
        if created_render_env:
            env_for_render.close()
        raise RuntimeError("Environment rendering returned None; ensure render_mode='rgb_array'")

    bg_frame = np.asarray(bg_frame, dtype=np.uint8)
    cfg = cast(RocketLander, env_for_render.unwrapped).cfg
    scale = cfg.scale
    height = bg_frame.shape[0]

    def world_to_pixel(x: float, y: float) -> tuple[float, float]:
        return x * scale, height - (y * scale)

    created_fig = False
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        created_fig = True

    ax.imshow(bg_frame)
    ax.set_axis_off()

    start_drawn = False
    goal_drawn = False
    end_drawn = False

    for index, flight_data in enumerate(trajectories):
        required_columns = {"x", "y", "theta"}
        missing_columns = required_columns.difference(flight_data.columns)
        if missing_columns:
            if created_render_env:
                env_for_render.close()
            raise ValueError(f"Missing required columns in trajectory {index}: {sorted(missing_columns)}")

        valid_positions = flight_data.dropna(subset=["x", "y"])
        if valid_positions.empty:
            continue

        trajectory_color = trajectory_colors[index]
        trajectory_label = trajectory_labels[index] if trajectory_labels is not None else f"Trajectory {index + 1}"

        xs_px = np.array([world_to_pixel(x, y)[0] for x, y in zip(valid_positions["x"], valid_positions["y"])])
        ys_px = np.array([world_to_pixel(x, y)[1] for x, y in zip(valid_positions["x"], valid_positions["y"])])
        ax.plot(xs_px, ys_px, color=trajectory_color, linewidth=2.5, alpha=0.85, label=trajectory_label, zorder=10)

        start_row = valid_positions.iloc[0]
        start_px, start_py = world_to_pixel(start_row["x"], start_row["y"])
        ax.scatter(
            start_px,
            start_py,
            s=150,
            marker="o",
            color=trajectory_color,
            edgecolors="black",
            linewidths=1.5,
            zorder=15,
            label="Start position" if not start_drawn else None,
        )
        start_drawn = True

        if "goal_x" in flight_data.columns and "goal_y" in flight_data.columns:
            goal_x = float(flight_data["goal_x"].iloc[0])
            goal_y = float(flight_data["goal_y"].iloc[0])
            goal_px, goal_py = world_to_pixel(goal_x, goal_y)
            ax.scatter(
                goal_px,
                goal_py,
                s=200,
                marker="*",
                color="cyan",
                edgecolors="black",
                linewidths=1.5,
                zorder=16,
                label="Goal position" if not goal_drawn else None,
            )
            goal_drawn = True

        end_row = valid_positions.iloc[-1]
        end_px, end_py = world_to_pixel(end_row["x"], end_row["y"])
        ax.scatter(
            end_px,
            end_py,
            s=120,
            marker="X",
            color=trajectory_color,
            edgecolors="black",
            linewidths=1.5,
            zorder=14,
            label="End position" if not end_drawn else None,
        )
        end_drawn = True

    if title is not None:
        ax.set_title(title, fontsize=14, fontweight="bold")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        marker_labels = ("Start position", "Goal position", "End position")
        ordered_handles = []
        ordered_labels = []
        seen_labels = set()

        for target_label in marker_labels:
            for handle, label in zip(handles, labels):
                if label == target_label and label not in seen_labels:
                    ordered_handles.append(handle)
                    ordered_labels.append(label)
                    seen_labels.add(label)
                    break

        for handle, label in zip(handles, labels):
            if label in seen_labels:
                continue
            ordered_handles.append(handle)
            ordered_labels.append(label)

        ax.legend(
            ordered_handles,
            ordered_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=10,
            framealpha=0.95,
        )

    if save_dir is not None and created_fig and fig is not None:
        filename = Path(save_dir) / "trajectory_realistic.png"
        fig.tight_layout()
        fig.savefig(filename, bbox_inches="tight", pad_inches=0.1, dpi=100)

    if created_fig and fig is not None:
        fig.tight_layout()

    if created_render_env:
        env_for_render.close()
    plt.close(fig)
    return fig
    cfg = cast(RocketLander, env.unwrapped).cfg

    scale = cfg.scale


def find_initial_pos_shape_for_malfunction(
    controllers: Controller | list[Controller] | None = None,
    controller_colors: list[str] | None = None,
    perturbation_key: str = "main_thruster_range",
    perturbation_range: tuple[float, float] = (0.5, 1.0),
    p_steps: int = 9,
    x_range: tuple[float, float] = (0.0, 2.5),
    x_steps: int = 9,
    trials: int = 1,
    save_dir: str | None = None,
    show_plot: bool = False,
) -> tuple[Figure, pd.DataFrame]:
    """
    Trace the success boundary instead of evaluating the full grid.

    The search starts at the lower-left corner of the requested range and traces the
    clean boundary with continuation: for each next horizontal offset it reuses the
    previous boundary as a seed, expands only until the success/failure transition is
    bracketed, and then refines it with binary search.

    Parameters
    - controllers: a single controller or a list of controllers to trace.
    - controller_colors: colors used for the filled success region of each controller.
    - perturbation_key: field on `UserArgs` to vary.
    - perturbation_range: perturbation search interval.
    - p_steps: binary-search refinement depth per x position.
    - x_range: horizontal-offset interval in meters, relative to the landing position.
    - x_steps: number of x positions used along the traced boundary.
    - trials: repeated runs per point; first trial is used for the boundary trace.
    - save_dir: optional directory to save the figure.
    - show_plot: if True, calls `plt.show()`.

    Returns
    - (fig, results_df)
    """
    import warnings

    from coco_rocket_lander.algs.pid import PID_controller
    from coco_rocket_lander.env.env_cfg import UserArgs

    if controllers is None:
        controllers = [PID_controller([10, 0, 10], [0.1, 0, 0.01], [5, 0.01, 6])]
    elif isinstance(controllers, Controller):
        controllers = [controllers]
    else:
        controllers = list(controllers)

    if not controllers:
        raise ValueError("controllers must contain at least one controller")

    if controller_colors is None:
        controller_colors = [f"C{index}" for index in range(len(controllers))]
    if len(controller_colors) != len(controllers):
        raise ValueError("controller_colors must match the number of controllers")

    if trials < 1:
        raise ValueError("trials must be at least 1")
    if x_steps < 2:
        raise ValueError("x_steps must be at least 2")
    if p_steps < 1:
        raise ValueError("p_steps must be at least 1")

    x_min, x_max = sorted((float(x_range[0]), float(x_range[1])))
    p_min, p_max = sorted((float(perturbation_range[0]), float(perturbation_range[1])))
    x_values = np.linspace(x_min, x_max, int(x_steps))

    import gymnasium as gym

    env_probe = gym.make("coco_rocket_lander/RocketLander-v0", args={})
    env_probe.reset(seed=0)
    env_probe_unwrapped = cast(RocketLander, env_probe.unwrapped)
    landing_pos = env_probe_unwrapped.get_landing_position()
    cfg = env_probe_unwrapped.cfg
    env_probe.close()

    landing_x = float(landing_pos[0])
    landing_y = float(landing_pos[1])
    default_y_frac = 0.9

    def _build_user_args(x_offset_m: float, perturbation_value: float) -> UserArgs:
        initial_x_abs = landing_x + float(x_offset_m)
        initial_x_frac = float(np.clip(initial_x_abs / float(cfg.width), 0.0, 1.0))
        user_args = UserArgs(initial_position=(initial_x_frac, default_y_frac, 0.0))
        if not hasattr(user_args, perturbation_key):
            raise ValueError(f"UserArgs has no attribute '{perturbation_key}'")
        setattr(user_args, perturbation_key, float(perturbation_value))
        return user_args

    def _simulate_success(controller: Controller, x_offset_m: float, perturbation_value: float) -> bool:
        from coco_rocket_lander.env.rocketlander import RocketLander
        from gymnasium.spaces import Box

        for trial_index in range(int(trials)):
            user_args = _build_user_args(x_offset_m, perturbation_value)
            env = gym.make(
                "coco_rocket_lander/RocketLander-v0",
                args=user_args,
            )
            try:
                obs, _ = env.reset(seed=69 + trial_index)
                unwrapped_env = cast(RocketLander, env.unwrapped)
                action_space = cast(Box, env.action_space)

                while True:
                    action_demand = np.asarray(controller.compute_action(obs.copy(), unwrapped_env), dtype=float)
                    action = np.clip(action_demand, action_space.low, action_space.high)
                    next_obs, _, done, _, _ = env.step(action)
                    if done:
                        final_obs = next_obs
                        break
                    obs = next_obs

                goal_x, goal_y, _ = unwrapped_env.get_landing_position()
                return bool(
                    abs(float(final_obs[0]) - float(goal_x)) <= 1.0
                    and abs(float(final_obs[1]) - float(goal_y)) <= 0.2
                )
            finally:
                env.close()

        return False

    def _binary_search_boundary(
        controller: Controller,
        x_offset_m: float,
        lower_p: float,
        upper_p: float,
        success_is_lower: bool,
    ) -> tuple[float, bool, bool, int]:
        lower_success = _simulate_success(controller, x_offset_m, lower_p)
        upper_success = _simulate_success(controller, x_offset_m, upper_p)
        evaluations = 2

        if lower_success == upper_success:
            if lower_success:
                boundary = upper_p if success_is_lower else lower_p
            else:
                boundary = lower_p if success_is_lower else upper_p
            return boundary, lower_success, upper_success, evaluations

        left = lower_p
        right = upper_p

        for _ in range(int(p_steps)):
            midpoint = 0.5 * (left + right)
            midpoint_success = _simulate_success(controller, x_offset_m, midpoint)
            evaluations += 1
            if success_is_lower:
                if midpoint_success:
                    left = midpoint
                else:
                    right = midpoint
            else:
                if midpoint_success:
                    right = midpoint
                else:
                    left = midpoint

        boundary = left if success_is_lower else right
        return boundary, lower_success, upper_success, evaluations

    def _trace_controller(controller: Controller) -> tuple[list[dict], list[float], bool, float]:
        controller_rows: list[dict] = []
        boundary_values: list[float] = []

        first_x = float(x_values[0])
        low_success = _simulate_success(controller, first_x, p_min)
        high_success = _simulate_success(controller, first_x, p_max)

        if low_success != high_success:
            success_is_lower = bool(low_success)
        else:
            success_is_lower = bool(low_success)

        first_boundary, _, _, first_evaluations = _binary_search_boundary(
            controller=controller,
            x_offset_m=first_x,
            lower_p=p_min,
            upper_p=p_max,
            success_is_lower=success_is_lower,
        )
        boundary_values.append(first_boundary)
        controller_rows.append(
            {
                "controller": controller.__class__.__name__,
                "x_offset_m": first_x,
                "boundary_perturbation": first_boundary,
                "success_is_lower": success_is_lower,
                "evaluations": first_evaluations + 2,
            }
        )

        previous_boundary = first_boundary
        previous_step = max((p_max - p_min) / max(2 * int(p_steps), 4), 1e-4)

        for x_offset_m in map(float, x_values[1:]):
            seed = float(np.clip(previous_boundary, p_min, p_max))
            seed_success = _simulate_success(controller, x_offset_m, seed)
            evaluations = 1
            local_step = previous_step

            if success_is_lower:
                if seed_success:
                    lower_p = seed
                    upper_p = min(p_max, seed + local_step)
                    while upper_p < p_max and _simulate_success(controller, x_offset_m, upper_p):
                        evaluations += 1
                        lower_p = upper_p
                        local_step *= 2.0
                        upper_p = min(p_max, lower_p + local_step)
                    boundary, _, _, boundary_evals = _binary_search_boundary(
                        controller=controller,
                        x_offset_m=x_offset_m,
                        lower_p=lower_p,
                        upper_p=upper_p,
                        success_is_lower=True,
                    )
                else:
                    upper_p = seed
                    lower_p = max(p_min, seed - local_step)
                    while lower_p > p_min and not _simulate_success(controller, x_offset_m, lower_p):
                        evaluations += 1
                        upper_p = lower_p
                        local_step *= 2.0
                        lower_p = max(p_min, upper_p - local_step)
                    boundary, _, _, boundary_evals = _binary_search_boundary(
                        controller=controller,
                        x_offset_m=x_offset_m,
                        lower_p=lower_p,
                        upper_p=upper_p,
                        success_is_lower=True,
                    )
            else:
                if seed_success:
                    upper_p = seed
                    lower_p = max(p_min, seed - local_step)
                    while lower_p > p_min and _simulate_success(controller, x_offset_m, lower_p):
                        evaluations += 1
                        upper_p = lower_p
                        local_step *= 2.0
                        lower_p = max(p_min, upper_p - local_step)
                    boundary, _, _, boundary_evals = _binary_search_boundary(
                        controller=controller,
                        x_offset_m=x_offset_m,
                        lower_p=lower_p,
                        upper_p=upper_p,
                        success_is_lower=False,
                    )
                else:
                    lower_p = seed
                    upper_p = min(p_max, seed + local_step)
                    while upper_p < p_max and not _simulate_success(controller, x_offset_m, upper_p):
                        evaluations += 1
                        lower_p = upper_p
                        local_step *= 2.0
                        upper_p = min(p_max, lower_p + local_step)
                    boundary, _, _, boundary_evals = _binary_search_boundary(
                        controller=controller,
                        x_offset_m=x_offset_m,
                        lower_p=lower_p,
                        upper_p=upper_p,
                        success_is_lower=False,
                    )

            boundary = float(np.clip(boundary, p_min, p_max))
            boundary_values.append(boundary)
            controller_rows.append(
                {
                    "controller": controller.__class__.__name__,
                    "x_offset_m": x_offset_m,
                    "boundary_perturbation": boundary,
                    "success_is_lower": success_is_lower,
                    "evaluations": evaluations + boundary_evals,
                }
            )
            previous_boundary = boundary
            previous_step = local_step

        boundary_span = max(boundary_values) - min(boundary_values) if boundary_values else 0.0
        return controller_rows, boundary_values, success_is_lower, boundary_span

    all_rows: list[dict] = []
    traced_boundaries: list[tuple[list[float], str, bool]] = []

    for controller, color in zip(controllers, controller_colors):
        controller_rows, boundary_values, success_is_lower, _ = _trace_controller(controller)
        for row in controller_rows:
            row["color"] = color
        all_rows.extend(controller_rows)
        traced_boundaries.append((boundary_values, color, success_is_lower))

    results_df = pd.DataFrame.from_records(all_rows)

    fig, ax = plt.subplots(figsize=(8, 6))
    for controller, color, boundary_data in zip(controllers, controller_colors, traced_boundaries):
        boundary_values, fill_color, success_is_lower = boundary_data
        label = controller.__class__.__name__
        ax.plot(x_values, boundary_values, color=fill_color, linewidth=2.0, label=f"{label} boundary")
        if success_is_lower:
            ax.fill_between(x_values, p_min, boundary_values, color=fill_color, alpha=0.16)
        else:
            ax.fill_between(x_values, boundary_values, p_max, color=fill_color, alpha=0.16)

    ax.set_xlabel("Initial horizontal offset [m]")
    ax.set_ylabel(perturbation_key)
    ax.set_title("Initial position vs. perturbation: landing success map")
    ax.grid(True, alpha=0.25)
    ax.legend()

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / "initial_pos_shape.png", bbox_inches="tight")

    if show_plot:
        plt.show()
    plt.close(fig)
    return fig, results_df


def plot_malfunction_space():
    pass
