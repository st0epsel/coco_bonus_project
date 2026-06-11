import os
from pathlib import Path
from typing import cast

from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from gymnasium.spaces import Box
from IPython.display import HTML
from IPython import display as ipythondisplay

import gymnasium as gym

from coco_rocket_lander.algs.controller import Controller
from coco_rocket_lander.env.rocketlander import RocketLander
from coco_rocket_lander.env.env_cfg import UserArgs


def plot_trajectory_sets(
        trajectories: list[pd.DataFrame],
        title: str | None = None,
        ax: Axes | None = None,
        arrow_count: int = 20,
        arrow_scale: float = 1.5,
        trajectory_labels: list[str] | None = None,
        save_dir: str | None = None,
) -> Axes:
    """
    Plot one or more rocket flight trajectories on the same axes.

    The helper expects the data produced by simulate_controller(): each DataFrame
    should contain state history columns (x, y, theta, ...), controller outputs
    (main_engine_thrust, nozzle_angle, ...), and goal columns (goal_x, goal_y,
    goal_theta).

    Parameters
    ----------
    trajectories:
        List of flight logs, one per simulation run.
    ax:
        Existing Matplotlib axes to draw on. If None, a new figure is created.
    arrow_count:
        Number of nozzle arrows to draw per trajectory.
    trajectory_labels:
        Optional labels for the trajectory lines. If omitted, generic labels are used.
    title:
        Optional plot title.
    save_dir:
        Optional directory to save the plot.

    Returns
    -------
    plt.Axes
        The axes containing the plot.
    """

    def get_n_distinct_colors(n: int) -> list[tuple[float, float, float]]:
        if n <= 0:
            return []

        colors = plt.cm.get_cmap('tab10', n)
        return [colors(i)[:3] for i in range(n)]

    if not trajectories:
        raise ValueError("trajectories must contain at least one DataFrame")

    if trajectory_labels is not None and len(trajectory_labels) != len(trajectories):
        raise ValueError("trajectory_labels must match the number of flight data sets")
    
    trajectory_colors = get_n_distinct_colors(len(trajectories)) # Distinct colors for each trajectory, with good visibility for the arrows and markers

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


    created_figure = False
    figure = None
    if ax is None:
        figure, ax = plt.subplots(figsize=(9, 7))
        created_figure = True

    # Plot each trajectory with its distinct color
    for index, flight_data in enumerate(trajectories):
        required_columns = {"x", "y", "theta", "main_engine_thrust", "nozzle_angle"}
        missing_columns = required_columns.difference(flight_data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns in flight data set {index}: {sorted(missing_columns)}")

        valid_positions = flight_data.dropna(subset=["x", "y"])
        if valid_positions.empty:
            continue


        # Plot trajectory line
        trajectory_color = trajectory_colors[index]
        trajectory_label = trajectory_labels[index] if trajectory_labels is not None else f"Trajectory {index + 1}"

        ax.plot(
            valid_positions["x"],
            valid_positions["y"],
            color=trajectory_color,
            linewidth=2.0,
            alpha=0.9,
            label=trajectory_label,
        )

        # Start, End and Goal markers
        start_row = valid_positions.iloc[0]
        end_row = valid_positions.iloc[-1]
        goal_x = float(flight_data["goal_x"].iloc[0]) if "goal_x" in flight_data.columns else np.nan
        goal_y = float(flight_data["goal_y"].iloc[0]) if "goal_y" in flight_data.columns else np.nan
        
        start_handle = ax.scatter(
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
        end_handle = ax.scatter(
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
            goal_handle = ax.scatter(
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
        
        # Plot thrust arrows
        control_rows = flight_data.dropna(subset=["main_engine_thrust", "nozzle_angle", "theta"])
        # Sum theta into nozzle angle to get the world frame angle of the thrust vector.
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

            # keep a visible but proportional arrow so the sequence reads like a trajectory
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

    # Fit the view to the actual flight path content instead of the default axes span.
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

    # 1:1 Aspect Ratio
    ax.set_aspect("equal", adjustable="box")

    # Build a clean legend that keeps the trajectory lines and the three key markers.
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

    if save_dir is not None and created_figure:
        filename = Path(save_dir) / "trajectory.png"
        print(f"        Saving trajectory plot to {filename}")
        if figure is not None:
            figure.tight_layout(rect=(0, 0, 0.8, 1))
            figure.savefig(
                filename,
                bbox_inches="tight",
                pad_inches=0.06,
                bbox_extra_artists=(legend,) if legend is not None else None,
            )

    if created_figure and figure is not None:
        figure.tight_layout(rect=(0, 0, 0.8, 1))

    return ax

def plot_flight_history(
        flight_data: pd.DataFrame, 
        save_dir: str, 
        video_data, 
        video_timestamps_s,
        fps,
        metrics_to_plot = (
            "Main thrust",
            "Nozzle angle",
            "Side thrust",
            "Normalized y",
            "Normalized theta",
            "Relative impulse",
        ),
        plot_raw = False,
    ) -> Figure:
    """
    Create horizontal pseudo-video (n frames from the video, evenly spaced in time, starting with the first control input and ending with the video)
    Below the pseudo-video, show:
        -  normalized main thruster output [0, 1]
        - main thruster angle [-1, 1]
        - normalized side thruster output [-1, 1]
        - y position over time [0, 1]
        - theta over time [-pi/2,pi/2]
        - total impulse integral [0, 1] (percentage of total impulse used up to that point in the flight)
    Total Impulse integral at the bottom of the image
    with a vertical line indicating the current time step for each frame in the pseudo-video.

    Args:
        flight_data: pd.DataFrame
            DataFrame containing the flight log, with columns for time, control inputs, and state history.
        save_dir: str
            Directory to save the resulting plot.
        video_data: dict[int, np.ndarray] | None
            Optional dictionary of video frames with corresponding timestamp keys, as extracted from the simulation video. If
        rel_img_t: list[float] | None
            Optional list of relative time points in the flight at which to sample frames for the pseudo-video, as a tuple of floats in [0, 1].
    """
    if video_data is not None and not isinstance(video_data, dict) and video_timestamps_s is not None:
        assert len(video_data) == len(video_timestamps_s), "video_data and video_timestamps_s must have the same length if both are provided. Got video_data length {}, video_timestamps_s length {}".format(len(video_data), len(video_timestamps_s))
    required_columns = [
        "time_s",
        "main_engine_thrust",
        "main_engine_thrust_raw",
        "nozzle_angle",
        "nozzle_angle_raw",
        "side_engine_thrust",
        "side_engine_thrust_raw",
        "y",
        "theta",
    ]
    missing_columns = [column for column in required_columns if column not in flight_data.columns]
    if missing_columns:
        raise ValueError(f"flight_data is missing required columns: {missing_columns}")

    plot_data = flight_data.dropna(subset=["time_s"]).copy()
    plot_data = plot_data.sort_values("time_s").reset_index(drop=True)
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

    sample_times_s = video_timestamps_s if video_data is not None else []

    available_metrics = [
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
    metrics = []
    for metric in available_metrics:
        if metric.get("name")  in metrics_to_plot:
            metrics.append(metric)

    # Create a figure with a grid layout: the top row for the pseudo-video frames, and one row per metric below.
    fig = plt.figure(figsize=(18, 12))
    grid = GridSpec(
        1 + len(metrics),
        len(video_timestamps_s),
        figure=fig,
        height_ratios=[1.35] + [0.72] * len(metrics),
        hspace=0.22,
        wspace=0.05,
    )

    # Top pseudo-video row.
    for pos_index, (index, frame) in enumerate(video_data.items()):
        frame_axis = fig.add_subplot(grid[0, pos_index])
        frame_axis.imshow(frame)
        frame_axis.set_axis_off()
        frame_axis.set_title(f"t = {(index/fps):.2f} s", fontsize=10)

    # Shared time axis data for all metric subplots.
    for metric_index, metric in enumerate(metrics, start=1):
        
        axis = fig.add_subplot(grid[metric_index, :])
        column = metric["column"]
        if column == "relative_total_impulse":
            values = plot_data[column]
        else:
            values = pd.to_numeric(plot_data[column], errors="coerce")

        lower, upper = metric["scale"]
        if column in {"main_engine_thrust", "nozzle_angle", "side_engine_thrust"}:
            series_for_plot = values
        elif column == "relative_total_impulse":
            series_for_plot = values
        else:
            series_for_plot = values

        axis.plot(plot_data["time_s"], series_for_plot, color=metric["color"], linewidth=2.0)
        axis.set_ylabel(metric["name"])
        axis.set_xlim(t_min, t_max)
        axis.set_ylim(lower, upper)
        axis.grid(True, alpha=0.25)

        # Add a right-side vertical axis showing the declared scale explicitly.
        twin_axis = axis.twinx()
        twin_axis.patch.set_visible(False)
        twin_axis.set_ylim(lower, upper)
        twin_axis.set_yticks([lower, (lower + upper) / 2.0, upper])
        twin_axis.tick_params(axis="y", labelsize=8)
        twin_axis.grid(False)

        print(f"     Sample times: {sample_times_s}")
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
                    axis.plot(plot_data["time_s"], normalized_raw * (upper - lower) + lower, color=metric["color"], alpha=0.18, linewidth=1.0, linestyle="--")

        if metric["name"] == "Relative impulse":
            axis.set_xlabel("Time [s]")

    figure_path = Path(save_dir) / "flight_history.png"
    print(f"        Saving flight history plot to {figure_path}")
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    return fig