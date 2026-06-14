import io
import os
import base64
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.figure import Figure
from gymnasium.spaces import Box
from IPython.display import HTML
from IPython import display as ipythondisplay

import gymnasium as gym

from coco_rocket_lander.algs.controller import Controller
from coco_rocket_lander.env.rocketlander import RocketLander
from coco_rocket_lander.env.env_cfg import UserArgs

from submission_folder.src.plot import plot_trajectory_sets, plot_flight_history, plot_trajectory_set_realistic


def get_unique_session_folder(SESSION_NAME: str = "MAYBE_THIS_THYME") -> str:
    id = 0
    while os.path.exists(Path(f"{SESSION_NAME}_{id}")):
        id += 1

    session_folder = Path(f"{SESSION_NAME}_{id}")
    os.makedirs(session_folder, exist_ok=True)
    return str(session_folder)

def quick_sim(
        controller: Controller,
        user_args: dict | UserArgs | None = None,
) -> float:
    """
    Run a simulation with the provided environment and conroller
    
    Arguments
    ---------
    controller: Controller
        Controller used in the simulation
    user_args: UserArgs
        Custom UserArgs to change environment, set to None for default values

    Returns
    -------
    float
        The final simulated time in seconds.
    """
    # create environment
    env = gym.make(
        "coco_rocket_lander/RocketLander-v0",
        args={} if user_args is None else user_args,
    )

    obs, info = env.reset(seed=0)  # specify a random seed for consistency
    unwrapped_env = cast(RocketLander, env.unwrapped)
    time_step = float(unwrapped_env.cfg.update_time)
    step_index = 0

    # run simulation loop
    while True:
        # get action from controller
        action = controller.compute_action(
            state=obs,
            env=unwrapped_env
        )

        # apply action
        next_obs, rewards, done, _, info = env.step(action)
        step_index += 1

        # check if simulation ended
        if done:
            break

        # update observations
        obs = next_obs

    env.close()
    return step_index * time_step

def simulate_controller(
        controller: Controller,
        user_args: dict | UserArgs | None = None,
        video_name: str | None = None,
        video_folder: str = 'video',
        kf_times: tuple[float, ...] | None = None,
) -> tuple[pd.DataFrame, dict[int, np.ndarray] | None, int]:
    """
    Run a simulation with the provided environment and conroller
    
    Arguments
    ---------
    controller: Controller
        Controller used in the simulation
    user_args: UserArgs
        Custom UserArgs to change environment, set to None for default values
    video_name: str
        Name of the output video automatically created in ./video; set to None for no video
    kf_times: tuple[float, ...] | None
        Optional tuple of time points in seconds at which to extract video frames for the flight history plotting. If None, no frames are extracted and returned.

    Returns
    -------
    pd.DataFrame
        Flight log with the rocket state history and controller outputs.
    dict[int, np.ndarray] | None
        Optional dictionary of video frames with corresponding timestamp
    """
    video_path = Path(video_folder)
    video_path.mkdir(parents=True, exist_ok=True)

    # create environment
    env = gym.make("coco_rocket_lander/RocketLander-v0", render_mode="rgb_array",
                   args= {} if user_args is None else user_args)
    if video_name is not None:
        env = gym.wrappers.RecordVideo(
            env=env, 
            video_folder = video_folder, 
            episode_trigger = lambda x: True,
            name_prefix=video_name,
            disable_logger=False
        )

    obs, info = env.reset(seed=69)  # specify a random seed for consistency
    unwrapped_env = cast(RocketLander, env.unwrapped)
    action_space = cast(Box, env.action_space)
    landing_position = unwrapped_env.get_landing_position()
    flight_records: list[dict[str, float | int | bool]] = []
    video_frames = {} if kf_times is not None else None
    step_index = 0
    frame_index = 0
    fps = unwrapped_env.cfg.fps

    # run simulation loop
    while True:
        # get action from controller
        action_demand = np.asarray(
            controller.compute_action(
            state=obs.copy(),
            env=unwrapped_env
            ),
            dtype=float,
        )
        action = np.clip(action_demand, action_space.low, action_space.high)

        # apply action
        next_obs, rewards, done, truncated, info = env.step(action)

        # Estimate the local load factor from the applied thrust forces.
        # This measures proper acceleration due to the engines and excludes gravity.
        lander_body = cast(Any, unwrapped_env.lander)
        nozzle_body = cast(Any, unwrapped_env.nozzle)
        left_leg_body = cast(Any, unwrapped_env.legs[0])
        right_leg_body = cast(Any, unwrapped_env.legs[1])
        actual_mass = (
            float(lander_body.mass)
            + float(nozzle_body.mass)
            + float(left_leg_body.mass)
            + float(right_leg_body.mass)
        )
        main_engine_force = float(action[0]) * float(unwrapped_env.cfg.main_engine_thrust)
        side_engine_force = float(action[1]) * float(unwrapped_env.cfg.side_engine_thrust)
        nozzle_world_angle = float(obs[4]) + float(action[2]) * float(unwrapped_env.cfg.max_nozzle_angle)

        main_force_x = -np.sin(nozzle_world_angle) * main_engine_force
        main_force_y = np.cos(nozzle_world_angle) * main_engine_force
        side_force_x = np.cos(float(obs[4])) * side_engine_force
        side_force_y = np.sin(float(obs[4])) * side_engine_force
        accel_x = (main_force_x + side_force_x) / actual_mass
        accel_y = (main_force_y + side_force_y) / actual_mass
        max_g = float(np.sqrt(accel_x**2 + accel_y**2) / 9.81)

        # Check if we need to extract the video frame for the history printing
        if video_frames is not None and kf_times is not None and len(kf_times) > frame_index and step_index / fps > kf_times[frame_index]:
            print(f"Extracting video frame at step = {step_index}, time = {step_index / fps:.2f}s (frame {frame_index + 1} of {len(kf_times)})")
            current_frame = env.render()
            if current_frame is not None:
                video_frames[step_index] = np.asarray(current_frame)
            frame_index += 1

        flight_records.append(
            {
                "step": step_index,
                "time_s": step_index / unwrapped_env.cfg.fps,
                "x": float(obs[0]),
                "y": float(obs[1]),
                "x_dot": float(obs[2]),
                "y_dot": float(obs[3]),
                "theta": float(obs[4]),
                "theta_dot": float(obs[5]),
                "leg_contact_left": bool(obs[6]),
                "leg_contact_right": bool(obs[7]),
                "goal_x": float(landing_position[0]),
                "goal_y": float(landing_position[1]),
                "goal_theta": float(landing_position[2]),
                "main_engine_thrust": float(action[0]),
                "side_engine_thrust": float(action[1]),
                "nozzle_angle": float(action[2]),
                "main_engine_thrust_raw": float(action_demand[0]),
                "side_engine_thrust_raw": float(action_demand[1]),
                "nozzle_angle_raw": float(action_demand[2]),
                "max_g": max_g,
                "reward": float(rewards),
                "terminated": bool(done),
                "truncated": bool(truncated),
                "next_x": float(next_obs[0]),
                "next_y": float(next_obs[1]),
                "next_x_dot": float(next_obs[2]),
                "next_y_dot": float(next_obs[3]),
                "next_theta": float(next_obs[4]),
                "next_theta_dot": float(next_obs[5]),
                "next_leg_contact_left": bool(next_obs[6]),
                "next_leg_contact_right": bool(next_obs[7]),
            }
        )

        # check if simulation ended
        if done:
            flight_records.append(
                {
                    "step": step_index + 1,
                    "time_s": (step_index + 1) / unwrapped_env.cfg.fps,
                    "x": float(next_obs[0]),
                    "y": float(next_obs[1]),
                    "x_dot": float(next_obs[2]),
                    "y_dot": float(next_obs[3]),
                    "theta": float(next_obs[4]),
                    "theta_dot": float(next_obs[5]),
                    "leg_contact_left": bool(next_obs[6]),
                    "leg_contact_right": bool(next_obs[7]),
                    "goal_x": float(landing_position[0]),
                    "goal_y": float(landing_position[1]),
                    "goal_theta": float(landing_position[2]),
                    "main_engine_thrust": np.nan,
                    "side_engine_thrust": np.nan,
                    "nozzle_angle": np.nan,
                    "main_engine_thrust_raw": np.nan,
                    "side_engine_thrust_raw": np.nan,
                    "nozzle_angle_raw": np.nan,
                    "max_g": np.nan,
                    "reward": np.nan,
                    "terminated": bool(done),
                    "truncated": bool(truncated),
                    "next_x": np.nan,
                    "next_y": np.nan,
                    "next_x_dot": np.nan,
                    "next_y_dot": np.nan,
                    "next_theta": np.nan,
                    "next_theta_dot": np.nan,
                    "next_leg_contact_left": np.nan,
                    "next_leg_contact_right": np.nan,
                }
            )
            break

        # update observations
        obs = next_obs
        step_index += 1
    
    #print(f"len(video_frames) = {len(video_frames) if video_frames is not None else 'None'}, expected frames based on kf_times: {len(kf_times) if kf_times is not None else 'None'}")
    if video_frames is not None and kf_times is not None and len(video_frames) < len(kf_times):
        print(f"Extracting video frame at step = {step_index}, time = {step_index / fps:.2f}s (frame {frame_index + 1} of {len(kf_times)})")
        current_frame = env.render()
        if current_frame is not None:
            video_frames[step_index] = np.asarray(current_frame)

    env.close() # video is saved at this step

    # Export flight data to csv
    flight_data = pd.DataFrame.from_records(flight_records)
    flight_data_name = f"{video_name}_flight_data.csv" if video_name is not None else "flight_data.csv"
    flight_data.to_csv(video_path / flight_data_name, index=False)

    # Export user args to csv
    if user_args is not None:
        user_args_dict = user_args if isinstance(user_args, dict) else user_args.__dict__
        user_args_df = pd.DataFrame([user_args_dict])
        user_args_name = f"{video_name}_user_args.csv" if video_name is not None else "user_args.csv"
        user_args_df.to_csv(video_path / user_args_name, index=False)
    
    if video_frames is not None and len(video_frames.keys()) == 0:
        print("Warning: kf_times were provided but no video frames were extracted. This may be because the video file was not found or could not be read, or because the specified kf_times were outside the duration of the flight.")
    
    if video_frames is not None and kf_times is not None and len(video_frames) != len(kf_times):
        print(f"Expected to extract {len(kf_times)} video frames based on kf_times, but extracted {len(video_frames)} frames.")

    return flight_data, video_frames, fps

def get_total_impulse(flight_data: pd.DataFrame) -> float:
    """
    Calculate the impulse generated by the engines combined as a proxy for fuel consumption. 
    Use the sum of (main_engine_thrust + side_engine_thrust) * time_step as a simple proxy for total fuel used,
    where time_step is the time between recorded steps in the flight data.
    Use zero order rectangular integration, and assume that the thrust value is constant between recorded steps.

    Arguments
    ---------
    flight_data: pd.DataFrame
        DataFrame containing the flight log, with a "main_engine_thrust" column.

    Returns
    -------
    float
        Total fuel consumed during the flight, in units of main engine thrust seconds.
    """
    if "main_engine_thrust" not in flight_data.columns:
        raise ValueError("flight_data must contain a 'main_engine_thrust' column")
    if "side_engine_thrust" not in flight_data.columns:
        raise ValueError("flight_data must contain a 'side_engine_thrust' column")

    time_step = float(flight_data["time_s"].iloc[1] - flight_data["time_s"].iloc[0]) if len(flight_data) > 1 else 0.0
    main_engine_impulse = (abs(flight_data["main_engine_thrust"]) * time_step).sum()
    print(f"Main engine impulse: {main_engine_impulse:.2f} (thrust * seconds)")
    side_engine_impulse = (abs(flight_data["side_engine_thrust"]) * time_step).sum()
    print(f"Side engine impulse: {side_engine_impulse:.2f} (thrust * seconds)")
    return main_engine_impulse + side_engine_impulse

def add_total_impulse(flight_data: pd.DataFrame) -> pd.DataFrame:
    """
    Add a "total_impulse" column to the flight data, which is the cumulative sum of the impulse generated by the engines up to each time step.

    Arguments
    ---------
    flight_data: pd.DataFrame
        DataFrame containing the flight log, with a "main_engine_thrust" column.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with an added "total_impulse" column.
    """
    if "main_engine_thrust" not in flight_data.columns:
        raise ValueError("flight_data must contain a 'main_engine_thrust' column")
    if "side_engine_thrust" not in flight_data.columns:
        raise ValueError("flight_data must contain a 'side_engine_thrust' column")

    time_step = float(flight_data["time_s"].iloc[1] - flight_data["time_s"].iloc[0]) if len(flight_data) > 1 else 0.0
    impulse_per_step = ((flight_data["main_engine_thrust"] + flight_data["side_engine_thrust"]).clip(lower=0) * time_step)
    flight_data_with_impulse = flight_data.copy()
    flight_data_with_impulse["total_impulse"] = impulse_per_step.cumsum()
    return flight_data_with_impulse

def add_relative_total_impulse(flight_data: pd.DataFrame) -> pd.DataFrame:
    """Add a "relative_total_impulse" column to the flight data, which is the cumulative sum of the impulse generated by the engines up to each time step, normalized by the total impulse at the end of the flight.
    Arguments
    ---------
    flight_data: pd.DataFrame
        DataFrame containing the flight log, with a "main_engine_thrust" column.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with an added "relative_total_impulse" column.
    """
    if "main_engine_thrust" not in flight_data.columns:
        raise ValueError("flight_data must contain a 'main_engine_thrust' column")
    if "side_engine_thrust" not in flight_data.columns:
        raise ValueError("flight_data must contain a 'side_engine_thrust' column")

    time_step = float(flight_data["time_s"].iloc[1] - flight_data["time_s"].iloc[0]) if len(flight_data) > 1 else 0.0
    impulse_per_step = ((flight_data["main_engine_thrust"] + flight_data["side_engine_thrust"]).clip(lower=0) * time_step)
    flight_data_with_impulse = flight_data.copy()
    flight_data_with_impulse["relative_total_impulse"] = impulse_per_step.cumsum() / impulse_per_step.sum()
    return flight_data_with_impulse

def evaluate_controller(
        controller: Controller,
        user_args: dict | UserArgs | None = None, 
        out_dir: str | None = None, 
        kf_t: tuple[float, ...] | None = None, 
        metrics_to_plot: tuple[str, ...] | None = None,
    ) -> tuple[Figure | None, Figure | None, Figure | None, pd.DataFrame]:
    """
    Run a full evaluation of the provided controller, including simulation, trajectory plotting, and flight history plotting.
    Arguments:
        controller: Controller
            The controller to evaluate.
        user_args: dict | UserArgs | None
            Custom UserArgs to change the environment, set to None for default values.
        out_dir: str | None
            The directory to save the evaluation results, set to None for default.
        kf_t: tuple[float, ...] | None
            The relative time points for the evaluation, set to None for default.
        metrics_to_plot: tuple[str, ...] | None
            The metrics to plot in the flight history, set to None for default.

    Returns:
        pd.DataFrame
            The evaluation results.
    """

    total_sim_time = quick_sim(controller, user_args)
    #print(f"Total simulated time: {total_sim_time:.2f} seconds")

    kf_t_s = tuple(t * total_sim_time for t in kf_t) if kf_t is not None else None
    print(f"Keyframe times (s): {kf_t_s if kf_t_s is not None else 'None'}")
    #print(f"Keyframe times (s): {kf_t_s if kf_t_s is not None else 'None'}")

    # Run the full simulation with video recording and data logging.
    video_name = f"evaluation_{controller.__class__.__name__}"
    flight_data, video_frames, fps = simulate_controller(
        controller = controller, 
        user_args = user_args, 
        video_name = video_name, 
        video_folder = out_dir if out_dir is not None else "video",
        kf_times = kf_t_s, 
    )

    # Plot the trajectory.
    trajectory_figure = plot_trajectory_sets(
        trajectories=[flight_data],
        title=f"Trajectory of {controller.__class__.__name__}",
        save_dir=out_dir,
    )

    real_traj_figure = plot_trajectory_set_realistic(
        trajectories=[flight_data],
        title=f"Realistic Trajectory of {controller.__class__.__name__}",
        save_dir=out_dir,
        user_args=user_args,
    )

    # Add relative total impulse
    flight_data = add_relative_total_impulse(flight_data)

    # Plot the flight history with the pseudo-video.
    flight_history_figure = plot_flight_history(
        flight_data=flight_data,
        save_dir=out_dir if out_dir is not None else "video",
        video_data=video_frames,
        video_timestamps_s=kf_t_s,
        fps=fps,
        metrics_to_plot = metrics_to_plot,
    )
    return flight_history_figure, trajectory_figure, real_traj_figure, flight_data


def show_video(path: str):
    try:
        video = io.open(path, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                    </video>'''.format(encoded.decode('ascii'))))
    except Exception as e:
        print(f"Error occurred while displaying video: {e}")

def m_to_rel_pos(pos: tuple[float, float]) -> tuple[float, float]:
    """Convert a position in meters to a relative position in the range [0, 1]"""
    scale: int = 30  # adjusts pixels to units conversions; should keep at 30
    viewport_width: int = 1000
    viewport_height: int = 800
    rel_pos = (pos[0] / (viewport_width / scale), pos[1] / (viewport_height / scale))
    return rel_pos

def landing_succesfull(flight_data: pd.DataFrame, h_tolerance_m: float = 1.0, v_tolerance_m: float = 0.2) -> bool:
    """Determine if the landing was successful based on the final state of the flight data.
    A successful landing is defined as having the final y position less than or equal to 0.1, and the final x position within 0.1 of the goal x position, and the final theta within 0.1 radians of the goal theta.
    Arguments
        flight_data: pd.DataFrame
            DataFrame containing the flight log, with columns "x", "y", "theta", "goal_x", and "goal_theta".
        h_tolerance_m: float
            horizontal tolerance in meters for the position to consider a landing succesfull
        v_tolerance_m: float
            vertical tolerance in meters for the position to consider a landing succesfull
    """
    final_state = flight_data.iloc[-1]
    goal_x = final_state["goal_x"]
    goal_y = final_state["goal_y"]
    return (abs(final_state["x"] - goal_x) <= h_tolerance_m) and (abs(final_state["y"] - goal_y) <= v_tolerance_m)


if __name__ == "__main__":
    pass