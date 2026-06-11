"""
Simulation and plotting utilities for MPC controller evaluation.

This module provides helpers for:
- Running simulations with different controllers
- Collecting trajectory data
- Generating comparison plots for presentation
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Tuple, Dict, List, Optional

from coco_rocket_lander.algs.controller import Controller
from coco_rocket_lander.env.env_cfg import UserArgs


def run_simulation(
    controller: Controller,
    user_args: Optional[Dict] = None,
    max_steps: int = 3000,
    seed: int = 69,
    verbose: bool = False,
) -> Dict:
    """
    Run a single simulation and return trajectory data.

    Parameters
    ----------
    controller : Controller
        The controller to use.
    user_args : Optional[Dict]
        Environment configuration (initial state, failure modes, etc.).
    max_steps : int
        Maximum simulation steps before timeout.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Print step-by-step info.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'states': (T, 8) array of observations
        - 'actions': (T, 3) array of actions
        - 'rewards': (T,) array of rewards
        - 'done': bool, whether simulation completed successfully
        - 'success': bool, whether landing was successful (reward > 0)
        - 't': (T,) array of time in seconds
        - 'final_reward': float, final reward from environment
    """
    if user_args is None:
        user_args = {}

    # Create environment
    env = gym.make(
        "coco_rocket_lander/RocketLander-v0",
        render_mode="rgb_array",
        args=user_args,
    )

    obs, info = env.reset(seed=seed)

    states = [obs.copy()]
    actions = []
    rewards = []
    final_reward = None
    done = False
    final_info = None

    t = 0
    max_time = max_steps / 60.0  # Convert to seconds at 60 fps

    while t < max_time:
        # Compute action
        action = controller.compute_action(state=obs, env=env.unwrapped)

        # Ensure normalization
        action = np.clip(action, [-np.inf, -1, -1], [1, 1, 1])

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        states.append(obs.copy())
        actions.append(action.copy())
        rewards.append(reward)

        if terminated or truncated:
            done = True
            final_reward = reward
            final_info = info
            if verbose:
                print(f"  Episode ended at t={t:.2f}s, final_reward={reward}")
            break

        t += 1 / 60.0

    env.close()

    states_array = np.array(states)
    actions_array = np.array(actions)
    rewards_array = np.array(rewards)
    time_array = np.arange(len(states)) / 60.0

    return {
        "states": states_array,
        "actions": actions_array,
        "rewards": rewards_array,
        "done": done,
        "success": done and final_reward > 0 if final_reward is not None else False,
        "t": time_array,
        "final_reward": final_reward,
        "max_time": t,
    }


def plot_trajectory_comparison(
    pid_result: Dict,
    mpc_result: Dict,
    pid_label: str = "PID Baseline",
    mpc_label: str = "Robust MPC",
    title: str = "Trajectory Comparison",
    figsize: Tuple = (16, 10),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot side-by-side comparison of two controller trajectories.

    Parameters
    ----------
    pid_result : Dict
        Simulation result from run_simulation() for PID controller.
    mpc_result : Dict
        Simulation result from run_simulation() for MPC controller.
    pid_label : str
        Label for PID trajectory.
    mpc_label : str
        Label for MPC trajectory.
    title : str
        Overall figure title.
    figsize : Tuple
        Figure size (width, height).

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        Figure and axes array.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Extract data
    pid_states = pid_result["states"]
    pid_actions = pid_result["actions"]
    pid_t = pid_result["t"]
    pid_success = pid_result["success"]

    mpc_states = mpc_result["states"]
    mpc_actions = mpc_result["actions"]
    mpc_t = mpc_result["t"]
    mpc_success = mpc_result["success"]

    # Color scheme
    pid_color = "#FF6B6B" if not pid_success else "#4ECDC4"
    mpc_color = "#FF6B6B" if not mpc_success else "#45B7D1"

    # 1. XY Trajectory
    ax = axes[0, 0]
    ax.plot(pid_states[:, 0], pid_states[:, 1], label=pid_label, color=pid_color, linewidth=2, alpha=0.7)
    ax.plot(mpc_states[:, 0], mpc_states[:, 1], label=mpc_label, color=mpc_color, linewidth=2, alpha=0.7)
    ax.scatter(pid_states[-1, 0], pid_states[-1, 1], color=pid_color, s=100, marker="x", linewidths=3)
    ax.scatter(mpc_states[-1, 0], mpc_states[-1, 1], color=mpc_color, s=100, marker="x", linewidths=3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("XY Trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect("equal")

    # 2. Altitude vs Time
    ax = axes[0, 1]
    ax.plot(pid_t, pid_states[:, 1], label=pid_label, color=pid_color, linewidth=2)
    ax.plot(mpc_t, mpc_states[:, 1], label=mpc_label, color=mpc_color, linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude y (m)")
    ax.set_title("Altitude vs Time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3. Angle vs Time
    ax = axes[0, 2]
    ax.plot(pid_t, np.degrees(pid_states[:, 4]), label=pid_label, color=pid_color, linewidth=2)
    ax.plot(mpc_t, np.degrees(mpc_states[:, 4]), label=mpc_label, color=mpc_color, linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (degrees)")
    ax.set_title("Rocket Angle vs Time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axhline(0, color="black", linestyle="--", alpha=0.3, linewidth=1)

    # 4. Velocity Magnitude
    ax = axes[1, 0]
    pid_vel_mag = np.sqrt(pid_states[:, 2]**2 + pid_states[:, 3]**2)
    mpc_vel_mag = np.sqrt(mpc_states[:, 2]**2 + mpc_states[:, 3]**2)
    ax.plot(pid_t, pid_vel_mag, label=pid_label, color=pid_color, linewidth=2)
    ax.plot(mpc_t, mpc_vel_mag, label=mpc_label, color=mpc_color, linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity Magnitude (m/s)")
    ax.set_title("Velocity Magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 5. Main Thrust
    ax = axes[1, 1]
    ax.plot(pid_t[:-1], pid_actions[:, 0], label=pid_label, color=pid_color, linewidth=2, alpha=0.7)
    ax.plot(mpc_t[:-1], mpc_actions[:, 0], label=mpc_label, color=mpc_color, linewidth=2, alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Main Thrust [0, 1]")
    ax.set_title("Main Engine Thrust")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([-0.1, 1.1])

    # 6. Nozzle Angle
    ax = axes[1, 2]
    ax.plot(pid_t[:-1], np.degrees(pid_actions[:, 2]), label=pid_label, color=pid_color, linewidth=2, alpha=0.7)
    ax.plot(mpc_t[:-1], np.degrees(mpc_actions[:, 2]), label=mpc_label, color=mpc_color, linewidth=2, alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Nozzle Angle (degrees)")
    ax.set_title("Nozzle Angle")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig, axes


def plot_single_trajectory(
    result: Dict,
    controller_name: str = "Controller",
    figsize: Tuple = (14, 8),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot a single trajectory in detail.

    Parameters
    ----------
    result : Dict
        Simulation result from run_simulation().
    controller_name : str
        Name of the controller.
    figsize : Tuple
        Figure size.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        Figure and axes array.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f"{controller_name} Trajectory", fontsize=14, fontweight="bold")

    states = result["states"]
    actions = result["actions"]
    t = result["t"]
    success = result["success"]

    color = "#4ECDC4" if success else "#FF6B6B"

    # 1. XY Trajectory
    ax = axes[0, 0]
    ax.plot(states[:, 0], states[:, 1], color=color, linewidth=2.5, label="Path")
    ax.scatter(states[0, 0], states[0, 1], color="green", s=150, marker="o", label="Start", zorder=5)
    ax.scatter(states[-1, 0], states[-1, 1], color="red", s=150, marker="x", linewidths=3, label="End", zorder=5)
    ax.set_xlabel("x (m)", fontsize=10)
    ax.set_ylabel("y (m)", fontsize=10)
    ax.set_title("XY Trajectory", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect("equal")

    # 2. Altitude
    ax = axes[0, 1]
    ax.plot(t, states[:, 1], color=color, linewidth=2.5)
    ax.fill_between(t, 0, states[:, 1], alpha=0.2, color=color)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Altitude y (m)", fontsize=10)
    ax.set_title("Altitude vs Time", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # 3. Angle
    ax = axes[0, 2]
    ax.plot(t, np.degrees(states[:, 4]), color=color, linewidth=2.5, label="Angle")
    ax.axhline(0, color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax.fill_between(t, 0, np.degrees(states[:, 4]), alpha=0.2, color=color)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Angle (degrees)", fontsize=10)
    ax.set_title("Rocket Angle", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # 4. Velocity Magnitude
    ax = axes[1, 0]
    vel_mag = np.sqrt(states[:, 2]**2 + states[:, 3]**2)
    ax.plot(t, vel_mag, color=color, linewidth=2.5)
    ax.fill_between(t, 0, vel_mag, alpha=0.2, color=color)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Velocity (m/s)", fontsize=10)
    ax.set_title("Velocity Magnitude", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # 5. Main Thrust
    ax = axes[1, 1]
    ax.plot(t[:-1], actions[:, 0], color=color, linewidth=2.5, marker="o", markersize=2, alpha=0.8)
    ax.fill_between(t[:-1], 0, actions[:, 0], alpha=0.2, color=color)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Normalized Thrust [0, 1]", fontsize=10)
    ax.set_title("Main Engine Thrust", fontsize=11, fontweight="bold")
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, alpha=0.3)

    # 6. Nozzle Angle
    ax = axes[1, 2]
    ax.plot(t[:-1], np.degrees(actions[:, 2]), color=color, linewidth=2.5, marker="o", markersize=2, alpha=0.8)
    ax.axhline(0, color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax.fill_between(t[:-1], 0, np.degrees(actions[:, 2]), alpha=0.2, color=color)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Nozzle Angle (degrees)", fontsize=10)
    ax.set_title("Nozzle Angle Control", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def create_failure_scenario(
    thrust_degradation: float = 0.7,
    nozzle_degradation: float = 0.6,
    mass_increase: float = 1.15,
    barge_offset: float = 0.05,
    barge_tilt: float = 0.05,
) -> Dict:
    """
    Create a challenging failure scenario combining multiple malfunctions.

    Parameters
    ----------
    thrust_degradation : float
        Thruster efficiency factor (0.7 = 30% loss). Default: 0.7.
    nozzle_degradation : float
        Nozzle angle range factor (0.6 = 40% range loss). Default: 0.6.
    mass_increase : float
        Mass multiplier (1.15 = 15% heavier). Default: 1.15.
    barge_offset : float
        Barge horizontal offset as fraction of width. Default: 0.05.
    barge_tilt : float
        Barge angular offset in radians. Default: 0.05.

    Returns
    -------
    Dict
        UserArgs dictionary for environment initialization.
    """
    scenario = {
        "main_thruster_range": thrust_degradation,
        "side_thruster_range": thrust_degradation,
        "nozzle_angle_range": nozzle_degradation,
        "mass_correction_factor": mass_increase,
        "initial_barge_position": (0.5 + barge_offset, barge_tilt),
    }
    return scenario


def evaluate_controller_robustness(
    controller: Controller,
    scenarios: List[Dict],
    scenario_names: List[str],
    num_seeds: int = 3,
    verbose: bool = False,
) -> Dict:
    """
    Evaluate controller robustness across multiple scenarios.

    Parameters
    ----------
    controller : Controller
        The controller to evaluate.
    scenarios : List[Dict]
        List of UserArgs dictionaries defining failure scenarios.
    scenario_names : List[str]
        Names of each scenario for reporting.
    num_seeds : int
        Number of random seeds per scenario.
    verbose : bool
        Print progress.

    Returns
    -------
    Dict
        Robustness evaluation results with success rates per scenario.
    """
    results = {}

    for scenario, name in zip(scenarios, scenario_names):
        successes = []
        for seed in range(num_seeds):
            result = run_simulation(
                controller=controller,
                user_args=scenario,
                seed=seed,
                verbose=verbose,
            )
            successes.append(1 if result["success"] else 0)

        success_rate = np.mean(successes)
        results[name] = {
            "success_rate": success_rate,
            "successes": successes,
        }

        if verbose:
            print(f"{name}: {success_rate*100:.1f}% success rate")

    return results
