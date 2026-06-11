from typing import Optional, Tuple

import numpy
import gymnasium as gym
import numpy as np

from coco_rocket_lander.algs import PID_controller
from coco_rocket_lander.env.rocketlander import RocketLander
from coco_rocket_lander.env import SystemModel

from submission_folder.src.utils import get_total_impulse, get_unique_session_folder, simulate_controller, plot_trajectory_sets, plot_flight_history, evaluate_controller
from submission_folder.src.user_args import UserArgs

SESSION_NAME = "extract_measurements"
SESSION_FOLDER = get_unique_session_folder(f"video/{SESSION_NAME}")


# Input User Arguments
# Note: x and y are provided as fraction of the screen width and height.
#       All other values use the units of rad, rad/s, and m/s respectively.
args = None
args = {
    #"initial_state": (0.25, 0.5, 0, 0, 0.3, 0)  # 6-tuple (x, y, x_dot, y_dot, theta, theta_dot),
    "initial_barge_position": (0.49, 0.00),  # 2-tuple (x, theta)
}


# Set up environment and mode to get the controller parameters.
env = gym.make("coco_rocket_lander/RocketLander-v0", render_mode="rgb_array", args=args)
unwrapped_env: RocketLander = env.unwrapped
model = SystemModel(unwrapped_env)


gravity = unwrapped_env.cfg.gravity  # fixed at -9.81
mass, inertia = unwrapped_env.get_mass_properties()
l1, l2 = unwrapped_env.get_dimensional_properties()

gravity_comp_fraction = -gravity * mass / unwrapped_env.cfg.main_engine_thrust # normalized main engine thrust to compensate for gravity

print(f"Gravity: {gravity}")
print(f"Mass, Inertia: {mass, inertia}")
print(f"l1, l2: {l1, l2}")
print(f"Gravity compensation fraction: {gravity_comp_fraction}")


# Controller Setup
engine_pid_params = [10, 0, 10]
engine_vector_pid_params = [0.085, 0.001, 10.55]
side_engine_pid_params = [5, 0, 6]

pid_controller = PID_controller(engine_pid_params, engine_vector_pid_params, side_engine_pid_params)

kf_t = (0.0, 0.25, 0.5, 0.75, 1.0)

metrics_to_plot = (
    "Main thrust",
    "Nozzle angle",
    "Side thrust",
#    "Normalized y",
#    "Normalized theta",
    "Relative impulse",
)

evaluate_controller(
    controller=pid_controller,
    user_args=args,
    out_dir=SESSION_FOLDER,
    kf_t = kf_t,
    metrics_to_plot = metrics_to_plot,
)













