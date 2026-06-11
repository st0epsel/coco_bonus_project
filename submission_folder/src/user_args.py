from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class UserArgs:
    """
    User arguments for tweaking the environment
    
    Note: x and y are provided as fraction of the screen width and height.
          All other values use the units of rad, rad/s, and m/s respectively.
    """

    initial_position: Optional[Tuple[float, ...]] = None  # 3-tuple (x, y, theta)
    initial_state: Optional[
        Tuple[float, ...]
    ] = None  # 6-tuple (x, y, x_dot, y_dot, theta, theta_dot)

    initial_barge_position: Optional[Tuple[float, ...]] = None  # 2-tuple (x, theta)

    # render crosses at the rocket center of mass and landing position
    render_landing_position: bool = True
    render_lander_center_position: bool = True

    # disturbances, which should generally be left disabled
    enable_wind: bool = False
    enable_moving_barge: bool = False

    random_initial_position: bool = False

    # rocket malfunctions
    main_thruster_range: float = 1.0
    """Range of force output of the main thruster in relation to nominal range:
       0.0 means no thrust output, 1.0 nominal thrust output"""
    side_thruster_range: float = 1.0
    """Range of force output of the side thrusters in relation to nominal range:
       0.0 means no thrust output, 1.0 nominal thrust output"""
    mass_correction_factor: float = 1.0
    """Factor how mass of lander is different: m_actual = m_nominal*mass_corretion_factor"
       1.0 means nominal mass, 1.1 is heavier than nominal"""
    nozzle_angle_range: float = 1.0
    """Range of the nozzle angle of the main trhuster in relation to nominal range:
       0.0 means no movement, 1.0 means nominal range"""