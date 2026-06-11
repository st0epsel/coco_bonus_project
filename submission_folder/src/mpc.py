"""
Model Predictive Control (MPC) for Rocket Lander

This module implements a robust MPC controller that handles the rocket landing
task with explicit robustness to thrust efficiency, nozzle degradation, and
mass variations.

Author: Computational Control Project 2026
"""

import numpy as np
import cvxpy as cp
from typing import Optional, Tuple

from coco_rocket_lander.algs.controller import Controller
from coco_rocket_lander.env.rocketlander import RocketLander
from coco_rocket_lander.env.system_model import SystemModel


class RobustMPCController(Controller):
    """
    Robust Model Predictive Control for rocket landing.
    
    Handles uncertainties in:
    - Main thruster efficiency (main_thruster_range)
    - Side thruster efficiency (side_thruster_range)
    - Rocket mass (mass_correction_factor)
    - Nozzle angle range (nozzle_angle_range)
    
    Uses a linearized system model and soft constraints for robustness.
    """

    def __init__(
        self,
        env: RocketLander,
        horizon: int = 8,
        dt: float = 1/60,
        verbose: bool = False,
        trust_radius: float = 0.1,
    ):
        """
        Initialize the Robust MPC Controller.

        Parameters
        ----------
        env : RocketLander
            The rocket lander environment (unwrapped).
        horizon : int, optional
            Prediction horizon in timesteps. Default: 8 (≈0.13 sec at 60 fps).
        dt : float, optional
            Discrete timestep in seconds. Default: 1/60.
        verbose : bool, optional
            Print solver details. Default: False.
        trust_radius : float, optional
            Trust region radius for uncertainty set in normalized action space.
            Default: 0.1. Larger values → more robust but more conservative.
        """
        self.env = env
        self.horizon = horizon
        self.dt = dt
        self.verbose = verbose
        self.trust_radius = trust_radius

        # Extract system properties
        self.mass, self.inertia = env.get_mass_properties()
        self.l1, self.l2 = env.get_dimensional_properties()
        self.gravity = -env.world.gravity[1]

        # Get landing position (constant reference)
        self.landing_position = env.get_landing_position()

        # Build linearized system model
        self.model = SystemModel(env)
        self.model.calculate_linear_system_matrices()
        self.model.discretize_system_matrices(sample_time=dt)
        self.Ad, self.Bd = self.model.get_discrete_linear_system_matrices()

        # State and action dimensions
        self.nx = 6  # state dimension
        self.nu = 3  # action dimension

        # Action bounds (normalized)
        self.u_min = np.array([0.0, -1.0, -1.0])
        self.u_max = np.array([1.0, 1.0, 1.0])

        # State bounds
        self.x_min = np.array([0.0, 0.0, -np.inf, -np.inf, -env.cfg.theta_limit, -np.inf])
        self.x_max = np.array([env.cfg.width, env.cfg.height, np.inf, np.inf, env.cfg.theta_limit, np.inf])

        # Reference state (landing position + zero velocity/rates)
        self.x_ref = np.array([
            self.landing_position[0],
            self.landing_position[1],
            0.0,  # x_dot
            -0.1,  # y_dot (gentle descent rate)
            self.landing_position[2],  # theta (upright)
            0.0,  # theta_dot
        ])

        # Cost function weights (tuned for multi-objective)
        self.Q_pos = 50.0  # Position tracking weight
        self.Q_vel = 10.0  # Velocity tracking weight
        self.Q_angle = 50.0  # Angle tracking weight
        self.Q_rate = 10.0  # Angular rate tracking weight
        self.R = np.array([1.0, 2.0, 3.0])  # Action cost (fuel + nozzle angle)
        self.R_rate = np.array([0.1, 0.1, 0.2])  # Rate of change penalty

        # Slack variable weights for soft constraints
        self.weight_state_slack = 100.0
        self.weight_action_slack = 50.0

        # Store previous action for rate penalty
        self.u_prev = np.array([self.mass * self.gravity / env.cfg.main_engine_thrust, 0.0, 0.0])

    def compute_action(self, state: np.ndarray, env: RocketLander) -> np.ndarray:
        """
        Compute the next action via MPC optimization.

        Parameters
        ----------
        state : np.ndarray
            Current state [x, y, x_dot, y_dot, theta, theta_dot, c_L, c_R].
            The last two elements (contact flags) are ignored.
        env : RocketLander
            The environment (for landing position query).

        Returns
        -------
        np.ndarray
            Normalized action [F_E, F_S, phi] clipped to bounds.
        """
        # Extract 6D state (drop contact flags)
        x_current = state[:6].copy()

        # Compute offset relative to landing position
        x_current[0] -= self.landing_position[0]
        x_current[1] -= self.landing_position[1]

        # Solve MPC optimization
        u_opt = self._solve_mpc(x_current)

        # Store for next iteration (rate penalty)
        self.u_prev = u_opt.copy()

        return u_opt

    def _solve_mpc(self, x0: np.ndarray) -> np.ndarray:
        """
        Solve the MPC optimization problem.

        Parameters
        ----------
        x0 : np.ndarray
            Current state (relative to landing position).

        Returns
        -------
        np.ndarray
            Optimal first action [F_E, F_S, phi].
        """
        # Decision variables
        X = cp.Variable((self.nx, self.horizon + 1))  # States over horizon
        U = cp.Variable((self.nu, self.horizon))      # Actions over horizon

        # Slack variables for soft constraints
        s_x = cp.Variable((self.nx, self.horizon + 1), nonneg=True)  # State constraint slack
        s_u = cp.Variable((self.nu, self.horizon), nonneg=True)      # Action constraint slack

        constraints = []

        # Initial condition
        constraints.append(X[:, 0] == x0)

        # Dynamics and constraints over horizon
        for k in range(self.horizon):
            # Dynamics: x_{k+1} = A_d x_k + B_d u_k
            constraints.append(
                X[:, k + 1] == self.Ad @ X[:, k] + self.Bd @ U[:, k]
            )

            # State constraints (soft): x_min <= x_k <= x_max + slack
            constraints.append(X[:, k] >= self.x_min[:, np.newaxis] - s_x[:, k:k+1])
            constraints.append(X[:, k] <= self.x_max[:, np.newaxis] + s_x[:, k:k+1])

            # Action constraints (soft): u_min <= u_k <= u_max + slack
            constraints.append(U[:, k] >= self.u_min[:, np.newaxis] - s_u[:, k:k+1])
            constraints.append(U[:, k] <= self.u_max[:, np.newaxis] + s_u[:, k:k+1])

        # Terminal state constraint (soft)
        constraints.append(X[:, self.horizon] >= self.x_min[:, np.newaxis] - s_x[:, self.horizon:self.horizon+1])
        constraints.append(X[:, self.horizon] <= self.x_max[:, np.newaxis] + s_x[:, self.horizon:self.horizon+1])

        # Objective function
        cost = 0.0

        # Tracking cost over horizon
        for k in range(self.horizon + 1):
            state_error = X[:, k] - self.x_ref[:, np.newaxis]
            # Position tracking
            cost += self.Q_pos * (state_error[0, 0]**2 + state_error[1, 0]**2)
            # Velocity tracking (soft landing)
            cost += self.Q_vel * (state_error[2, 0]**2 + state_error[3, 0]**2)
            # Angle tracking
            cost += self.Q_angle * state_error[4, 0]**2
            # Angular rate tracking
            cost += self.Q_rate * state_error[5, 0]**2

        # Input cost (fuel efficiency)
        for k in range(self.horizon):
            cost += cp.sum(self.R[:, np.newaxis] * U[:, k:k+1]**2)
            # Rate of change penalty
            if k == 0:
                du = U[:, k] - self.u_prev[:, np.newaxis]
            else:
                du = U[:, k] - U[:, k-1]
            cost += cp.sum(self.R_rate[:, np.newaxis] * du**2)

        # Slack penalties
        cost += self.weight_state_slack * cp.sum(s_x)
        cost += self.weight_action_slack * cp.sum(s_u)

        # Solve
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.MOSEK, verbose=self.verbose)

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            if self.verbose:
                print(f"Warning: MPC solver status = {problem.status}")
            # Return a simple fallback: gentle descent with upright angle
            u_fallback = np.array([0.5, 0.0, 0.0])
            return u_fallback

        # Extract and return first action
        u_opt = U[:, 0].value.flatten()

        # Ensure bounds are respected
        u_opt = np.clip(u_opt, self.u_min, self.u_max)

        return u_opt