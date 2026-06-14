from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np
import scipy.linalg

from coco_rocket_lander.algs.controller import Controller
from coco_rocket_lander.env.system_model import SystemModel


@dataclass
class MPCConfig:
    """
    State order x, y, x_dot, y_dot, theta, theta_dot 
    Input (normalized) order is F_E, F_S, phi]
    """

    # simulation
    dt: float = 0.1 # MPC prediction sample time [s] (independent of the 60 fps environment)
    horizon: int = 30 # Number of prediction steps N (lookahead = ``dt * horizon`` seconds) 

    # cost weights 
    q_diag: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 50.0, 5.0) # state cost Q
    r_diag: Sequence[float] = (0.5, 0.5, 0.5) # input cost R 
    terminal: str = "LQR" # Terminal cost: 'LQR' solves the DARE for stability, else 'scaled_q' uses a scaled version of the state cost Q
    qf_scale: float = 10.0 # Terminal weight multiplier used when terminal == 'scaled_q'

    # constraints (nominal, normalized actuator limits)
    fe_limits: Tuple[float, float] = (0.0, 1.0)
    fs_limits: Tuple[float, float] = (-1.0, 1.0)
    phi_limits: Tuple[float, float] = (-1.0, 1.0)
    soft_theta: bool = True # Soft-constrain ``|theta| <= theta_margin * theta_limit`` via slack
    theta_margin: float = 0.9 # Fraction of the environment angle limit used as the soft attitude bound
    slack_weight: float = 1.0e4 # Penalty on soft-constraint slack (large => closer to a hard constraint)

    # runtime behaviour
    control_every: int = 1 # Re-solve the QP every ``control_every`` frames (1 = every frame)
    cut_thrust_on_contact: bool = True # Zero all thrust once both legs touch down so the rocket settles to sleep
    solver: str = "OSQP" #cvxpy solver ('OSQP', 'ECOS', etc.)
    warm_start: bool = True


class MPCController(Controller):
    """Receding-horizon MPC controller for the rocket lander.

    Parameters
    ----------
    params:
        Tunable parameters. Defaults to :class:`MPCConfig` if omitted.
    env:
        Optional unwrapped ``RocketLander``. If given, the model and QP are built
        eagerly; otherwise they are built lazily on the first ``compute_action`` call.
    """

    def __init__(self, env, params: Optional[MPCConfig] = None):
        if params is not None:
            self.p = params
        else:
            self.p = MPCConfig()

        self.N_STATE = 6
        self.N_ACTION = 3
        self._frame = 0
        self._last_action = np.zeros(self.N_ACTION)

        # Setup variables for the MPC problem, to be initialized in _build()
        self.A: Optional[np.ndarray] = None
        self.B: Optional[np.ndarray] = None
        self.u_eq: Optional[np.ndarray] = None
        self.prob: Optional[cp.Problem] = None
        self.x0: Optional[cp.Parameter] = None
        self.xref: Optional[cp.Parameter] = None
        self.U: Optional[cp.Variable] = None

        env = getattr(env, "unwrapped", env)
        model = SystemModel(env)
        model.calculate_linear_system_matrices()
        model.discretize_system_matrices(sample_time=self.p.dt)
        A, B = model.get_discrete_linear_system_matrices()
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)

        # Compensate for gravity at the equilibrium point (stationary hover)
        gravity_comp = -env.cfg.gravity * env.get_mass_properties()[0] / env.cfg.main_engine_thrust
        self.u_eq = np.array([gravity_comp, 0.0, 0.0])

        self._problem_setup(env)

        
    def _problem_setup(self, env) -> None:
        p = self.p
        n = self.N_STATE
        m = self.N_ACTION
        N = p.horizon

        Q = np.diag(p.q_diag)
        R = np.diag(p.r_diag)
        P = self._final_cost(Q, R)

        self.x0 = cp.Parameter(n)
        self.xref = cp.Parameter(n)
        X = cp.Variable((N + 1, n))
        U = cp.Variable((N, m))

        theta_max = env.cfg.theta_limit * p.theta_margin
        S = cp.Variable((N, 1), nonneg=True) if p.soft_theta else None

        cost = 0
        cons = [X[0] == self.x0]
        for k in range(N):
            du = U[k] - self.u_eq
            cost += cp.quad_form(X[k] - self.xref, Q) + cp.quad_form(du, R)
            cons += [X[k + 1] == self.A @ X[k] + self.B @ du]
            cons += [
                U[k, 0] >= p.fe_limits[0], U[k, 0] <= p.fe_limits[1],
                U[k, 1] >= p.fs_limits[0], U[k, 1] <= p.fs_limits[1],
                U[k, 2] >= p.phi_limits[0], U[k, 2] <= p.phi_limits[1],
            ]
            if p.soft_theta:
                cons += [
                    X[k + 1, 4] <= theta_max + S[k, 0],
                    X[k + 1, 4] >= -theta_max - S[k, 0],
                ]
                cost += p.slack_weight * S[k, 0]

        cost += cp.quad_form(X[N] - self.xref, P)

        self.U = U
        self.prob = cp.Problem(cp.Minimize(cost), cons)

    # Generate actions at each environment step
    def compute_action(self, state: np.ndarray, env) -> np.ndarray:
        env = getattr(env, "unwrapped", env)

        # detect contact of both legs and cut thrust to let the rocket settle.
        if self.p.cut_thrust_on_contact and state[6] > 0 and state[7] > 0:
            self._last_action = np.zeros(self.N_ACTION)
            return self._last_action.copy()

        if self._frame % self.p.control_every == 0:
            self._solve(state, env)
        self._frame += 1
        return self._last_action.copy()

    def _final_cost(self, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        if self.p.terminal == "LQR":
            try:
                P = scipy.linalg.solve_discrete_are(self.A, self.B, Q, R)
                return 0.5 * (P + P.T)
            except Exception:
                # Fallback
                #print(f"DARE failed to solve, falling back to scaled Q terminal cost.")
                pass  
        return self.p.qf_scale * Q

    # Core logic - Solve the MPC problem
    def _solve(self, state: np.ndarray, env) -> None:
        x_now = np.asarray(state[:self.N_STATE], dtype=float)
        x_des, y_des, theta_des = env.get_landing_position()

        self.x0.value = x_now
        self.xref.value = np.array([x_des, y_des, 0.0, 0.0, theta_des, 0.0])

        try:
           self.prob.solve(solver=self.p.solver, warm_start=self.p.warm_start)
        except Exception:
            self._last_action = self._clip(self.u_eq.copy())
            return

        if self.U.value is None or self.prob.status not in (
            "optimal",
            "optimal_inaccurate",
        ):
            # Infeasible / failed: hover on gravity compensation as a safe fallback.
            self._last_action = self._clip(self.u_eq.copy())
            return

        self._last_action = self._clip(self.U.value[0])

    def _clip(self, u: np.ndarray) -> np.ndarray:
        p = self.p
        return np.array([
            np.clip(u[0], *p.fe_limits),
            np.clip(u[1], *p.fs_limits),
            np.clip(u[2], *p.phi_limits),
        ])
