import abc
import numpy as np

from coco_rocket_lander.algs.controller import Controller

class PID_controller(Controller):
    """
    PID controller: 3 decoupled PID controllers for
    - main engine thrust (Fe)
    - main engine angle (phi)
    - side engine thrust (Fs)
    """

    def __init__(self,
                 Fe_PID_params, phi_PID_params, FsTheta_PID_params
                 ):
        """
        Initialization of PID controller

        Arguments
        ---------
        Fe_PID_params: list
            List of PID parameters for the main engine thrust (Kp, Ki, Kd)
        phi_PID_params: list
            List of PID parameters for the main engine angle (Kp, Ki, Kd)
        FsTheta_PID_params: list
            List of PID parameters for the side engine thrust(Kp, Ki, Kd)
        """
        # Set up underlying PID controllers
        self._pid_benchmark = PID_Benchmark(
            Fe_PID_params, phi_PID_params, FsTheta_PID_params
        )

    def compute_action(self, state, env):
        landing_position = env.get_landing_position()  # (x, y, theta) in [m, m, radians]

        # offset so that (0, 0) is at landing position
        pid_state = state
        pid_state[0] = state[0] - landing_position[0]
        pid_state[1] = state[1] - landing_position[1]

        # get action
        action = np.array(self._pid_benchmark.pid_algorithm(pid_state))
        
        return action

class PID_Framework():
    """ Sets the skeleton code for the actual pid algorithms (children) to inherit. """

    @abc.abstractmethod
    def pid_algorithm(self, s, x_target, y_target):
        pass

class PID_Benchmark(PID_Framework):
    """ Tuned PID Benchmark against which all other algorithms are compared. """

    def __init__(self, Fe_PID_params, psi_PID_params, FsTheta_PID_params):
        super(PID_Benchmark, self).__init__()
        self.Fe_PID = PID(Fe_PID_params[0], Fe_PID_params[1], Fe_PID_params[2])
        self.psi_PID = PID(psi_PID_params[0], psi_PID_params[1], psi_PID_params[2])
        self.Fs_theta_PID = PID(FsTheta_PID_params[0], FsTheta_PID_params[1], FsTheta_PID_params[2])

    def pid_algorithm(self, s, x_target=None, y_target=None):
        dx, dy, vel_x, vel_y, theta, omega, legContact_left, legContact_right = s
        if x_target is not None:
            dx = dx - x_target
        if y_target is not None:
            dy = dy - y_target
        # ------------------------------------------
        y_ref = -0.1  # Adjust speed
        y_error = y_ref - dy + 0.1 * dx
        y_dterror = -vel_y + 0.1 * vel_x

        Fe = self.Fe_PID.compute_output(y_error, y_dterror) * (abs(dx) * 50 + 1)
        # ------------------------------------------
        theta_ref = 0
        theta_error = theta_ref - theta + 0.2 * dx  # theta is negative when slanted to the north east
        theta_dterror = -omega + 0.2 * vel_x
        Fs_theta = self.Fs_theta_PID.compute_output(theta_error, theta_dterror)
        Fs = -Fs_theta  # + Fs_x
        # ------------------------------------------
        theta_ref = 0
        theta_error = -theta_ref + theta
        theta_dterror = omega
        if (abs(dx) > 0.01 and dy < 0.5):
            theta_error = theta_error - 0.06 * dx  # theta is negative when slanted to the right
            theta_dterror = theta_dterror - 0.06 * vel_x
        psi = self.psi_PID.compute_output(theta_error, theta_dterror)

        if legContact_left and legContact_right:  # legs have contact
            Fe = 0
            Fs = 0

        return Fe, Fs, psi

class PID_psi(PID_Framework):
    """ PID for controlling just the angle of the rocket nozzle. """

    def __init__(self):
        super(PID_psi, self).__init__()
        self.psi = PID(0.1, 0, 0.01)

    def pid_algorithm(self, s, x_target=None, y_target=None):
        dx, dy, vel_x, vel_y, theta, omega, legContact_left, legContact_right = s
        theta_ref = 0
        theta_error = -theta_ref + theta
        theta_dterror = omega
        if (abs(dx) > 0.01 and dy < 0.5):
            theta_error = theta_error - 0.06 * dx  # theta is negative when slanted to the right
            theta_dterror = theta_dterror - 0.06 * vel_x
        psi = self.psi.compute_output(theta_error, theta_dterror)
        return psi

class PID_Fs(PID_Framework):
    """ PID for controlling just the Fs/theta angle of the rocket nozzle. """

    def __init__(self):
        super(PID_Fs, self).__init__()
        self.Fs = PID(5, 0.01, 6)

    def pid_algorithm(self, s, x_target=None, y_target=None):
        dx, dy, vel_x, vel_y, theta, omega, legContact_left, legContact_right = s
        theta_ref = 0
        theta_error = theta_ref - theta + 0.2 * dx  # theta is negative when slanted to the north east
        theta_dterror = -omega + 0.2 * vel_x
        Fs = self.Fs.compute_output(theta_error, theta_dterror)
        if legContact_left and legContact_right:  # legs have contact
            Fs = 0
        return -Fs

class PID_Fe(PID_Framework):
    """ PID for controlling just the Fs/theta angle of the rocket nozzle. """

    def __init__(self):
        super(PID_Fe, self).__init__()
        self.Fe = PID(10, 0, 10)

    def pid_algorithm(self, s, x_target=None, y_target=None):
        dx, dy, vel_x, vel_y, theta, omega, legContact_left, legContact_right = s
        y_ref = -0.1  # Adjust speed
        y_error = y_ref - dy + 0.1 * dx
        y_dterror = -vel_y + 0.1 * vel_x
        Fe = self.Fe.compute_output(y_error, y_dterror) * (abs(dx) * 50 + 1)
        if legContact_left and legContact_right:  # legs have contact
            Fe = 0
        return Fe

class PID():
    """ Called from the children of PID_Framework"""
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.accumulated_error = 0

    def increment_intregral_error(self, error, pi_limit=3):
        self.accumulated_error = self.accumulated_error + error
        if (self.accumulated_error > pi_limit):
            self.accumulated_error = pi_limit
        elif (self.accumulated_error < pi_limit):
            self.accumulated_error = -pi_limit

    def compute_output(self, error, dt_error):
        self.increment_intregral_error(error)
        return self.Kp * error + self.Ki * self.accumulated_error + self.Kd * dt_error