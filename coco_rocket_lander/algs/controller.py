"""
Controller Interface
"""
from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym

from coco_rocket_lander.env.rocketlander import RocketLander

class Controller(ABC):
    """
    Controller Interface
    """

    @abstractmethod
    def compute_action(
            self,
            state: np.ndarray,
            env: RocketLander,
    ) -> np.ndarray:
        """
        Compute an action given a state and access to the environment

        Arguments
        ---------
        state: np.ndarray
            Current State of the system
        env: rocketlander
            Environment (Rocket Lander), without wrappings

        Returns
        -------
        np.ndarray
            Computed Action
        """
        pass


