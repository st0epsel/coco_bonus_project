# set up `show_video` function for Google Colab
import base64
import glob
import io
from IPython.display import HTML
from IPython import display as ipythondisplay

from dataclasses import dataclass
from typing import Optional, Tuple

def show_video(prefix: str):
    """
    Display a video in a Jupyter Notebook

    Arguments
    ---------
    prefix: str

    """
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4list = [name.strip('video/') for name in mp4list]
        valid_videos = [name for name in mp4list if name.startswith(prefix)]
        if len(valid_videos) == 0:
            raise FileNotFoundError(f"Did not find a video starting with '{prefix}'. Found: {mp4list}")
        if len(valid_videos) > 1:
            raise ValueError(f"Found multiple videos starting with '{prefix}', please be more specific! Found: {valid_videos}")
        mp4 = valid_videos[0]  # we should only have one
        video = io.open('video/' + mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                    </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Did not find any files ending with .mp4 in the video folder!")

@dataclass
class UserArgs:
    """User arguments for tweaking the environment"""

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