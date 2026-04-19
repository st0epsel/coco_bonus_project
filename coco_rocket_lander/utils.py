# set up `show_video` function for Google Colab
import base64
import glob
import io
from IPython.display import HTML
from IPython import display as ipythondisplay

import gymnasium as gym

from coco_rocket_lander.algs.controller import Controller
from coco_rocket_lander.env.env_cfg import UserArgs

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

def simulate_controller(
        controller: Controller,
        user_args: dict|UserArgs = None,
        video_name: str = None,
) -> Controller:
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
    """
    # create environment
    env = gym.make("coco_rocket_lander/RocketLander-v0", render_mode="rgb_array",
                   args= {} if user_args is None else user_args)
    if video_name is not None:
        env = gym.wrappers.RecordVideo(env, 'video', episode_trigger = lambda x: True,
                                       name_prefix=video_name)

    obs, info = env.reset(seed=0)  # specify a random seed for consistency

    # run simulation loop
    while True:
        # get action from controller
        action = controller.compute_action(
            state=obs,
            env=env.unwrapped
        )

        # apply action
        next_obs, rewards, done, _, info = env.step(action)

        # check if simulation ended
        if done:
            break

        # update observations
        obs = next_obs

    env.close() # video is saved at this step
