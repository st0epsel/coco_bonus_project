# Computational Control 2026 Rocket Lander Project

A Box2D Gymnasium environment which simulates a Falcon 9 ocean barge landing

Modified by Dylan Vogel, Gerasimos Maltezos, and Benjamin Stadler for the 2023 and 2026 Computation Control course at ETH Zurich  
Original environment created by Reuben Ferrante (https://github.com/arex18/rocket-lander)

# Instructions

## Introduction
You are a control expert, and you are asked to act as a consultant for an aerospace company. This company wants to hear your opinion on their rocket landing control scheme. You are provided with a Jupyter notebook. It contains the following pieces of information
- A simulator of the rocket landing dynamics:
- A linearized model of the rocket derived by their engineers
- Specifications of the control problem (limits on the actuators, constraints on the landing)
- A basic working controller (PID-type) that is already implemented in the simulator.
Their main concern is robustness with respect to malfunctions: the rocket is subject to extreme stress during the mission, and by the time it needs to land, a few things may have happened.

### Failure Modes:
- Its thrusters may operate at lower efficiency
- The nozzle that controls the exhaust direction may not swing as expected
- More or less fuel than expected was used in the flight profile up to the landing.

### Rocket Dynamics Description
Graphic available in \images\rocket_dynamics.png.
- Rocket radius(distance between rocket center line and hull): l_s 
- Distance between center of mass (COM) and horizontal stabilizer nozzles at the tip of the rocket: l_2
- Distance between COM and rocket bottom: l_1
- Distance between rocket bottom and thruster outlet: l_n
- Rocket angle (angle between rocket center line and gravity): theta
- Thruster angle (angle between rocket center line and thrust direction): phi
- Thruster force: F_E
- Left stabilizer nozzle force (perpendicular to rocker center line): F_L
- Right stabilizer nozzle force (perpendicular to rocker center line): F_R
- Rocket mass: m

## Tasks
You need to produce two documents:
- A 5-slide presentation.
- A Jupyter notebook. (\submission_folder\submission.ipynb)

The presentation is intended for the Chief Technology Officer of the company. The presentation needs to be compelling, graphically pleasant, void of typos and mistakes, and professional.

Each slide needs to serve a specific purpose:
1. [ ] Current state: Show how the current controller performs in the standard operating conditions. Briefly comment on the behaviour. [1 point] 
2. [ ] Failure mode: Show that the company’s concerns are correct: under the three conditions mentioned before, their controller can fail the task. In this slide, you want to motivate the need for a better controller. It is your job to illustrate compelling failure scenarios, that is, some that are plausible but that their controller cannot handle well. [3 points] 
3. [ ] Your recommendation: Explain what type of controller you would recommend (among those seen in class). Provide the three most important reasons that support your choice (list three strengths of the controller and explain how they are achieved). [3 points]
4. [ ] Demonstration: Show how the controller that you proposed outperforms their current controller in the failure scenario that you identified in slide 2. Comment on the results: you need to be convincing! [3 points]
5. [ ] Weaknesses: List the three most important aspects of the proposed solution that can be considered challenging, that requires some special resources, or that need to be verified in testing and development. [3 points]

When executed, the notebook produce exactly the main figures that you used in Slide 2 and Slide 4 of the presentation. Follow the instruction in the notebook for reproducibility.

The Jupyter notebook is intended as support material for your presentation. Assume that the engineers in the technical team of the company will read it to understand what you are proposing. It needs to work flawlessly: any glitch will make you look unprofessional! When executed, the notebook produce exactly the main figures that you used in Slide 2 and Slide 4 of the presentation. Follow the instruction in the notebook for reproducibility. The code needs to be interleaved with proper documentation. In particular, it needs to be explained:
- How the failure scenario has been modelled in the simulation, what parameters are available to define it, and how they should be set. Make the comments consistent with what you have presented in slide 2. [2 points]
- What parameters appear in the controller, how they should be set, and how do you recommend modifying them to tune your controller. Make the comments consistent with what you have presented in slide 3. [2 points]

## Grading 
- Presentation: [13 points], as described above.
- Jupyter notebook: [4 points], as described above. 
- Controller test: Your controller will be automatically tested on three test scenario representing the challenges described in the assignment. Each successful test will grant 3 points, for a total of [9 points]. 
- Exam question: At the written exam, you will be asked two short questions on your submission. You will have to show that you have studied the problem in detail, that you can defend your recommendation, and that you can make a quick judgement on a variation of the problem. Each question can grant 2 points, for a total of [4 points]. 
These 30 points count towards the total of 100 points for the course.

# Installation and Setup

## Usage

The simplest way to use the project is to use the provided JupyterHub on [Moodle](https://moodle-app2.let.ethz.ch/course/view.php?id=27609). However, for the ones keep on working offline, an installation guide is provided below.

## Bugs or Package Requests

Is something not working or are you missing your favorite package?

In that case either raise an issue on [GitLab](https://gitlab.ethz.ch/coco_2026/coco_project_2026/) or write an e-mail to [Benjamin](mailto:bestadle@ethz.ch).


## Installation Guide

This project uses **Python 3.13** and **Poetry** for dependency management and requires a few **system-level dependencies**.

This guide walks you through the installation step-by-step for **Linux** using **Ubuntu/Debian**. If you're using a different OS feel free to send us your installation steps such that we can add it for everyone. See above for where to send it to.


### 1. Install System Dependencies

First, we install python 3.13 and further system dependencies via the Debian package manager `apt`.

```bash
sudo apt update
sudo apt install -y python3.13-venv ffmpeg xvfb swig pipx
```

Second, we install `poetry` using `pipx` to keep it isolated from system Python. Check if you have poetry already installed by running
```bash
poetry --version
```

If not, install `poetry` as below:

```bash
pipx install poetry
```


### 2. Clone the Repository

Assuming you have set up an SSH key for your ETHZ gitlab, we simply clone the repository. See [here](https://gitlab.ethz.ch/help/user/ssh.md) for more information about the SSH keys.

```bash
git clone git@gitlab.ethz.ch:coco_2026/coco_project_2026.git
cd coco_project_2026
```

Alteratively, you can also clone via HTTPS, however, then you will need to enter your password for every push.


### 3. Install Python Dependencies

Ensure Poetry uses Python 3.13:

```bash
poetry env use python3.13
```

Install the dependencies into our virtual environment:

```bash
poetry install
```

_Note_ If it fails, try to pass the flag `--no-root`.

### 4. Setup MOSEK

In case you wish to use a state-of-the-art solver you may use [MOSEK](https://www.mosek.com/).

1. Obtain personal academic license from [here](https://www.mosek.com/license/request/personal-academic/)
2. Move license to `~/mosek/` or set the environment variable `MOSEKLM_LICENSE_FILE` to point to your file.

## Running Jupyter Notebooks

This project includes **Jupyter notebooks** that can be run either:

* In your **web browser**
* In **VS Code**

First, we install a new Jupyter Kernel:

```bash
poetry run python -m ipykernel install --user --name coco --display-name "Computational Control"
```

###  Web Browser


Start up Jupyter Lab from the project root directory:

```bash
poetry run jupyter lab
```


This will start a local server and open your browser automatically.

If it does not open automatically, copy the URL shown in the terminal, e.g. `http://localhost:8888`


Inside Jupyter:

1. Open a notebook
2. Click **Kernel → Change Kernel**
3. Select: `Computational Control`


### VS Code

Ensure that you have following extensions installed:

* Python
* Jupyter

The open vscode and select the right python interpreter `.venv/bin/python`.


Open a notebook (`.ipynb` file) and select the same Kernel as you would in the web browser with the top right menu.
