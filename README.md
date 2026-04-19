# Computational Control 2026 Rocket Lander Project

A Box2D Gymnasium environment which simulates a Falcon 9 ocean barge landing

Modified by Dylan Vogel, Gerasimos Maltezos, and Benjamin Stadler for the 2023 and 2026 Computation Control course at ETH Zurich  
Original environment created by Reuben Ferrante (https://github.com/arex18/rocket-lander)

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
