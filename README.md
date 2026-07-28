# Description
This is IS-MPC, a framework for humanoid gait generation.

The main reference is:<br />
[N. Scianca, D. De Simone, L. Lanari, G. Oriolo, "MPC for Humanoid Gait Generation: Stability and Feasibility"](https://ieeexplore.ieee.org/document/8955951)<br />
*Transactions on Robotics*, 2020

The extension available in this repository uses the 3D LIP and can also generate vertical motions. Main reference:<br />
[M. Cipriano, P. Ferrari, N. Scianca, L. Lanari, G. Oriolo, "Humanoid motion generation in a world of stairs"](https://www.sciencedirect.com/science/article/pii/S0921889023001343)<br />
*Robotics and Autonomous Systems*, 2023


# Setup
This project uses [uv](https://docs.astral.sh/uv/). To create the environment, run
```
uv sync
```

To run the simulation
```
uv run python sim_mujoco.py
```
