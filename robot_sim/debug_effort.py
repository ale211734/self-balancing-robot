"""Prova minima: la coppia impostata con set_joint_effort_target muove le ruote?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from robot_cfg import ROBOT_CFG


def T(x):
    return x.torch if hasattr(x, "torch") else torch.as_tensor(x)


sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 200, device=args.device))
g = sim_utils.GroundPlaneCfg()
g.func("/World/ground", g)
for i in range(2):
    sim_utils.create_prim(f"/World/envs/env_{i}", "Xform", translation=(0.0, 2.0 * i, 0.0))
cfg = ROBOT_CFG.copy()
cfg.prim_path = "/World/envs/env_.*/Robot"
robot = Articulation(cfg)
sim.reset()
print("attuatori:", {k: type(v).__name__ for k, v in robot.actuators.items()}, flush=True)
print("effort_limit:", T(robot.data.joint_effort_limits)[0].tolist() if hasattr(robot.data, "joint_effort_limits") else "n/d", flush=True)
import numpy as np
for k in range(500):
    tau_k = 0.0 if k < 200 else 0.15   # 1 s a riposo sul piede, poi +0.15 N*m su entrambe
    robot.set_joint_effort_target(torch.full((2, 2), tau_k, device=sim.device))
    robot.write_data_to_sim()
    sim.step()
    robot.update(1 / 200)
    if k in (190, 199, 205, 210, 220, 240, 270, 300, 350, 400, 499):
        q = T(robot.data.root_quat_w)[0]
        psi = np.degrees(float(torch.asin(torch.clamp(2 * (q[3] * q[1] - q[2] * q[0]), -1, 1))))
        print(f"t={(k+1)/200:.2f} x={T(robot.data.root_pos_w)[0,0].item():+.4f} xd={T(robot.data.root_lin_vel_w)[0,0].item():+.3f} "
              f"pitch={psi:+.1f} wj={[round(v,1) for v in T(robot.data.joint_vel)[0].tolist()]} "
              f"jointnames={robot.joint_names}", flush=True)
print("FINE", flush=True)
simulation_app.close()
