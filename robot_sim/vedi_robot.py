"""Mostra il robot nella finestra di Isaac Sim (non headless).

Il robot parte dritto, cade in avanti e si appoggia sul piede anteriore; poi si ripete.
Chiudere la finestra per terminare. Uso: env_isaac\\Scripts\\python.exe -u robot_sim\\vedi_robot.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--freno", type=float, default=2.0, help="smorzamento sulle ruote (N*m*s/rad)")
parser.add_argument("--ciclo", type=float, default=6.0, help="secondi prima di ripartire")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from robot_cfg import ROBOT_CFG


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 200, device=args.device))
    sim.set_camera_view([0.45, -0.45, 0.25], [0.0, 0.0, 0.05])
    ground = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8)
    )
    ground.func("/World/ground", ground)
    luce = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.85, 0.85))
    luce.func("/World/Light", luce)

    cfg = ROBOT_CFG.copy()
    cfg.prim_path = "/World/Robot"
    cfg.actuators["ruote"].damping = args.freno
    robot = Articulation(cfg)
    sim.reset()

    dt = sim.get_physics_dt()
    passi_ciclo = int(args.ciclo / dt)
    n = 0
    while simulation_app.is_running():
        if n % passi_ciclo == 0:
            stato = robot.data.default_root_state.clone()
            robot.write_root_pose_to_sim(stato[:, :7])
            robot.write_root_velocity_to_sim(stato[:, 7:])
            robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
            robot.reset()
            print("ripartenza: robot dritto", flush=True)
        robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_pos))
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
        n += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
