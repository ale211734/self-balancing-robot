"""Prova fisica del modello: il robot cade da fermo e si deve appoggiare sul piede anteriore.

Confronta l'inclinazione a riposo con quella calcolata dal CAD (8,4 gradi in avanti).
Uso: env_isaac\\Scripts\\python.exe robot_sim\\test_modello.py --headless
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--freno", type=float, default=2.0, help="smorzamento sulle ruote (N*m*s/rad), simula motori spenti con riduttore")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from robot_cfg import ROBOT_CFG


def pitch_deg(q):
    """Inclinazione attorno a y (positiva = naso in giu', cioe' in avanti), q = (w, x, y, z)."""
    x, y, z, w = q  # Isaac Lab v3: ordine (x, y, z, w)
    sinp = 2 * (w * y - z * x)
    return math.degrees(math.asin(max(-1.0, min(1.0, sinp))))


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 200, device=args.device))
    sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8)
    ).func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

    cfg = ROBOT_CFG.copy()
    cfg.prim_path = "/World/Robot"
    cfg.actuators["ruote"].damping = args.freno
    robot = Articulation(cfg)
    sim.reset()

    print("corpi:", robot.body_names, " giunti:", robot.joint_names)
    print(f"massa totale nel simulatore: {float(robot.root_physx_view.get_masses().numpy().sum()):.4f} kg")

    dt = sim.get_physics_dt()
    for i in range(int(4.0 / dt)):
        robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_pos))
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
        if i % int(0.5 / dt) == 0:
            q = robot.data.root_quat_w[0].tolist()
            v = robot.data.root_lin_vel_w[0, 0].item()
            print(f"t={i * dt:4.1f}s  inclinazione={pitch_deg(q):6.2f} gradi  x={robot.data.root_pos_w[0, 0].item():+.3f} m  v={v:+.3f} m/s")

    q = robot.data.root_quat_w[0].tolist()
    print(f"RIPOSO: inclinazione finale {pitch_deg(q):.2f} gradi (atteso dal CAD: +8.4 in avanti)")


if __name__ == "__main__":
    main()
    simulation_app.close()
