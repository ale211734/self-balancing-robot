"""Registra un episodio di simulazione (posa del corpo e angolo ruote) in animazione.json per il visualizzatore.

Per ora: robot dritto che cade in avanti e si appoggia sul piede anteriore. Quando ci sara' la policy, basta
sostituire il blocco di controllo con le azioni della rete.
Uso: env_isaac\\Scripts\\python.exe -u robot_sim\\registra.py --headless
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--freno", type=float, default=2.0)
parser.add_argument("--durata", type=float, default=4.0)
parser.add_argument("--fps", type=int, default=50)
parser.add_argument("--out", type=str, default=str(Path(__file__).parent / "animazione.json"))
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
    ground = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8)
    )
    ground.func("/World/ground", ground)
    cfg = ROBOT_CFG.copy()
    cfg.prim_path = "/World/Robot"
    cfg.actuators["ruote"].damping = args.freno
    robot = Articulation(cfg)
    sim.reset()

    dt = sim.get_physics_dt()
    ogni = int(round(1.0 / args.fps / dt))
    nomi = list(robot.joint_names)
    frames = []
    for i in range(int(args.durata / dt)):
        robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_pos))
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
        if i % ogni == 0:
            frames.append(
                {
                    "p": [round(v, 5) for v in robot.data.root_pos_w[0].tolist()],
                    # Isaac Lab v3 da' (x, y, z, w); il viewer vuole (w, x, y, z)
                    "q": [round(v, 5) for v in (lambda q: [q[3], q[0], q[1], q[2]])(robot.data.root_quat_w[0].tolist())],
                    "g": [round(v, 5) for v in robot.data.joint_pos[0].tolist()],
                }
            )
    Path(args.out).write_text(json.dumps({"fps": args.fps, "giunti": nomi, "frames": frames}))
    print(f"REGISTRATO {len(frames)} frame in {args.out}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
