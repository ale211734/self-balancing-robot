"""Misura una rete addestrata, senza finestra.

Esegue la rete a mano (senza gli strumenti di Isaac Lab) su 96 robot: 32 con comando fermo, 32 avanti a 0,2 m/s, 32 indietro.
Dopo i primi 2 s (il rialzo iniziale) misura per ogni gruppo: % di tempo con il piede anteriore / posteriore a terra,
inclinazione media del corpo, velocita' media e numero di cadute. Serve per sapere se la rete segue davvero il comando.

La rete e' un MLP 7 -> 64 -> 64 -> 2 con ingressi normalizzati: gli stessi pesi che andranno esportati sull'ESP32.
Uso: env_isaac6\\Scripts\\python.exe -u robot_sim\\valuta_rete.py --visualizer none --checkpoint <model.pt> [--passi 2000]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=r"G:\robotTest\logs\rsl_rl\robot_balance\2026-10-23_18-28-38\model_299.pt")
parser.add_argument("--passi", type=int, default=2000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import numpy as np
import torch

from robot_rl.robot_env import RobotEnv, RobotEnvCfg


def carica_rete(path, dev):
    d = torch.load(path, map_location=dev, weights_only=False)["actor_state_dict"]
    mean, std = d["obs_normalizer._mean"], d["obs_normalizer._std"]
    w = [(d[f"mlp.{i}.weight"], d[f"mlp.{i}.bias"]) for i in (0, 2, 4)]

    def policy(obs):
        x = (obs - mean) / (std + 1e-8)
        x = torch.nn.functional.elu(x @ w[0][0].T + w[0][1])
        x = torch.nn.functional.elu(x @ w[1][0].T + w[1][1])
        return x @ w[2][0].T + w[2][1]

    return policy


def main():
    cfg = RobotEnvCfg()
    cfg.scene.num_envs = 96
    env = RobotEnv(cfg)
    dev = env.device
    policy = carica_rete(args.checkpoint, dev)
    n = env.num_envs
    gruppi = {"fermo": slice(0, 32), "avanti 0.2": slice(32, 64), "indietro 0.2": slice(64, 96)}
    vcmd = torch.zeros(n, device=dev)
    vcmd[32:64] = 0.2
    vcmd[64:96] = -0.2
    obs, _ = env.reset()
    env.v_cmd[:] = vcmd
    acc = {k: dict(ant=0.0, post=0.0, n=0, psi=[], xd=[], cadute=0, resets=0) for k in gruppi}
    for t in range(args.passi):
        env.v_cmd[:] = vcmd
        obs = env._get_observations()
        with torch.no_grad():
            a = policy(obs["policy"])
        obs, rew, term, trunc, info = env.step(a)
        z = env.robot.data.root_pos_w.torch[:, 2]
        sp, cp = torch.sin(env.psi), torch.cos(env.psi)
        h_ant = z + (-0.057 * sp - 0.0245 * cp)
        h_post = z + (0.057 * sp - 0.0145 * cp)
        if t * 0.01 >= 2.0:    # salta i primi 2 s di ogni run (rialzo iniziale)
            for k, s in gruppi.items():
                g = acc[k]
                g["ant"] += (h_ant[s] < 0.002).float().sum().item()
                g["post"] += (h_post[s] < 0.002).float().sum().item()
                g["n"] += 32
                g["psi"].append(env.psi[s].mean().item())
                g["xd"].append(env.xd[s].mean().item())
        for k, s in gruppi.items():
            acc[k]["cadute"] += term[s].float().sum().item()
    print(f"RISULTATI ({args.passi} passi, 96 robot)")
    print(f"{'comando':14s} | piede ant. a terra | piede post. a terra | inclinazione media | vel. media | cadute")
    for k, g in acc.items():
        print(f"{k:14s} | {100*g['ant']/g['n']:17.1f}% | {100*g['post']/g['n']:18.1f}% | "
              f"{np.degrees(np.mean(g['psi'])):+14.1f} gr | {np.mean(g['xd']):+8.3f}   | {int(g['cadute'])}", flush=True)
    print("FINE", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
