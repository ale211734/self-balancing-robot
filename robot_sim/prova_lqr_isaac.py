"""Conferma in Isaac Lab del controllo LQR con il modello del motore.

Senza --solo: 7 coppie x 3 prove in parallelo, stampa una tabella (OK/NO) e termina.
Con --solo 0.3 --visualizer kit: 3 robot vicini (rialzo, spinta+, spinta-), in tempo reale, ripetuti all'infinito.
Prove: 'rialzo' (parte appoggiato sul piede anteriore, motori spenti 1 s, poi LQR), 'spinta+/-'
(parte in equilibrio, a t=2 s impulso di velocita' angolare sul corpo).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ritardo", type=float, default=0.0, help="ritardo aggiuntivo misura->comando (s)")
parser.add_argument("--impulso", type=float, default=0.5, help="impulso di velocita' angolare (rad/s)")
parser.add_argument("--solo", type=float, default=0.0, help="una sola coppia di stallo, robot vicini, tempo reale")
parser.add_argument("--una_volta", action="store_true", help="con --solo: una sola prova veloce, stampa le misure sui piedi e termina")
parser.add_argument("--sin_armature", action="store_true", help="diagnostica: niente inerzia riflessa del rotore")
parser.add_argument("--diag", action="store_true", help="stampa la traiettoria del robot indice 1 (spinta+)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

import design_lqr as dl
from robot_cfg import ROBOT_CFG

TAUS = [0.05, 0.08, 0.12, 0.2, 0.3, 0.45, 0.7]
PROVE = ["rialzo", "spinta+", "spinta-"]
DT, CTRL = 1 / 200, 1 / 100
T_FINE, T_ATTIVO, T_SPINTA = 7.0, 1.0, 2.0


def T(x):
    """ProxyArray/warp -> torch."""
    return x.torch if hasattr(x, "torch") else torch.as_tensor(x)


def main():
    taus = [args.solo] if args.solo else TAUS
    cfgs = [(t, p) for t in taus for p in PROVE]
    n = len(cfgs)
    sim = SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device))
    if args.solo:
        sim.set_camera_view([-2.4, 0.5, 1.1], [0.0, 0.5, 0.05])   # fissa, da dietro e in alto, vede tutto il percorso
    g = sim_utils.GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8),
                                 color=(0.10, 0.10, 0.12))   # pavimento scuro, per vedere i piedini
    g.func("/World/ground", g)
    luce = sim_utils.DomeLightCfg(intensity=2500.0)
    luce.func("/World/Light", luce)
    for i in range(n):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform", translation=(0.0, (0.5 if args.solo else 2.0) * i, 0.0))
    cfg = ROBOT_CFG.copy()
    cfg.prim_path = "/World/envs/env_.*/Robot"
    mot0 = dl.MOTORE
    if not args.sin_armature:
        cfg.actuators["ruote"].armature = mot0["j_rotore"] * mot0["rapporto"] ** 2
    cfg.actuators["ruote"].effort_limit_sim = 5.0
    robot = Articulation(cfg)
    sim.reset()
    dev = sim.device

    K, kv, bv, fr = [], [], [], []
    for tau, _ in cfgs:
        m = dict(mot0, tau_stall=tau)
        K.append(dl.lqr(m, dl.modello(m)))
        kv.append(tau / m["v_nom"]); bv.append(tau / m["w0"]); fr.append(m["attrito"])
    K = torch.tensor(np.array(K), dtype=torch.float32, device=dev)
    kv, bv, fr = (torch.tensor(v, dtype=torch.float32, device=dev) for v in (kv, bv, fr))
    vmax = mot0["v_batt"] * mot0["pwm_max"]
    alpha = dl.alpha
    passi_ctrl = int(round(CTRL / DT))
    lag = int(round(args.ritardo / CTRL))
    attivo_da = torch.tensor([T_ATTIVO if p == "rialzo" else 0.0 for _, p in cfgs], device=dev)
    spinta_segno = torch.tensor([1.0 if p == "spinta+" else (-1.0 if p == "spinta-" else 0.0) for _, p in cfgs], device=dev)

    def stato(r):
        q = T(r.data.root_quat_w)   # (x, y, z, w) in Isaac Lab v3
        psi = torch.asin(torch.clamp(2 * (q[:, 3] * q[:, 1] - q[:, 2] * q[:, 0]), -1, 1))
        return psi + alpha, T(r.data.root_ang_vel_w)[:, 1], T(r.data.root_lin_vel_w)[:, 0], T(r.data.root_pos_w)[:, 0]

    def episodio():
        st = T(robot.data.default_root_state).clone()
        for i, (_, p) in enumerate(cfgs):
            st[i, 1] = (0.5 if args.solo else 2.0) * i   # posizione dell'ambiente i (altrimenti si sovrappongono)
            if p != "rialzo":
                th = -alpha   # equilibrio: corpo inclinato indietro di alpha
                st[i, 3:7] = torch.tensor([0.0, np.sin(th / 2), 0.0, np.cos(th / 2)], device=dev)
                st[i, 2] = 0.0325 + 0.001
        robot.write_root_pose_to_sim(st[:, :7])
        robot.write_root_velocity_to_sim(st[:, 7:])
        robot.write_joint_state_to_sim(T(robot.data.default_joint_pos).clone(), T(robot.data.default_joint_vel).clone())
        robot.reset()
        storia = [torch.zeros(n, device=dev) for _ in range(lag + 1)]
        V = torch.zeros(n, device=dev)
        phi_max = torch.zeros(n, device=dev)
        x0 = stato(robot)[3].clone()
        cont_a = torch.zeros(n, device=dev); cont_p = torch.zeros(n, device=dev)   # passi con piede anteriore/posteriore a terra
        h_min_a = torch.full((n,), 9.0, device=dev); h_min_p = torch.full((n,), 9.0, device=dev)
        psi_min = torch.full((n,), 9.0, device=dev); psi_max = torch.full((n,), -9.0, device=dev)
        n_fin = 0
        xr_acc = 0.0
        for k in range(int((30.0 if args.solo else T_FINE) / DT)):
            if not simulation_app.is_running():
                return None
            t = k * DT
            phi, phid, xd, xp = stato(robot)
            if k % passi_ctrl == 0:
                if args.solo:   # tutti e tre: 1,2 m in avanti e 1,2 m indietro a 0,2 m/s (rampe di 1 s), a ripetizione
                    tau_c = (t - 2.0) % 14.0
                    u_c, sg = (tau_c, 1.0) if tau_c < 7.0 else (tau_c - 7.0, -1.0)
                    vr = 0.0 if t < 2.0 else sg * 0.2 * min(1.0, u_c, 7.0 - u_c)
                    xr_acc += vr * CTRL
                    uno = torch.ones_like(xp)
                    ref = torch.stack([uno * xr_acc + x0, uno * vr, torch.zeros_like(xp), torch.zeros_like(xp)], dim=1)
                else:
                    ref = torch.zeros(n, 4, device=dev)
                u = torch.clamp(-(K * (torch.stack([xp, xd, phi, phid], dim=1) - ref)).sum(1), -vmax, vmax)
                storia.append(u); storia.pop(0)
                V = torch.where(attivo_da <= t, storia[0], torch.zeros_like(u))
            wj = T(robot.data.joint_vel)   # velocita' relativa ruota-corpo
            tau = kv[:, None] * V[:, None] - bv[:, None] * wj - fr[:, None] * torch.tanh(wj / 0.5)
            robot.set_joint_effort_target(tau)
            robot.write_data_to_sim()
            if abs(t - T_SPINTA) < DT / 2 and not args.solo:
                lin = T(robot.data.root_lin_vel_w).clone(); ang = T(robot.data.root_ang_vel_w).clone()
                ang[:, 1] += spinta_segno * args.impulso
                robot.write_root_velocity_to_sim(torch.cat([lin, ang], dim=1))
            sim.step()
            robot.update(DT)
            phi_max = torch.maximum(phi_max, phi.abs())
            if args.solo and t >= 2.0:   # misura il contatto dei piedi durante la guida (da 2 s in poi)
                psi = phi - alpha
                z_ax = T(robot.data.root_pos_w)[:, 2]
                # quota del bordo piu' basso di ogni piede sopra il pavimento (corpo ruotato di psi attorno a y)
                h_a = z_ax + (-0.057 * torch.sin(psi) + (-0.0245) * torch.cos(psi))
                h_p = z_ax + (0.057 * torch.sin(psi) + (-0.0145) * torch.cos(psi))
                cont_a += (h_a < 0.002).float(); cont_p += (h_p < 0.002).float(); n_fin += 1
                h_min_a = torch.minimum(h_min_a, h_a); h_min_p = torch.minimum(h_min_p, h_p)
                psi_min = torch.minimum(psi_min, psi); psi_max = torch.maximum(psi_max, psi)
            if args.solo and not args.una_volta:
                time.sleep(DT)   # tempo reale, per guardare
            if args.diag and k % 10 == 0 and t < 3.5:
                print(f"d t={t:4.2f} phi={np.degrees(phi[1].item()):6.2f} phid={phid[1].item():6.2f} xd={xd[1].item():6.3f} "
                      f"V={V[1].item():5.2f} wj={wj[1,0].item():6.1f}", flush=True)
        phi, _, xd, _ = stato(robot)
        if args.solo and n_fin:
            for i, (_, p) in enumerate(cfgs):
                print(f"PIEDI {p:8s}: anteriore a terra {100*cont_a[i].item()/n_fin:5.1f}% (quota min {1000*h_min_a[i].item():5.1f} mm)  "
                      f"posteriore a terra {100*cont_p[i].item()/n_fin:5.1f}% (quota min {1000*h_min_p[i].item():5.1f} mm)  "
                      f"inclinazione corpo {np.degrees(psi_min[i].item()):+5.1f}..{np.degrees(psi_max[i].item()):+5.1f} gradi", flush=True)
        return (phi.abs() < np.radians(3)) & (xd.abs() < 0.15), phi_max

    if args.solo and args.una_volta:
        episodio()
        print("FINE", flush=True)
        return
    if args.solo:
        print("Finestra aperta: premi Play se la simulazione e' ferma. La prova si ripete da sola.", flush=True)
        while simulation_app.is_running():
            episodio()
        return
    ok, phi_max = episodio()
    print(f"ritardo={args.ritardo}s impulso={args.impulso}")
    print("tau_stall | rialzo spinta+ spinta- | phi_max(gradi) per prova")
    for i in range(0, n, 3):
        r = [("OK" if ok[i + j] else "NO") for j in range(3)]
        pm = [f"{np.degrees(phi_max[i + j].item()):5.1f}" for j in range(3)]
        print(f"{cfgs[i][0]:9.2f} | {r[0]:>6} {r[1]:>7} {r[2]:>7} | {' '.join(pm)}", flush=True)
    print("FINE", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
