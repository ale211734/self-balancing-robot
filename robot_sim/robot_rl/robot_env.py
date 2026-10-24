"""Ambiente RL (workflow "diretto" di Isaac Lab) del robot a due ruote.

COSA FA
    Mette migliaia di copie del robot in parallelo sulla GPU. Ogni copia deve restare in equilibrio, rialzarsi dal piede
    anteriore e seguire un comando di velocita' in avanti/indietro. L'algoritmo PPO (rsl_rl) impara una rete neurale che,
    a ogni istante di controllo (100 volte al secondo), decide la tensione da dare ai due motori.

COME SI USA
    Non si esegue direttamente: lo carica train_robot.py (addestramento), play_robot.py (riproduzione) e valuta_rete.py
    (misure). Il nome registrato in gymnasium e' "Robot-Balance-Direct" (vedi robot_rl/__init__.py).

DOVE SI CAMBIANO I PARAMETRI
    - Ricompense, durata, comandi, reset ........ classe RobotEnvCfg qui sotto
    - Rete neurale e algoritmo PPO .............. classe RobotPPORunnerCfg in fondo a questo file
    - Motore (coppia, velocita', attrito) ....... dizionario MOTORE in design_lqr.py
    - Masse, baricentro, inerzie ................ robot_config/robot_params.json (da Fusion 360) -> gen_urdf.py -> robot.urdf
    - Giunti e attuatori delle ruote ............ robot_cfg.py

LIMITI DI QUESTA VERSIONE (v1)
    Motore tipico fisso, sensori ideali (nessun rumore sull'IMU), nessun ritardo, nessuna zona morta, nessuna
    randomizzazione di masse/attrito. Serve a verificare che il robot sia fattibile; per il robot vero vanno aggiunti.

CONVENZIONI
    - Assi: x avanti, y a sinistra, z in alto. L'origine del robot e' al centro dell'asse delle ruote.
    - psi  = inclinazione del corpo attorno a y (+ = in avanti, "naso in giu'"), ricavata dal quaternione.
    - phi  = angolo del baricentro dalla verticale sull'asse = psi + alpha. Vale 0 quando il robot e' in equilibrio.
    - alpha = angolo (7,5 gradi) di cui il baricentro e' avanti all'asse: per questo l'equilibrio e' a psi = -7,5 gradi.
    - Quaternioni di Isaac Lab v3 nell'ordine (x, y, z, w).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

import design_lqr as dl  # contiene il dizionario MOTORE e l'angolo alpha del baricentro
from robot_cfg import ROBOT_CFG  # descrizione del robot (URDF, giunti, attuatori)

# Orientamento della luce direzionale (x, y, z, w): solo estetica della finestra.
LIGHT_ORIENTATION = (-0.14644663035869598, -0.3535534143447876, -0.3535534143447876, 0.8535533547401428)
# Quota dell'asse delle ruote da terra = raggio della ruota (65 mm di diametro).
Z_ASSE = 0.0325

# Copia della descrizione del robot, adattata a questo ambiente. {ENV_REGEX_NS} viene sostituito da Isaac Lab con il
# nome di ogni ambiente (env_0, env_1, ...): cosi' lo stesso robot viene clonato in ogni ambiente.
_robot = ROBOT_CFG.copy()
_robot.prim_path = "{ENV_REGEX_NS}/Robot"
# "armature" = inerzia del rotore del motore vista dalla ruota = J_rotore * rapporto^2. E' molto piu' grande dell'inerzia
# della ruota stessa (circa 20 volte) e cambia molto la dinamica: va tenuta.
_robot.actuators["ruote"].armature = dl.MOTORE["j_rotore"] * dl.MOTORE["rapporto"] ** 2
# Limite di coppia imposto dal simulatore (N*m): largo, perche' la coppia vera e' gia' limitata dal modello del motore.
_robot.actuators["ruote"].effort_limit_sim = 5.0


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Scena: pavimento, robot e luce."""

    # Pavimento scuro (cosi' i piedini rossi si vedono) con attrito 0,8 (gomma di camera d'aria su pavimento).
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8),
            color=(0.10, 0.10, 0.12),
        ),
    )
    robot: ArticulationCfg = _robot
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(intensity=2000.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=LIGHT_ORIENTATION),
    )


@configclass
class RobotEnvCfg(DirectRLEnvCfg):
    """Tutti i parametri dell'ambiente: durata, spazi, reset, ricompense."""

    # --- tempi -------------------------------------------------------------------------------------------------
    decimation = 2  # ogni passo di controllo = 2 passi di fisica -> controllo a 100 Hz (fisica a 200 Hz)
    episode_length_s = 10.0  # durata massima di un episodio (1000 passi di controllo)

    # --- dimensioni di azioni e osservazioni -----------------------------------------------------------------
    action_space = 2  # tensione normalizzata [-1, 1] per ciascun motore (sinistro, destro)
    observation_space = 7  # vedi _get_observations
    state_space = 0  # nessuno stato "privilegiato" per il critico

    # --- simulazione -----------------------------------------------------------------------------------------
    # PhysxCfg() forza la fisica PhysX con il motore Kit: serve perche' il robot viene caricato da URDF e il terreno
    # usa asset di Isaac Sim (senza Kit mancherebbe omni.client).
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation, physics=PhysxCfg())
    # Numero di robot in parallelo (puo' essere cambiato con --num_envs). env_spacing = distanza tra gli ambienti (m).
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=2048, env_spacing=1.0, replicate_physics=True, clone_in_fabric=True)

    # --- reset -----------------------------------------------------------------------------------------------
    # Inclinazione iniziale del corpo in gradi (+ = in avanti). L'intervallo include:
    #   +8   = appoggiato sul piede anteriore (rialzo da fermo)
    #   -7,5 = equilibrio
    #   -14  = molto inclinato all'indietro (il piede posteriore tocca a -19,3)
    psi0_range = (-14.0, 8.0)
    v_cmd_max = 0.2  # velocita' comandata massima (m/s): il comando e' scelto a caso in [-0,2; +0,2]
    frac_con_comando = 0.8  # frazione di episodi con comando diverso da zero (gli altri devono stare fermi)
    phi_max_term = 40.0  # gradi: se il baricentro si allontana oltre questo angolo il robot e' "caduto" e l'episodio finisce

    # --- ricompense -------------------------------------------------------------------------------------------
    # Ogni voce e' espressa "per secondo" e viene poi moltiplicata per il passo di controllo (0,01 s).
    # L'algoritmo cerca di massimizzare la somma: il segno + premia, il segno - penalizza.
    rew_alive = 1.0  # premio per ogni istante in cui il robot non e' caduto
    rew_phi = -20.0  # penalita' per rad^2 di inclinazione del baricentro dalla verticale (10 gradi = -0,6 al secondo)
    rew_vel = -2.0  # penalita' per (m/s)^2 di errore sulla velocita' comandata (piccola: serve da guida lontano dal comando)
    rew_vel_exp = 3.0  # premio "a campana": massimo (3 al secondo) quando la velocita' coincide con il comando
    vel_sigma2 = 0.04  # larghezza della campana in (m/s)^2: a 0,2 m/s di errore il premio vale ancora il 37%
    rew_action = -0.02  # penalita' sull'uso di tensione (risparmia batteria, evita azioni inutili)
    rew_dact = -0.05  # penalita' sulla variazione dell'azione tra due passi (evita vibrazioni e commutazioni rapide)
    rew_foot = -3.0  # penalita' per ogni piedino che tocca terra
    rew_term = -100.0  # penalita' una tantum quando il robot cade (e' molto piu' di quello che si guadagna restando in piedi)


class RobotEnv(DirectRLEnv):
    """Ambiente vettorializzato: tutte le funzioni lavorano su tutti i robot insieme (tensori di dimensione num_envs)."""

    cfg: RobotEnvCfg

    def __init__(self, cfg: RobotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        # Indici dei due giunti delle ruote (sinistra e destra).
        ids, _ = self.robot.find_joints("giunto_ruota_.*")
        self._wheel_ids = torch.tensor(ids, device=self.device)

        # Parametri del motore DC (tutti da design_lqr.MOTORE, valori tipici da verificare):
        m = dl.MOTORE
        self.kv = m["tau_stall"] / m["v_nom"]  # N*m per volt: coppia prodotta da 1 V a ruota ferma
        self.bv = m["tau_stall"] / m["w0"]  # N*m*s/rad: freno della forza controelettromotrice (cresce con la velocita')
        self.fr = m["attrito"]  # N*m: attrito secco del riduttore
        self.vmax = m["v_batt"] * m["pwm_max"]  # volt massimi: batteria 7,4 V x PWM limitato all'80%
        self.alpha = float(dl.alpha)  # angolo del baricentro davanti all'asse

        # Buffer per ogni robot (tutti inizializzati a zero).
        n = self.num_envs
        self.actions = torch.zeros(n, 2, device=self.device)  # azione corrente (tensione normalizzata)
        self.prev_actions = torch.zeros(n, 2, device=self.device)  # azione del passo precedente
        self.v_cmd = torch.zeros(n, device=self.device)  # velocita' comandata (m/s)
        self.phi = torch.zeros(n, device=self.device)  # angolo del baricentro dalla verticale (rad)
        self.psi = torch.zeros(n, device=self.device)  # inclinazione del corpo (rad)
        self.psid = torch.zeros(n, device=self.device)  # velocita' di inclinazione (rad/s)
        self.xd = torch.zeros(n, device=self.device)  # velocita' in avanti (m/s)
        self.wj = torch.zeros(n, 2, device=self.device)  # velocita' delle due ruote rispetto al corpo (rad/s)
        self.foot = torch.zeros(n, device=self.device)  # numero di piedini a terra (0, 1 o 2)

    # ---- fisica -----------------------------------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Riceve l'azione della rete (puo' uscire da [-1, 1]) e la limita."""
        self.prev_actions = self.actions
        self.actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        """Converte la tensione nella coppia di ciascun motore e la applica alle ruote (chiamata a ogni passo di fisica).

        Modello DC:  coppia = kv * V  -  bv * velocita'  -  attrito * tanh(velocita'/0,5)
          - kv*V ........ spinta proporzionale alla tensione
          - bv*velocita' . forza controelettromotrice: piu' la ruota gira, meno coppia resta (a velocita' massima = 0)
          - attrito ..... attrito secco del riduttore (liscio con tanh per non avere scatti)
        """
        wj = self.robot.data.joint_vel.torch
        volt = self.actions * self.vmax
        tau = self.kv * volt - self.bv * wj - self.fr * torch.tanh(wj / 0.5)
        self.robot.set_joint_effort_target_index(target=tau, joint_ids=self._wheel_ids)

    # ---- stato ------------------------------------------------------------------------------------------------
    def _update_state(self):
        """Legge dal simulatore le grandezze che servono a osservazioni, ricompense e terminazione."""
        q = self.robot.data.root_quat_w.torch  # quaternione (x, y, z, w)
        # inclinazione attorno a y dal quaternione (positiva = in avanti)
        self.psi = torch.asin(torch.clamp(2 * (q[:, 3] * q[:, 1] - q[:, 2] * q[:, 0]), -1.0, 1.0))
        self.phi = self.psi + self.alpha  # 0 quando il baricentro e' sopra l'asse
        self.psid = self.robot.data.root_ang_vel_w.torch[:, 1]
        self.xd = self.robot.data.root_lin_vel_w.torch[:, 0]
        self.wj = self.robot.data.joint_vel.torch
        # Quota del bordo piu' basso di ciascun piedino sopra il pavimento, ricavata dalla geometria del CAD:
        # anteriore a x = +57 mm, 24,5 mm sotto l'asse; posteriore a x = -57 mm, 14,5 mm sotto l'asse
        # (il posteriore e' stato accorciato per non toccare in equilibrio). Contatto se sotto i 2 mm.
        z = self.robot.data.root_pos_w.torch[:, 2]
        sp, cp = torch.sin(self.psi), torch.cos(self.psi)
        h_ant = z + (-0.057 * sp - 0.0245 * cp)
        h_post = z + (0.057 * sp - 0.0145 * cp)
        self.foot = (h_ant < 0.002).float() + (h_post < 0.002).float()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Dice quali robot hanno finito: caduti (fallimento) oppure scaduto il tempo (successo)."""
        self._update_state()
        time_out = self.episode_length_buf >= self.max_episode_length
        fallito = self.phi.abs() > math.radians(self.cfg.phi_max_term)
        return fallito, time_out

    def _get_observations(self) -> dict:
        """Cosa vede la rete (7 numeri, divisi per una scala tipica per tenerli intorno a 1).

        0: inclinazione del corpo psi (come la misurerebbe l'IMU, senza conoscere alpha)
        1: velocita' di inclinazione (giroscopio)
        2-3: velocita' delle due ruote (encoder)
        4: velocita' comandata (normalizzata: +1 = 0,2 m/s)
        5-6: azione precedente (la rete sa cosa ha comandato un attimo fa)
        """
        obs = torch.cat(
            (
                self.psi.unsqueeze(1) / 0.5,
                self.psid.unsqueeze(1) / 5.0,
                self.wj / 30.0,
                (self.v_cmd / self.cfg.v_cmd_max).unsqueeze(1),
                self.actions,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Punteggio di questo passo per ogni robot (vedi i pesi in RobotEnvCfg)."""
        c = self.cfg
        term = self.reset_terminated.float()  # 1 se il robot e' appena caduto
        r = (
            c.rew_alive * (1.0 - term)  # sopravvivere
            + c.rew_term * term  # caduta
            + c.rew_phi * self.phi**2  # stare dritto
            + c.rew_vel * (self.xd - self.v_cmd) ** 2  # seguire la velocita' (penalita' quadratica, guida debole)
            + c.rew_vel_exp * torch.exp(-((self.xd - self.v_cmd) ** 2) / c.vel_sigma2) * (1.0 - term)  # premio a campana
            + c.rew_action * (self.actions**2).sum(1)  # non usare tensione inutilmente
            + c.rew_dact * ((self.actions - self.prev_actions) ** 2).sum(1)  # azioni dolci
            + c.rew_foot * self.foot  # non appoggiarsi ai piedini
        )
        return r * self.step_dt  # da "per secondo" a "per passo"

    # ---- reset ------------------------------------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Rimette in partenza i robot indicati, con condizioni iniziali casuali."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # Statistica registrata nei log: frazione di episodi arrivati a fine tempo senza cadere.
        survived = self.reset_time_outs[env_ids].float()
        self.extras.setdefault("log", {})["Metrics/success_rate"] = survived.mean()
        super()._reset_idx(env_ids)

        n = len(env_ids)
        dev = self.device
        c = self.cfg
        # Inclinazione iniziale e piccola velocita' angolare casuali.
        psi0 = torch.empty(n, device=dev).uniform_(math.radians(c.psi0_range[0]), math.radians(c.psi0_range[1]))
        psid0 = torch.empty(n, device=dev).uniform_(-0.5, 0.5)
        # Posizione: sopra l'origine del proprio ambiente, con le ruote appoggiate a terra (+1 mm di margine).
        pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += self.scene.env_origins[env_ids]
        pose[:, 2] = self.scene.env_origins[env_ids][:, 2] + Z_ASSE + 0.001
        # Orientamento: rotazione di psi0 attorno a y, quaternione (x, y, z, w) = (0, sin(psi0/2), 0, cos(psi0/2)).
        pose[:, 3] = 0.0
        pose[:, 4] = torch.sin(psi0 / 2)
        pose[:, 5] = 0.0
        pose[:, 6] = torch.cos(psi0 / 2)
        vel = torch.zeros(n, 6, device=dev)  # (vx, vy, vz, wx, wy, wz)
        vel[:, 4] = psid0
        zero = torch.zeros(n, 2, device=dev)
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=zero, env_ids=env_ids)  # ruote ferme
        self.robot.write_joint_velocity_to_sim_index(velocity=zero, env_ids=env_ids)

        # Nuovo comando di velocita': con probabilita' frac_con_comando uniforme in [-v_cmd_max, +v_cmd_max], altrimenti zero.
        con = (torch.rand(n, device=dev) < c.frac_con_comando).float()
        self.v_cmd[env_ids] = con * torch.empty(n, device=dev).uniform_(-c.v_cmd_max, c.v_cmd_max)
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0


@configclass
class RobotPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Parametri dell'algoritmo PPO (libreria rsl_rl) e della rete neurale."""

    num_steps_per_env = 24  # passi raccolti da ogni robot prima di ogni aggiornamento della rete
    max_iterations = 300  # iterazioni di addestramento (sovrascritto da --max_iterations)
    save_interval = 100  # salva un checkpoint ogni tot iterazioni
    experiment_name = "robot_balance"  # cartella dei log: logs/rsl_rl/robot_balance/<data>
    # "Attore" = la rete che sceglie le azioni (e' quella che finira' sull'ESP32): 7 ingressi -> 64 -> 64 -> 2 uscite.
    actor = RslRlMLPModelCfg(
        hidden_dims=[64, 64],
        activation="elu",
        obs_normalization=True,  # normalizza gli ingressi con media/deviazione apprese (da replicare sul robot)
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),  # esplorazione: rumore sulle azioni
    )
    # "Critico" = la rete che stima quanto punteggio futuro ci si aspetta; serve solo in addestramento.
    critic = RslRlMLPModelCfg(hidden_dims=[64, 64], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,  # quanto puo' cambiare la strategia a ogni aggiornamento (PPO)
        entropy_coef=0.005,  # incentivo a continuare a esplorare
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",  # il passo di apprendimento si adatta da solo
        gamma=0.99,  # peso del futuro
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
