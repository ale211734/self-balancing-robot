"""Descrizione del robot a due ruote per Isaac Lab, a partire da robot.urdf.

IL FLUSSO DEI DATI
    Fusion 360 (CAD) -> robot_config/robot_params.json (masse, baricentro, inerzie) -> gen_urdf.py -> robot.urdf
    -> questo file (come Isaac Lab carica e comanda il robot) -> ambiente RL (robot_rl/robot_env.py).
    Se cambia il CAD: rigenerare il json da Fusion e rilanciare gen_urdf.py.

IL ROBOT
    Un corpo (base_link) e due ruote (ruota_sx, ruota_dx) con giunto continuo attorno a y (asse delle ruote).
    Le due ruote sono i soli giunti comandabili. I piedini e le piastre sono solo geometria di collisione del corpo.
"""
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Percorso del modello fisico generato da gen_urdf.py.
URDF = str(Path(__file__).parent / "robot.urdf")
# Raggio della ruota (m): serve a posizionare il robot con le ruote appoggiate a terra.
RAGGIO_RUOTA = 0.0325

ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=URDF,
        fix_base=False,  # il robot e' libero di muoversi e cadere
        merge_fixed_joints=False,
        force_usd_conversion=True,  # rigenera il file USD a ogni avvio (cosi' le modifiche al URDF si vedono sempre)
        # Guadagni del controllo di posizione/velocita' dei giunti: tutti a zero, perche' le ruote vengono comandate
        # con la COPPIA calcolata dal modello del motore (vedi _apply_action in robot_rl/robot_env.py), non con un PD.
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            target_type="none",
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,  # velocita' massima con cui il simulatore separa corpi che si compenetrano (m/s)
            enable_gyroscopic_forces=True,  # effetti giroscopici delle ruote che girano
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # le parti del robot non si urtano tra loro
            solver_position_iteration_count=8,  # precisione del solutore dei vincoli (piu' alto = piu' preciso e piu' lento)
            solver_velocity_iteration_count=2,
        ),
    ),
    # Posizione iniziale: sopra il pavimento con le ruote appena appoggiate; giunti a zero.
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, RAGGIO_RUOTA + 0.001),
        joint_pos={"giunto_ruota_.*": 0.0},
    ),
    actuators={
        # I due motori delle ruote. Nessun PD (stiffness e damping a 0): il simulatore applica solo la coppia (effort)
        # che gli viene data a ogni passo. La coppia vera (tensione, forza controelettromotrice, attrito) e' calcolata
        # nell'ambiente. "armature" (inerzia riflessa del rotore) e il limite di coppia vengono impostati in robot_env.py.
        "ruote": ImplicitActuatorCfg(
            joint_names_expr=["giunto_ruota_.*"],
            effort_limit_sim=1.0,  # N*m, sostituito da robot_env.py
            velocity_limit_sim=60.0,  # rad/s, oltre la velocita' a vuoto reale del motore (29 rad/s a 6 V)
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
