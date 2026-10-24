"""Lancia l'addestramento del robot con PPO (libreria rsl_rl). E' un sottile involucro dello script di Isaac Lab:
registra prima il nostro ambiente (robot_rl) e poi passa gli argomenti al lancio standard.

Uso (da G:\\robotTest):
  env_isaac6\\Scripts\\python.exe -u robot_sim\\train_robot.py --rl_library rsl_rl --task Robot-Balance-Direct
        --visualizer none --num_envs 2048 --max_iterations 1000

Argomenti utili:
  --visualizer none|kit  none = senza finestra (veloce); kit = con finestra (lento: usare poche decine di robot)
  --num_envs N           numero di robot in parallelo (2048 e' un buon valore per la 5060 Ti)
  --max_iterations N     iterazioni di addestramento (ogni iterazione = 24 passi x num_envs robot)
Risultati: logs/rsl_rl/robot_balance/<data>/model_<iterazione>.pt (rete) e file TensorBoard.
Nota: alla fine il processo puo' non chiudersi da solo; fermarlo a mano dopo "Learning iteration N-1/N".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import warp as wp

wp.config.enable_backward = False

import robot_rl  # noqa: F401,E402  (registra l'ambiente)
from isaaclab_rl.entrypoints import run_train_cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_train_cli(sys.argv[1:]))
