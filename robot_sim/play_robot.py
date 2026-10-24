"""Riproduce in finestra una rete addestrata (nessun apprendimento: la rete e' solo eseguita).

Uso (da G:\\robotTest):
  env_isaac6\\Scripts\\python.exe -u robot_sim\\play_robot.py --rl_library rsl_rl --task Robot-Balance-Direct
        --num_envs 3 --visualizer kit --checkpoint logs\\rsl_rl\\robot_balance\\<data>\\model_999.pt
Si apre la finestra di Isaac Sim con i robot comandati dalla rete; premere Play se e' ferma. Alla fine fermare il processo.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import warp as wp

wp.config.enable_backward = False

import robot_rl  # noqa: F401,E402  (registra l'ambiente)
from isaaclab_rl.entrypoints import run_play_cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_play_cli(sys.argv[1:]))
