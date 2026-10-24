"""Registra l'ambiente RL del robot in gymnasium (import leggero: Isaac Lab viene caricato solo alla creazione)."""
import gymnasium as gym

gym.register(
    id="Robot-Balance-Direct",
    entry_point="robot_rl.robot_env:RobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "robot_rl.robot_env:RobotEnvCfg",
        "rsl_rl_cfg_entry_point": "robot_rl.robot_env:RobotPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
