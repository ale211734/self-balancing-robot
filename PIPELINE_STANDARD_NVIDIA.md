# Regola del progetto: seguire SOLO la pipeline standard NVIDIA

Decisione del 2026-10-24: ripartire con il modello CAD importato in **FreeCAD** (workbench **CROSS**) e seguire
alla lettera la pipeline standard NVIDIA/Isaac Lab, senza scorciatoie o ponti scritti a mano. "Minimo sindacale":
si fa solo quello che la pipeline richiede, niente di più, niente script/file extra "perché comodi".

## La pipeline standard (questo e solo questo)

```
CAD (FreeCAD, modello + materiali/densita' per le masse)
   -> workbench CROSS: definizione link e giunti, export URDF (masse/inerzie/mesh incluse)
   -> Isaac Sim URDF Importer (sim_utils.UrdfFileCfg) -> USD (corpi rigidi, giunti, collisioni)
   -> isaaclab.assets.ArticulationCfg (robot_cfg.py) -> robot caricato nella scena
   -> isaaclab.envs.DirectRLEnv (robot_env.py) -> ambiente RL
   -> rsl_rl (train_robot.py) -> addestramento PPO -> checkpoint
   -> play_robot.py / valuta_rete.py (API Isaac Lab standard) -> verifica
```

## Cosa NON fare (era il motivo del riavvio)

- Niente bridge intermedi scritti a mano (il vecchio `robot_config/robot_params.json` + `gen_urdf.py`
  esistevano solo perche' Fusion 360 non ha un esportatore URDF nativo). FreeCAD/CROSS esporta l'URDF
  direttamente dal CAD: usare quello, non ricostruire un JSON intermedio.
- Niente script di verifica/esperimento paralleli alla pipeline (vecchi `test_modello.py`, `vedi_robot.py`,
  `prova_lqr_isaac.py`, `sweep_ritardo.py`) a meno che non siano il modo previsto da Isaac Lab per
  verificare un passo della pipeline stessa.
- Niente file di debug/log lasciati nel progetto (txt di output di run passate, `__pycache__`, ecc.).

## Cosa e' comunque legittimo tenere (non e' un ponte, e' un dato riusabile)

- Il dizionario `MOTORE` (coppia, attrito, batteria, rapporto di riduzione) e le funzioni fisiche del
  pendolo invertito in `design_lqr.py`: non sono "pipeline NVIDIA", ma sono dati/equazioni del motore
  che `robot_env.py` legge per calcolare l'armatura del giunto — servono indipendentemente da quale CAD
  ha generato l'URDF. Vanno adattati (nomi giunti/link) al nuovo URDF da FreeCAD, non ricreati da zero.
- `robot_cfg.py`, `robot_env.py`, `train_robot.py`, `play_robot.py`, `valuta_rete.py`: sono il modo
  normale e previsto di usare Isaac Lab (non workaround), vanno solo riadattati al nuovo URDF.

## Prossimi passi
1. Importare `export\robot_2ruote_esp32_varD.step` in FreeCAD.
2. Installare/usare il workbench CROSS per definire i due giunti (corpo-ruota sx/dx, asse y) e assegnare
   materiali/densita' corrette ai pezzi (motori, celle, ecc. — vedi le masse reali/stimate gia' note in
   `STATO_robot_2ruote_RL.md`).
3. Esportare l'URDF da CROSS -> sostituisce `robot_sim\robot.urdf` e `robot_sim\meshes\`.
4. Adattare `robot_cfg.py`/`robot_env.py` ai nomi di link/giunti del nuovo URDF (potrebbero cambiare
   rispetto a `giunto_ruota_sx/dx`, `base_link`, `ruota_sx/dx`).
5. Rilanciare train/play/valuta e confrontare con i risultati vecchi (8,33 gradi di appoggio, reward RL)
   per verificare che il nuovo modello sia equivalente.
