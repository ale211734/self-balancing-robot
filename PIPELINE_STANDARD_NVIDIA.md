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

## Correzione 2026-10-24: ripartenza letterale da zero

L'utente ha chiarito che "minimo sindacale" significa ripartire da zero sul serio: in `robot_sim\` resta
SOLO `vedi_robot.py` (strumento per vedere il robot nella finestra Isaac Sim). Cancellati anche
`design_lqr.py` (dizionario MOTORE), `robot_cfg.py` (ArticulationCfg/UrdfFileCfg), `robot_rl\robot_env.py`
(DirectRLEnv), `train_robot.py`, `play_robot.py`, `valuta_rete.py` — anche se erano "uso normale di Isaac
Lab" e non bridge, vanno riscritti da capo quando il nuovo URDF da FreeCAD e' pronto, non riadattati.
`vedi_robot.py` importa `ROBOT_CFG` da `robot_cfg.py`: restava non funzionante finche' non si riscriveva
`robot_cfg.py` insieme al nuovo URDF. L'utente ha poi chiesto di cancellare anche `vedi_robot.py`:
`robot_sim\` ora e' completamente vuota. Tutto (config robot, ambiente RL, script) va scritto da capo
dopo l'export URDF da FreeCAD.

## Prossimi passi
1. FATTO: `export\robot_2ruote_esp32_varD.step` importato in FreeCAD -> `export\robot_cross.FCStd`.
2. FATTO: in CROSS 3 link (base_link 394,1 g, ruota_sx/dx 41,2 g) e 2 giunti continuous; totale 476,5 g.
   Masse e inerzie calcolate con lo script nella guida (sezione "Calcolo di masse e inerzie") dalle masse
   per pezzo stimate in Fusion (da sostituire con i pesi veri). NON usare il pulsante "Calculate mass and
   inertia" di CROSS: usa solo il primo pezzo di ogni link e una sola densita'.
3. Esportare l'URDF da CROSS in `robot_description\` (OutputPath del robot): servono solo `urdf\` e
   `meshes\`; i file ROS generati (package.xml, CMakeLists.txt, launch\, rviz\) non servono a Isaac.
   `robot_description\` e' solo output generato (sovrascritto a ogni export); il codice Isaac Lab scritto
   a mano va in `robot_sim\`.
4. Scrivere da capo `robot_cfg.py` (ArticulationCfg/UrdfFileCfg) e l'ambiente RL (DirectRLEnv) usando i
   nomi di link/giunti del nuovo URDF (potrebbero differire da `giunto_ruota_sx/dx`, `base_link`,
   `ruota_sx/dx` usati prima).
5. Scrivere da capo gli script di training/visualizzazione/valutazione e confrontare i risultati con
   quelli vecchi (8,33 gradi di appoggio, reward RL) per verificare che il nuovo modello sia equivalente.
