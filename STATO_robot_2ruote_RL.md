# Robot a 2 ruote in equilibrio con RL — stato del progetto

Aggiornato: 2026-10-23. Obiettivo (da `PROGETTO_ROBOT.md`): robot a due ruote in equilibrio addestrato con RL in
Isaac Lab, poi inseguimento e ostacoli. Questo file riassume cosa è stato fatto e **dove si trova ogni cosa**.

## 1. Perché in Isaac Lab non si vedono i parametri
La finestra di Isaac Sim mostra solo la **scena 3D** (pavimento, robot, luci) e il pannello Stage con le parti del robot
(masse e giunti si leggono selezionando un corpo). Tutto il resto è **codice Python**, non ha un pannello:

| Cosa | Dove è |
|---|---|
| Masse, baricentro, inerzie (da Fusion 360) | `robot_config\robot_params.json` |
| Modello fisico (URDF) e colori | `robot_sim\gen_urdf.py` → `robot_sim\robot.urdf` |
| Forme 3D (STL del CAD) | `robot_sim\meshes\` |
| Giunti delle ruote, inerzia del rotore (armature), limiti | `robot_sim\robot_cfg.py` e `robot_sim\robot_rl\robot_env.py` |
| **Motore (coppia, velocità, attrito, batteria, PWM)** | `robot_sim\design_lqr.py` → dizionario `MOTORE` (**valori tipici, da verificare**) |
| **Ricompense, durata episodio, comandi, reset** | `robot_sim\robot_rl\robot_env.py` → `RobotEnvCfg` |
| Rete e algoritmo (PPO, 64×64) | `robot_sim\robot_rl\robot_env.py` → `RobotPPORunnerCfg` |

Per cambiare un parametro si modifica il file e si rilancia: la GUI non lo mostra né lo cambia.

## 2. CAD (Fusion 360)
File attuale: `export\robot_2ruote_esp32_varD.f3d` / `.step` (con gomme: `..._varC_gomme`; la D ha motori a 16 mm e ruote a ±88 mm).
- ESP32-S3 DevKitC-1 (equilibrio) + scheda camera ESP32-S3/OV2640, 3 ToF, batteria 2S, driver TB6612, step-down, IMU.
- Baricentro: x +3,0 mm (davanti all'asse, così a riposo cade sul piede anteriore), z 23 mm sopra l'asse. Massa 476,5 g.
- Piede anteriore a 8 mm da terra (tocca a 8,4° in avanti), posteriore a 18 mm (tocca a 19,3° indietro). Equilibrio previsto a −7,5°.
- Ruote 65 mm con gola per 2-3 strati di camera d'aria da corsa (attrito stimato 0,8).
- Masse = stime (motori 70 g, celle 46 g, stampa 0,6 g/cm³): **pesare i pezzi veri**.

## 3. Ambienti software
- `env_isaac6` (Python 3.12, **Isaac Sim 6.2 + Isaac Lab v3.0.0**, torch 2.11+cu128): quello in uso. La finestra funziona.
- `env_isaac` (Isaac Sim 5.1 + Lab 2.3.2): riserva. La finestra crasha (renderer RTX/Blackwell).
- Differenze v3: `--visualizer none` (senza finestra) / `--visualizer kit` (con finestra); quaternioni (x,y,z,w); task senza `-v0`.

## 4. Risultati
**Verifica fisica:** il modello si appoggia a 8,33° sul piede anteriore (CAD: 8,4°); massa 0,4765 kg.

**LQR (controllo classico, 4 guadagni):** rialzo dal piede e spinte ok da ~0,08 N·m di coppia di stallo a 6 V (sensori ideali).
Fascia buona 0,12–0,3 N·m. **Il ritardo è critico:** regge fino a ~20 ms, con 40 ms cade. Pendolo corto (l = 28 mm) e inerzia
riflessa dei rotori (~9e-4 kg·m²) molto alta.

**RL (PPO, rsl_rl):** ambiente `Robot-Balance-Direct`. Con 2048 robot, 1000 iterazioni ≈ 7 min.
Ultimo addestramento (premio a campana sulla velocità), valutazione con `valuta_rete.py`:

| Comando | Velocità ottenuta | Piede ant. a terra | Piede post. a terra | Cadute |
|---|---|---|---|---|
| fermo | −0,02 m/s | 0% | 0% | 0 |
| avanti 0,2 | +0,11 m/s | 0% | 0% | 0 |
| indietro 0,2 | −0,10 m/s | 0% | 5,3% | 0 |

Sta in piedi e segue il comando a circa metà velocità. Checkpoint: `logs\rsl_rl\robot_balance\2026-10-23_18-51-51\model_999.pt`.
Un addestramento visibile (128 robot, 600 iter) era in corso con la finestra aperta.

## 5. Comandi (PowerShell, dalla cartella `G:\robotTest`)
```
$env:OMNI_KIT_ACCEPT_EULA='Y'
# addestrare (senza finestra, veloce)
env_isaac6\Scripts\python.exe -u robot_sim\train_robot.py --rl_library rsl_rl --task Robot-Balance-Direct --visualizer none --num_envs 2048 --max_iterations 1000
# addestrare guardando (lento)
env_isaac6\Scripts\python.exe -u robot_sim\train_robot.py --rl_library rsl_rl --task Robot-Balance-Direct --visualizer kit --num_envs 128 --max_iterations 600
# vedere una rete addestrata in finestra
env_isaac6\Scripts\python.exe -u robot_sim\play_robot.py --rl_library rsl_rl --task Robot-Balance-Direct --num_envs 3 --visualizer kit --checkpoint <model.pt>
# misurare una rete (piedi a terra, velocità, cadute)
env_isaac6\Scripts\python.exe -u robot_sim\valuta_rete.py --visualizer none --checkpoint <model.pt>
# demo del controllo LQR con 3 robot in finestra
env_isaac6\Scripts\python.exe -u robot_sim\prova_lqr_isaac.py --visualizer kit --solo 0.3
```
Gli script di Isaac **non terminano da soli**: dopo l'uso fermare il processo Python. Non filtrare i log con "Error" (ci sono molti avvisi).
Il visualizzatore da browser (registrazione) è `robot_sim\viewer.html` (`registra.py` + `crea_viewer.py`).

## 6. Cosa manca per il robot vero
1. **Motore vero:** misurare coppia di stallo, velocità a vuoto, attrito, zona morta e rapporto di riduzione; correggere `MOTORE`.
2. **Condizioni dure** in simulazione: rumore e deriva dell'IMU, ritardo variabile (10–30 ms), zona morta, variazioni di massa,
   baricentro, attrito e batteria (6,6–8,4 V), spinte casuali.
3. Migliorare l'inseguimento di velocità (ora ~metà del comando) e il piede posteriore all'indietro.
4. Esportare la rete (ONNX → array C) e misurare il tempo di inferenza sull'ESP32-S3 (obiettivo < 1 ms; 64×64 ≈ 5 mila parametri).
5. Poi: ostacoli (3 ToF), slalom, inseguimento del gatto con camera.

## 7. Limiti noti
- Parametri del motore tipici, sensori ideali, nessun ritardo/rumore/randomizzazione nell'addestramento attuale.
- Collisioni semplificate (piastre, piedi e ruote; motori e batterie solo come massa).
- Con più robot nello stesso script ricordare l'offset di ogni ambiente nello stato iniziale (altrimenti si sovrappongono).
