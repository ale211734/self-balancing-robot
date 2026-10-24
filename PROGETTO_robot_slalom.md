# Robot a due ruote in equilibrio con slalom: stato del progetto

Data: 2026-10-21

## Obiettivo
Verificare, prima di comprare, se il progetto e' fattibile col proprio hardware: robot a due ruote
(pendolo inverso) che sta in equilibrio e fa lo slalom tra birilli, controllato da Arduino Mega.
Budget componenti: circa 60-65 EUR (lista nella conversazione originale).

## Esito dei test sul PC (passo 1 e 2 della guida)
PC: Windows 11 Pro, i7-12700, 64 GB RAM, RTX 5060 Ti 16 GB (driver 610.88), Python 3.11.9.

Test Cartpole con Isaac Lab (headless): **superato**
- 150 iterazioni, 9,8 milioni di passi, 49 secondi
- VRAM massima misurata: 4,2 GB su 16 GB
- Tutti gli episodi arrivano al timeout, nessuno esce dai limiti (asta in equilibrio)
- Non provata la variante Cartpole con camera RGB (non serve: il robot usa sensori di distanza)

Conclusione: il PC regge Isaac Sim e Isaac Lab per questo progetto. Il PC serve solo per addestrare;
sul robot gira solo la rete addestrata, sul Mega (da verificare al passo 4).

## Software installato (G:\robotTest)
- `env_isaac\` ambiente virtuale Python 3.11
- PyTorch **2.7.0+cu128** e torchvision 0.22.0+cu128 (necessari per la GPU Blackwell)
- Isaac Sim **5.1.0**
- `IsaacLab\` fissato al tag **v2.3.2** (il ramo `main` punta a Isaac Sim 6.2 e non ha piu' `isaaclab.bat`)
- `tensordict==0.8.3` (compatibile con torch 2.7; pip ne installa uno piu' nuovo che rompe tutto)
- `setuptools<81` (serve per compilare `flatdict`)
- `fusion360-mcp-server` clonato in `fusion360-mcp-server\`

Problemi incontrati e soluzioni:
- `isaaclab.bat --install` non trova Python: installati i pacchetti a mano con
  `pip install -e source/isaaclab -e source/isaaclab_assets -e source/isaaclab_tasks -e "source/isaaclab_rl[rsl-rl]"`
- pip sostituisce torch con una versione piu' nuova: rimettere
  `pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128`
- Al primo avvio di Isaac Sim va accettata la EULA NVIDIA (variabile `OMNI_KIT_ACCEPT_EULA=Y`)

Comando del test:
```
$env:OMNI_KIT_ACCEPT_EULA='Y'
G:\robotTest\env_isaac\Scripts\python.exe G:\robotTest\IsaacLab\scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Cartpole-v0 --headless
```

## MCP di Fusion 360
- Server: https://github.com/faust-machines/fusion360-mcp-server (licenza MIT)
- Add-in copiato in `%APPDATA%\Autodesk\Autodesk Fusion 360\API\AddIns\Fusion360MCP`
- Registrato in Claude Code (utente) con:
  `claude mcp add -s user fusion360 -- uvx --with "mcp<2" fusion360-mcp-server --mode socket`
  (`mcp<2` e' necessario: con la libreria piu' recente il server si chiude subito)
- Per usarlo: aprire Fusion, Shift+S -> Add-Ins -> Fusion360MCP -> Run (porta 9876, solo localhost),
  poi riavviare Claude Code. Nota: lo strumento `execute_code` esegue Python arbitrario in Fusion,
  la porta non ha autenticazione: non esporla in rete.
- Disinstallare: `claude mcp remove fusion360` e cancellare la cartella dell'add-in.

## Modello CAD (Fusion 360)
File: `G:\robotTest\export\robot_slalom.f3d` (parametrico) e `robot_slalom.step` (neutro).
Punti di ripristino intermedi: `G:\robotTest\checkpoints\` (cancellabili).
Il documento in Fusion non e' salvato nel cloud Autodesk (Ctrl+S per farlo).

Convenzioni: origine tra le ruote, asse X in avanti, Y sinistra (+Y = "sx"), Z in alto. Unita' mm.
32 pezzi: piastra_bassa, piastra_alta, 4 colonnine, 4 mezze staffe motore, motore_sx/dx, ruota_sx/dx,
portabatterie, 2 celle 18650, driver TB6612, step-down 5V, IMU MPU6050, Arduino Mega con 6 distanziali,
3 sensori ToF, 2 appoggi anti-caduta.

Dettaglio motore JGA25-370: encoder, corpo con due piani laterali, riduttore, boccola di centraggio,
albero a D, 2 fori M3 a 17 mm. Ruota con lato piatto nel foro per il grano.

Parametri principali (Modifica -> Parametri utente, 55 parametri):
`motor_d`, `motor_gb_L`, `motor_body_L`, `motor_enc_L`, `shaft_d`, `shaft_L`, `wheel_D`, `wheel_w`,
`plate_L`, `plate_W`, `plate_t`, `deck_gap`, `clamp_L`, `clamp_wall`, `batt_*`, `cell_*`, `mega_*`, `tof_*`.

### Versione ESP32 (2026-10-23) - file attuale: `export\robot_2ruote_esp32_varB.f3d` / `.step`
Arduino Mega sostituito da ESP32-S3 DevKitC-1 (piastra bassa, centrata) + scheda camera ESP32-S3/OV2640
(in piedi davanti sulla piastra alta, obiettivo verso +X). Batterie sulla piastra alta, centro a x = -12 mm.
Driver, step-down e IMU sulla piastra bassa vicino all'asse (x = 0, ai lati). ToF centrale sulla piastra bassa.
Baricentro stimato (masse da scheda tecnica, stampa 0,6 g/cm3): 474 g, x = 0, y = 0, z = 55,6 mm
(23 mm sopra l'asse ruote). I corpi nuovi hanno misure fisse, non parametri utente.
Variante C (`robot_2ruote_esp32_varC.f3d`, attuale): batterie spostate in avanti di 14 mm = baricentro 3 mm davanti
all'asse (x = +3,0, 23,2 mm sopra l'asse); piede posteriore accorciato (distanza da terra 18 mm, tocca a 19,3 deg).
Piede anteriore a 8 mm (tocca a 8,4 deg in avanti). Inclinazione d'equilibrio prevista: 7,4 deg all'indietro.
Idea d'uso: a riposo appoggiato sul piede anteriore, si rialza accelerando le ruote in avanti; poi equilibrio.
Limite: a riposo esiste comunque anche l'appoggio posteriore (se cade all'indietro oltre 7,4 deg); la policy deve
saper rialzarsi anche da li', e allo spegnimento il robot va inclinato in avanti dal controllo.
Varianti precedenti: `robot_2ruote_esp32.f3d` (batterie in basso), `..._varA.f3d` (batterie in alto, pesi lontani dall'asse).

### Cose da verificare / limiti noti
- Le quote del motore vengono da schede tecniche che tra venditori non coincidono: lunghezza riduttore
  (22 mm, range 17-27), diametro encoder (22), distanza tra i piani laterali (20,5), boccola (7).
  Misurare il motore vero con il calibro e correggere i parametri.
- Il portabatterie e' bloccato a `batt_W` = 40 mm: cambiarlo manda in crisi lo schizzo.
- Gli schizzi di Fusion vanno cambiati un parametro alla volta: due parametri insieme possono
  deformare lo schizzo.
- Mancano BMS, cavi, connettori, pin del Mega; i ToF sono scatole con le misure giuste.
- Un sesto distanziale del Mega ha le quote non collegate ai parametri.
- La cronologia di Fusion e' stata riavvolta una volta durante il lavoro (la piastra bassa e' stata rifatta):
  non usare Ctrl+Z o cliccare sulla cronologia mentre lo script lavora.

## Rischio sim-to-real (importante)
Nessuna simulazione garantisce che il robot vero funzioni. La simulazione serve a scartare progetti
sbagliati prima di spendere. Per ridurre il rischio: masse e misure reali, rumore sull'IMU, ritardo
sui comandi (10-20 ms), attrito variabile, parametri randomizzati (+-20%).
L'equilibrio puo' essere fatto anche con PID/LQR senza rete neurale; la rete serve soprattutto per lo slalom.
Ordine di acquisto consigliato: prima il minimo (motori, driver, IMU, batteria, circa 35 EUR) per
l'equilibrio con PID, poi i sensori ToF (15 EUR) solo quando lo slalom funziona in simulazione.

## Prossimi passi
1. Estrarre da Fusion massa e baricentro di ogni pezzo (`get_physical_properties`, con densita' reali
   o masse dalle schede tecniche, es. cella 18650 circa 45 g, Mega circa 37 g).
2. Scrivere l'URDF con giunti rotanti motore-ruota e giunti fissi per il telaio.
3. Creare il task di equilibrio in Isaac Lab (modello simile al Cartpole) e addestrarlo.
4. Aggiungere rumore, ritardi e randomizzazione dei parametri; poi lo slalom con sensori di distanza simulati.
5. Passo 4 della guida: caricare la rete (pesi in array C) sul Mega con input finti e misurare
   la durata di un'inferenza con `micros()`. Sotto 5 ms va bene per ~100 Hz; altrimenti ESP32 (6 EUR).
6. Comprare solo dopo che la simulazione funziona.
