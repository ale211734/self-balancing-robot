# Robot autobilanciato a 2 ruote addestrato con RL — brief di progetto

Documento di contesto per Claude Code. Descrive cosa vogliamo costruire, l'architettura scelta e, soprattutto, cosa va verificato **in simulazione prima di comprare l'hardware**.

## Obiettivo

Costruire un piccolo robot a 2 ruote autobilanciato (pendolo rovesciato), con il controllo dell'equilibrio appreso con reinforcement learning (PPO) in simulazione e poi trasferito sul robot reale (sim2real). Ispirazione: Microduck di Hugging Face / Pollen Robotics, presentato al GTC Berlin 2026.

Traguardi, in ordine:

1. **Equilibrio**: stare in piedi, fermo, resistendo a piccole spinte.
2. **Guida a comando**: seguire comandi di velocità lineare e di rotazione (joystick o Wi-Fi).
3. **Evitare ostacoli e pareti** con 3 sensori ToF.
4. **Slalom** tra birilli.
5. **Seguire un bersaglio** con la camera: prima un marcatore ArUco, poi il gatto di casa (solo inseguimento lento, senza raggiungerlo, fermandosi a distanza).

## Vincoli

- Budget robot: circa 90-110 €.
- Usare il più possibile strumenti NVIDIA sul PC (Isaac Sim / Isaac Lab, TAO Toolkit, TensorRT). Ripiego più leggero: mjlab / MuJoCo Warp.
- A bordo del robot non c'è hardware NVIDIA: la policy gira su un ESP32-S3.
- Lo sviluppatore conosce bene C, C++ e Python; ha stampante 3D, stazione saldante e alimentatore da banco.

## Hardware

### PC di sviluppo (già disponibile)

- GPU NVIDIA RTX 5060 Ti 16 GB (architettura Blackwell: servono CUDA e PyTorch recenti, presumibilmente CUDA 12.8 o superiore, da verificare).
- 64 GB di RAM.
- Usato per: simulazione e training RL, rilevamento del gatto (detector), invio dei comandi al robot via Wi-Fi.

### Robot (da acquistare dopo le verifiche in simulazione)

| Componente | Note |
|---|---|
| 2 motori DC JGA25-370 con encoder Hall, 6 V, 280 RPM | Encoder a quadratura, 6 fili. Rapporto di riduzione e impulsi per giro da leggere nella scheda tecnica |
| Driver TB6612FNG | 2 canali, PWM a circa 20 kHz |
| IMU MPU6050 (modulo GY-521) | Dati grezzi via I2C a 400 kHz, fusione fatta da noi |
| ESP32-S3 DevKitC-1 | Scheda dell'equilibrio: legge i sensori ed esegue la policy |
| ESP32-S3 con camera OV2640 | Seconda scheda: invia il video al PC via Wi-Fi |
| 3 sensori ToF VL53L1X | Sinistra, centro, destra; cono di circa 27°; indirizzi I2C assegnati via pin XSHUT |
| Pacco 2S 18650 (7,4 V nominali, 8,4 V a piena carica) | Capacità reale probabilmente 1000-1500 mAh; spegnimento sotto circa 6,6 V |
| Regolatore step-down 5 V da 2-3 A, condensatori, interruttore, fusibile | Alimentazione della camera separata da quella dei motori |
| Telaio, ruote (circa 65 mm), appoggi anti-caduta | Stampati in 3D |

## Architettura

```
PC (RTX 5060 Ti)
  ├─ Training: Isaac Lab (o mjlab) + PPO  ──►  policy esportata (ONNX → array C)
  └─ Runtime: detector (TAO/TensorRT) sul video ──► angolo e distanza del bersaglio
                         ▲ video Wi-Fi                     │ comandi Wi-Fi (lenti, 10-20 Hz)
                         │                                 ▼
Robot   ESP32-S3 + camera ──(UART)──► ESP32-S3 equilibrio (100-200 Hz)
                                         ├─ IMU, encoder, 3 ToF
                                         ├─ policy RL: equilibrio + inseguimento comandi
                                         └─ livello di sicurezza: i ToF fermano il robot vicino agli ostacoli
```

Principi:

- **Due livelli separati.** La policy RL tiene l'equilibrio e segue comandi di velocità lineare e di rotazione. Inseguimento, slalom ed evitamento producono quei comandi (all'inizio come regole in C++, eventualmente in seguito appresi).
- **L'equilibrio non dipende mai dal Wi-Fi né dalla camera.** Se i comandi dal PC mancano per più di circa 0,5 s, il robot si ferma restando in equilibrio.
- **La camera decide il bersaglio** ("c'è un gatto a 15° a sinistra"); **i ToF trattano tutto come ostacolo** e hanno sempre la precedenza.

## Cosa verificare in simulazione (prima di comprare)

### 1. Il PC regge gli strumenti

- [ ] Isaac Sim Compatibility Checker: la 5060 Ti è sotto il minimo dichiarato (RTX 4080 16 GB), quindi verificare che funzioni.
- [ ] Isaac Lab installato con CUDA e PyTorch compatibili con Blackwell.
- [ ] Training del task Cartpole (stesso problema fisico del robot) in modalità headless: tempo di training, ambienti paralleli sostenibili, VRAM usata (`nvidia-smi`).
- [ ] Se Isaac Sim è troppo pesante: ripetere con mjlab / MuJoCo Warp.

### 2. Modello del robot

- [ ] Modello del robot (URDF o MJCF) con masse, dimensioni e inerzie realistiche: batteria (circa 90-100 g, montata bassa), due schede, motori, telaio.
- [ ] Motori modellati dalla scheda tecnica: coppia di stallo, velocità a vuoto (280 RPM a 6 V), alimentazione reale 7,4 V con PWM limitato al 70-80%.
- [ ] Encoder con risoluzione reale (impulsi per giro × rapporto di riduzione).

### 3. Il hardware scelto è sufficiente

- [ ] **Coppia dei motori:** il robot recupera l'equilibrio dopo una spinta? Confrontare 280 RPM con 170 RPM e diametri di ruota diversi.
- [ ] **Velocità:** con comandi tra 0,2 e 0,5 m/s resta margine per recuperare l'equilibrio.
- [ ] **Frenata:** spazio necessario per fermarsi partendo da 0,3 m/s (serve a fissare la soglia dei ToF, circa 40-50 cm).
- [ ] **Frequenza di controllo:** la policy funziona a 100 Hz; verificare cosa succede a 50 Hz.
- [ ] **Dimensione della rete:** la policy più piccola che funziona (obiettivo: MLP di poche migliaia di parametri, eseguibile sull'ESP32-S3 in meno di 1 ms).

### 4. Robustezza sim2real (domain randomization)

Introdurre e verificare che la policy resti stabile con:

- [ ] zona morta dei motori (circa 10-20% di PWM) e attrito del riduttore;
- [ ] rumore e offset dell'IMU, rumore dell'encoder a bassa velocità;
- [ ] ritardo tra misura e comando (qualche millisecondo);
- [ ] variazioni di massa, posizione del baricentro e tensione della batteria (6,6-8,4 V);
- [ ] attrito ruota-pavimento variabile;
- [ ] spinte casuali durante gli episodi;
- [ ] partenza da appoggiato sugli appoggi anti-caduta (comportamento "rialzarsi").

### 5. Compiti di alto livello

- [ ] Guida con comandi casuali di velocità e rotazione.
- [ ] Evitamento con 3 raggi simulati (ray caster o rangefinder) che imitano i ToF: rumore, cono stretto, letture sbagliate quando il robot si inclina (filtrare usando l'angolo dell'IMU).
- [ ] Slalom con birilli in posizioni casuali.
- [ ] Inseguimento di un bersaglio mobile, ricevendo **angolo e distanza del bersaglio** (non immagini) con rumore e ritardo di 100-200 ms, fermandosi a circa 50-60 cm. Caso di ostacolo tra robot e bersaglio: l'ostacolo ha la precedenza.

### 6. Verifiche fuori simulazione, gratuite

- [ ] Detector di oggetti con classe "gatto" (modello COCO pre-addestrato dal catalogo NVIDIA NGC, eventualmente con fine-tuning in TAO) provato con una webcam sul PC con il gatto vero: affidabilità e fotogrammi al secondo.
- [ ] Policy esportata in ONNX e poi in array C, eseguita su un microcontrollore con input finti: misura del tempo di inferenza con `micros()`.

## Primi compiti per Claude Code

1. Preparare l'ambiente (Isaac Lab o mjlab) e uno script per lanciare e misurare il training del Cartpole.
2. Scrivere il modello del robot a 2 ruote con parametri configurabili (masse, ruote, motori).
3. Definire l'ambiente RL: osservazioni (inclinazione, velocità angolare, velocità delle ruote, comandi), azioni (comandi ai due motori), reward (stare dritto, seguire i comandi, penalità per cadute ed energia), terminazione oltre un angolo limite.
4. Aggiungere la domain randomization dell'elenco al punto 4.
5. Esportare la policy in ONNX e scrivere un convertitore in array C con una funzione di inferenza in C puro.
6. Solo dopo: ambienti per evitamento, slalom e inseguimento.

## Domande aperte

- Isaac Sim gira in modo accettabile sulla 5060 Ti, o conviene mjlab?
- I motori a 280 RPM hanno coppia sufficiente? Meglio 170 RPM con ruote più grandi?
- Quanta latenza aggiunge il video ESP32-S3 → PC → robot?
- Come si comportano i ToF quando il robot si inclina?
