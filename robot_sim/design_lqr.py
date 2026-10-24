"""Controllo LQR, modello del motore e ricerca della coppia minima (simulatore leggero a 2 gradi di liberta').

A COSA SERVE QUESTO FILE
    1. Contiene il dizionario MOTORE: i parametri del motore usati in TUTTO il progetto (anche dall'ambiente RL).
    2. Contiene le equazioni fisiche del robot come pendolo rovesciato su due ruote (funzione `dinamica`).
    3. Calcola il controllo LQR (funzione `lqr`): 4 guadagni che, dalle masse del CAD, danno la tensione ai motori.
    4. Con `python design_lqr.py` prova il controllo per varie coppie di motore e stampa la coppia minima che serve.

COORDINATE (modello a 2 gradi di liberta')
    x   = avanzamento dell'asse delle ruote (m)
    phi = angolo del baricentro del corpo rispetto alla verticale sull'asse (+ = in avanti). phi = 0 e' l'equilibrio.
Il modello e' non lineare (Lagrange) e include l'inerzia del rotore riflessa dalla riduzione, il motore DC e i fermi dei piedi.

ATTENZIONE: tutti i parametri del motore sono TIPICI, non misurati. Vanno verificati con il motore vero.
Uso: env_isaac6\\Scripts\\python.exe robot_sim\\design_lqr.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm, solve_discrete_are

P = json.loads((Path(__file__).parent.parent / "robot_config" / "robot_params.json").read_text())
G = 9.81

# ---- Motore tipico JGA25-370 6 V, ~280 RPM (DA VERIFICARE con il motore vero) ----------------------------------------
# Modello DC: coppia = (tau_stall/v_nom) * V  -  (tau_stall/w0) * velocita'  -  attrito.
# Quando si ha il motore vero: misurare tau_stall (coppia a ruota bloccata), w0 (velocita' a vuoto), il rapporto di
# riduzione, l'attrito e la zona morta, e correggere qui. Tutto il progetto (LQR e RL) legge questi valori.
MOTORE = dict(
    tau_stall=0.30,       # N*m: coppia a ruota bloccata a 6 V, all'albero d'uscita (0,3 N*m = 3 kg*cm; sweep: 0,05-0,7)
    w0=280 * 2 * np.pi / 60,  # rad/s: velocita' a vuoto a 6 V (280 giri/min = 29,3 rad/s = 0,95 m/s con ruote da 65 mm)
    v_nom=6.0,            # V: tensione a cui valgono tau_stall e w0
    rapporto=34.0,        # rapporto di riduzione del riduttore (tipico per questi motori; moltiplica l'inerzia del rotore)
    j_rotore=4e-7,        # kg*m^2: inerzia di un rotore (tipica 370) -> vista dalla ruota vale j_rotore * rapporto^2
    attrito=0.01,         # N*m: attrito secco all'uscita, per ciascun motore
    v_batt=7.4,           # V: batteria 2S nominale (8,4 V piena, 6,6 V quasi scarica)
    pwm_max=0.8,          # frazione massima del PWM usata: tensione massima ai motori = 7,4 * 0,8 = 5,9 V
)

# ---- Parametri del corpo, letti dal json esportato da Fusion 360 (robot_config/robot_params.json) ------------------
body, wh = P["corpo_base"], P["ruota"]
m_b = body["massa"]                        # kg: massa del corpo (senza ruote)
cx, cz = body["com"][0], body["com"][2]    # m: baricentro del corpo rispetto all'asse delle ruote (avanti, in alto)
l = float(np.hypot(cx, cz))                # m: distanza asse-baricentro = lunghezza del "pendolo" (~28 mm: molto corto!)
alpha = float(np.arctan2(cx, cz))          # rad: baricentro davanti all'asse -> equilibrio con il corpo inclinato indietro di alpha
I_b = body["inerzia_com"][1][1]            # kg*m^2: inerzia del corpo attorno all'asse y, rispetto al suo baricentro
m_w = 2 * wh["massa_con_gomma"]            # kg: massa delle due ruote insieme (con la gomma)
r = wh["raggio_esterno"]                   # m: raggio ruota (32,5 mm)
I_w = 2 * wh["inerzia_com_ruota_sx"][1][1] # kg*m^2: inerzia delle due ruote attorno al loro asse

# Fermi dei piedini: il corpo non puo' inclinarsi oltre questi angoli perche' il piede tocca terra.
PHI_FRONT = alpha + np.radians(8.4)        # appoggio sul piede anteriore: phi massimo (tocca a 8,4 gradi in avanti)
PHI_REAR = alpha - np.radians(19.3)        # appoggio sul piede posteriore: phi minimo (tocca a 19,3 gradi indietro)


def modello(motore):
    """Costanti che dipendono dal motore: inerzia riflessa dei rotori e massa equivalente in avanti (M11)."""
    ja = 2 * motore["j_rotore"] * motore["rapporto"] ** 2   # inerzia riflessa dei due rotori
    return dict(M11=m_b + m_w + (I_w + ja) / r**2, ja=ja)


def dinamica(s, V, motore, md):
    """Equazioni del robot (pendolo rovesciato su due ruote). Ritorna la derivata dello stato.

    s = [x, xdot, phi, phidot]   V = tensione (V) data a entrambi i motori.
    Il motore produce una coppia sulle ruote e, per reazione, una coppia opposta sul corpo: e' per questo che spingere
    le ruote in avanti fa inclinare il corpo all'indietro. La matrice M e' la matrice di massa (con l'inerzia dei rotori),
    `rhs` raccoglie le forze: coppia del motore e gravita'.
    """
    x, xd, ph, phd = s
    k = motore["tau_stall"] / motore["v_nom"]            # N*m/V
    b = motore["tau_stall"] / motore["w0"]               # N*m*s/rad
    w_rel = xd / r - phd
    tau = k * V - b * w_rel - motore["attrito"] * np.tanh(w_rel / 0.5)
    T = 2 * tau
    M = np.array([[md["M11"], m_b * l * np.cos(ph) - md["ja"] / r],
                  [m_b * l * np.cos(ph) - md["ja"] / r, m_b * l**2 + I_b + md["ja"]]])
    rhs = np.array([T / r + m_b * l * np.sin(ph) * phd**2,
                    -T + m_b * G * l * np.sin(ph)])
    acc = np.linalg.solve(M, rhs)
    return np.array([xd, acc[0], phd, acc[1]])


def lqr(motore, md, q=(0.5, 2.0, 400.0, 10.0), rv=1.0, dt=0.01):
    """Calcola i 4 guadagni K del controllo LQR: tensione = -K @ [x, xdot, phi, phidot].

    Passi: (1) linearizza `dinamica` attorno all'equilibrio (matrici A e B, per differenze finite);
           (2) le discretizza a 100 Hz (dt); (3) risolve l'equazione di Riccati, che minimizza il "costo"
           somma(q_i * stato_i^2) + rv * tensione^2.
    q = peso di (posizione, velocita', inclinazione, velocita' di inclinazione): piu' e' alto, piu' quella grandezza e' tenuta vicina a zero.
          L'inclinazione (400) pesa molto: la priorita' e' stare dritti. rv = quanto "costa" usare tensione.
    """
    f = lambda s, v: dinamica(s, v, motore, md)
    e = 1e-6
    A = np.column_stack([(f(np.eye(4)[i] * e, 0) - f(-np.eye(4)[i] * e, 0)) / (2 * e) for i in range(4)])
    B = ((f(np.zeros(4), e) - f(np.zeros(4), -e)) / (2 * e)).reshape(4, 1)
    n = 5
    M = expm(np.block([[A, B], [np.zeros((1, 5))]]) * dt)
    Ad, Bd = M[:4, :4], M[:4, 4:]
    Pm = solve_discrete_are(Ad, Bd, np.diag(q), np.array([[rv]]))
    K = np.linalg.solve(Bd.T @ Pm @ Bd + rv, Bd.T @ Pm @ Ad)
    return K.flatten()


def simula(motore, K, s0, durata=5.0, dt=0.001, ctrl=0.01, spinta=None, ritardo=0.0, v_batt=None):
    """Simula il robot con il controllo K. Ritorna (successo, traccia).

    s0 = stato iniziale [x, xdot, phi, phidot]; durata in s; dt = passo della fisica (1 ms); ctrl = periodo del controllo (10 ms);
    spinta = (istante, delta velocita' angolare) per dare un colpo al corpo; ritardo = ritardo misura->comando in s;
    v_batt = tensione batteria alternativa. Successo = a fine prova e' dritto (< 3 gradi) e quasi fermo (< 0,1 m/s).
    """
    md = modello(motore)
    vmax = (v_batt or motore["v_batt"]) * motore["pwm_max"]
    s = np.array(s0, float)
    u = 0.0
    pend = []
    traccia = []
    for i in range(int(durata / dt)):
        t = i * dt
        if spinta and abs(t - spinta[0]) < dt / 2:
            s[3] += spinta[1]   # impulso sulla velocita' angolare del corpo (rad/s)
        if i % int(ctrl / dt) == 0:
            sm = s.copy()
            sm[2] -= 0.0
            pend.append(-K @ sm)
            lag = int(round(ritardo / ctrl))
            u = float(np.clip(pend[-1 - lag] if len(pend) > lag else 0.0, -vmax, vmax))
        # RK4 con tensione costante nell'intervallo
        k1 = dinamica(s, u, motore, md); k2 = dinamica(s + dt / 2 * k1, u, motore, md)
        k3 = dinamica(s + dt / 2 * k2, u, motore, md); k4 = dinamica(s + dt * k3, u, motore, md)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if s[2] > PHI_FRONT and s[3] > 0:      # fermo piede anteriore
            s[2], s[3] = PHI_FRONT, 0.0
        if s[2] < PHI_REAR and s[3] < 0:       # fermo piede posteriore
            s[2], s[3] = PHI_REAR, 0.0
        traccia.append((t, s[0], s[2], u))
        if abs(s[2]) > np.radians(60):
            return False, np.array(traccia)
    t = np.array(traccia)
    ok = abs(s[2]) < np.radians(3) and abs(s[1]) < 0.1
    return ok, t


def prova(tau_stall, **kw):
    """Prova un motore con coppia di stallo `tau_stall`: rialzo dal piede anteriore e spinte sul corpo avanti/indietro."""
    mot = dict(MOTORE, tau_stall=tau_stall)
    K = lqr(mot, modello(mot))
    ris = {}
    # 1) rialzarsi dal piede anteriore (phi = PHI_FRONT, fermo)
    ok, tr = simula(mot, K, [0, 0, PHI_FRONT, 0], durata=6.0, **kw)
    ris["rialzo"] = (ok, float(np.max(np.abs(tr[:, 1]))))
    # 2) spinta: velocita' improvvisa 0.3 m/s a 1 s partendo in equilibrio
    for w in (0.5, 1.0):
        res = [simula(mot, K, [0, 0, 0.0, 0], durata=6.0, spinta=(1.0, sgn * w), **kw) for sgn in (1, -1)]
        ris[f"spinta{w}"] = (all(o for o, _ in res), float(max(np.max(np.abs(t[:, 2])) for _, t in res)))
    return ris, K


if __name__ == "__main__":
    print(f"parametri: m_corpo={m_b*1e3:.0f} g  l={l*1e3:.1f} mm  alpha={np.degrees(alpha):.1f} gradi  "
          f"J_riflessa={modello(MOTORE)['ja']:.2e} kg m2 (ruote {I_w:.1e})")
    print(f"{'tau_stall':>9} | {'rialzo':>7} {'x_max m':>7} | {'dw=0.5':>7} {'phi_max':>7} | {'dw=1.0':>7}")
    for ts in [0.05, 0.08, 0.12, 0.2, 0.3, 0.45, 0.7]:
        ris, K = prova(ts)
        r1, r2, r3 = ris["rialzo"], ris["spinta0.5"], ris["spinta1.0"]
        print(f"{ts:9.2f} | {'OK' if r1[0] else 'NO':>7} {r1[1]:7.2f} | {'OK' if r2[0] else 'NO':>7} "
              f"{np.degrees(r2[1]):7.1f} | {'OK' if r3[0] else 'NO':>7}")
    print("K (tau_stall 0.3):", np.round(prova(0.3)[1], 2))
