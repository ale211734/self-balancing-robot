"""Coppia minima con ritardo tra misura e comando."""
import design_lqr as d

for rit in (0.02, 0.04):
    print("ritardo", rit, "s")
    for ts in (0.08, 0.12, 0.2, 0.3, 0.45, 0.7):
        ris, K = d.prova(ts, ritardo=rit)
        print(f"  tau_stall {ts:4.2f}: rialzo {ris['rialzo'][0]}  dw0.5 {ris['spinta0.5'][0]}  dw1.0 {ris['spinta1.0'][0]}")
