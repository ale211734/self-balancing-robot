"""Genera robot.urdf dal file robot_params.json (esportato da Fusion 360).

Corpo base + due ruote con giunto continuo attorno a y. Frame: origine al centro dell'asse
ruote, x avanti, y sinistra, z su. Tutto in SI.
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
PARAMS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "robot_config" / "robot_params.json"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "robot.urdf"

p = json.loads(PARAMS.read_text())
base, wheel, feet = p["corpo_base"], p["ruota"], p["piedi"]
AX = 0.0325  # quota dell'asse da terra: nel frame del robot l'asse e' z = 0


def inertia_tag(I):
    return (
        f'<inertia ixx="{I[0][0]:.6e}" ixy="{I[0][1]:.6e}" ixz="{I[0][2]:.6e}" '
        f'iyy="{I[1][1]:.6e}" iyz="{I[1][2]:.6e}" izz="{I[2][2]:.6e}"/>'
    )


def box(name, x0, x1, y0, y1, z0, z1):
    """Collisione a scatola con estremi nel frame del robot."""
    return (
        f'    <collision name="{name}">\n'
        f'      <origin xyz="{(x0 + x1) / 2:.5f} {(y0 + y1) / 2:.5f} {(z0 + z1) / 2:.5f}"/>\n'
        f'      <geometry><box size="{x1 - x0:.5f} {y1 - y0:.5f} {z1 - z0:.5f}"/></geometry>\n'
        f"    </collision>\n"
    )


# piastre (130 x 140 x 3 mm) e piedi, in quote relative all'asse
z_low = 0.050 - AX
z_up = 0.083 - AX
coll = box("piastra_bassa", -0.065, 0.065, -0.070, 0.070, z_low, z_low + 0.003)
coll += box("piastra_alta", -0.065, 0.065, -0.070, 0.070, z_up, z_up + 0.003)
coll += box("piede_anteriore", feet["anteriore"]["x_min"], feet["anteriore"]["x_max"], -0.004, 0.004,
            feet["anteriore"]["quota_terra_da_asse"], z_low)
coll += box("piede_posteriore", feet["posteriore"]["x_min"], feet["posteriore"]["x_max"], -0.004, 0.004,
            feet["posteriore"]["quota_terra_da_asse"], z_low)

MESHES = HERE / "meshes"
COLORI = {  # nome parziale del pezzo -> (nome materiale, rgba)
    "piastra": ("pla", "0.85 0.85 0.85 1"), "colonnina": ("pla", "0.85 0.85 0.85 1"),
    "staffa": ("pla", "0.85 0.85 0.85 1"), "appoggio": ("piedi", "0.95 0.1 0.1 1"),
    "portabatterie": ("pla", "0.85 0.85 0.85 1"), "supporto": ("pla", "0.85 0.85 0.85 1"),
    "motore": ("metallo", "0.55 0.58 0.62 1"), "cella": ("batteria", "0.1 0.35 0.75 1"),
    "gomma": ("gomma", "0.08 0.08 0.08 1"), "ruota": ("ruota", "0.95 0.45 0.1 1"),
    "esp32": ("pcb", "0.05 0.35 0.15 1"), "cam_": ("pcb", "0.05 0.35 0.15 1"),
    "driver": ("pcb", "0.7 0.1 0.1 1"), "step_down": ("pcb", "0.1 0.1 0.6 1"), "IMU": ("pcb", "0.1 0.4 0.4 1"),
    "ToF": ("tof", "0.2 0.2 0.2 1"),
}


def colore(nome):
    for chiave, val in COLORI.items():
        if nome.startswith(chiave):
            return val
    return ("grigio", "0.5 0.5 0.5 1")


def visuals(prefissi_ruota=None, escludi_ruote=True, dy=0.0):
    """Un <visual> per ogni mesh STL esportata da Fusion (mm, frame mondo con origine a terra)."""
    out = ""
    for stl in sorted(MESHES.glob("*.stl")):
        nome = stl.stem
        e_ruota = nome.startswith(("ruota", "gomma"))
        if prefissi_ruota is None:
            if e_ruota:
                continue
        elif not nome.startswith(prefissi_ruota):
            continue
        mat, rgba = colore(nome)
        out += (
            f'    <visual name="{nome}">\n'
            f'      <origin xyz="0 {dy:.5f} {-AX:.5f}"/>\n'
            f'      <geometry><mesh filename="meshes/{stl.name}" scale="0.001 0.001 0.001"/></geometry>\n'
            f'      <material name="{mat}"><color rgba="{rgba}"/></material>\n'
            f"    </visual>\n"
        )
    return out


bc = base["com"]
parts = [
    '<?xml version="1.0"?>',
    f'<!-- generato da gen_urdf.py da {PARAMS.name} ({p["versione"]}) -->',
    '<robot name="robot_2ruote">',
    '  <link name="base_link">',
    "    <inertial>",
    f'      <origin xyz="{bc[0]:.6f} {bc[1]:.6f} {bc[2]:.6f}"/>',
    f'      <mass value="{base["massa"]:.5f}"/>',
    f"      {inertia_tag(base['inerzia_com'])}",
    "    </inertial>",
    coll.rstrip("\n"),
    visuals().rstrip("\n"),
    "  </link>",
]

Iw = wheel["inerzia_com_ruota_sx"]
for side, sign in (("sx", 1), ("dx", -1)):
    y = sign * wheel["y_sx"]
    parts += [
        f'  <link name="ruota_{side}">',
        "    <inertial>",
        '      <origin xyz="0 0 0"/>',
        f'      <mass value="{wheel["massa_con_gomma"]:.5f}"/>',
        f"      {inertia_tag(Iw)}",
        "    </inertial>",
        "    <collision>",
        '      <origin xyz="0 0 0" rpy="1.5707963 0 0"/>',
        f'      <geometry><cylinder radius="{wheel["raggio_esterno"]}" length="{wheel["larghezza"]}"/></geometry>',
        "    </collision>",
        visuals((f"ruota_{side}", f"gomma_{side}"), dy=-y).rstrip("\n"),
        "  </link>",
        f'  <joint name="giunto_ruota_{side}" type="continuous">',
        '    <parent link="base_link"/>',
        f'    <child link="ruota_{side}"/>',
        f'    <origin xyz="0 {y:.5f} 0"/>',
        '    <axis xyz="0 1 0"/>',
        '    <limit effort="1.0" velocity="60.0"/>',
        "  </joint>",
    ]
parts.append("</robot>")
OUT.write_text("\n".join(parts) + "\n")
print(f"scritto {OUT}  massa totale {base['massa'] + 2 * wheel['massa_con_gomma']:.4f} kg")
