"""Crea viewer.html: animazione 3D del robot (mesh STL del CAD + registrazione di registra.py).

Il file e' autonomo (dati incorporati); serve solo la libreria three.js da internet.
Uso: env_isaac\\Scripts\\python.exe robot_sim\\crea_viewer.py [animazione.json] [viewer.html]
"""
import base64
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).parent
ANIM = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "animazione.json"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "viewer.html"
AX = 0.0325
Y_RUOTA = 0.088

COLORI = [
    ("gomma", "#141414"), ("ruota", "#f27319"), ("motore", "#8c949e"), ("cella", "#1a59bf"),
    ("esp32", "#0d5926"), ("cam_", "#0d5926"), ("driver", "#b31a1a"), ("step_down", "#1a1a99"),
    ("IMU", "#1a6666"), ("ToF", "#333333"),
]


def colore(nome):
    for k, c in COLORI:
        if nome.startswith(k):
            return c
    return "#d9d9d9"


def leggi_stl(path):
    d = path.read_bytes()
    n = struct.unpack("<I", d[80:84])[0]
    v = []
    for i in range(n):
        o = 84 + i * 50
        f = struct.unpack("<12f", d[o:o + 48])
        v += list(f[3:12])
    return v


def pezzo(path, dy):
    v = leggi_stl(path)
    # mm, frame mondo con origine a terra -> m, frame del link (origine sull'asse, spostata di dy in y)
    out = []
    for i in range(0, len(v), 3):
        out += [v[i] * 0.001, v[i + 1] * 0.001 - dy, v[i + 2] * 0.001 - AX]
    import array
    a = array.array("f", out)
    return {"nome": path.stem, "colore": colore(path.stem), "b64": base64.b64encode(a.tobytes()).decode()}


base, ruota = [], {"sx": [], "dx": []}
for stl in sorted((HERE / "meshes").glob("*.stl")):
    n = stl.stem
    if n.startswith(("ruota_", "gomma_")):
        lato = n.split("_")[1]
        ruota[lato].append(pezzo(stl, Y_RUOTA if lato == "sx" else -Y_RUOTA))
    else:
        base.append(pezzo(stl, 0.0))

anim = json.loads(ANIM.read_text())
dati = {"base": base, "ruota": ruota, "anim": anim, "yr": Y_RUOTA}

HTML = r"""<!doctype html>
<html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robot a due ruote - simulazione</title>
<style>
 html,body{margin:0;height:100%;background:#1b1e23;color:#e8e8e8;font-family:system-ui,sans-serif;overflow:hidden}
 #c{display:block;width:100%;height:100%}
 #ui{position:fixed;left:12px;right:12px;bottom:12px;display:flex;gap:12px;align-items:center;background:#0008;padding:8px 12px;border-radius:8px;flex-wrap:wrap}
 button{background:#3a6df0;color:#fff;border:0;border-radius:6px;padding:6px 14px;font-size:14px;cursor:pointer}
 #t{flex:1;min-width:140px} #info{font-variant-numeric:tabular-nums;font-size:14px}
 #h{position:fixed;top:10px;left:12px;font-size:13px;opacity:.8}
</style></head><body>
<canvas id="c"></canvas>
<div id="h">Trascina = ruota la vista &middot; rotella = zoom &middot; Maiusc+trascina = sposta</div>
<div id="ui"><button id="pp">Pausa</button><button id="ri">Riavvia</button>
<input id="t" type="range" min="0" max="0" value="0"><span id="info"></span>
<label>velocit&agrave; <select id="v"><option value="1">1x</option><option value="0.5">0,5x</option><option value="0.25">0,25x</option></select></label></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const D = __DATI__;
function geo(b64){const bin=atob(b64),buf=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)buf[i]=bin.charCodeAt(i);
 const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(new Float32Array(buf.buffer),3));g.computeVertexNormals();return g;}
function mesh(p){return new THREE.Mesh(geo(p.b64),new THREE.MeshStandardMaterial({color:p.colore,flatShading:true,roughness:.6,metalness:.1}));}
const canvas=document.getElementById('c'),R=new THREE.WebGLRenderer({canvas,antialias:true});
const S=new THREE.Scene();S.background=new THREE.Color(0x1b1e23);
const cam=new THREE.PerspectiveCamera(40,1,0.01,50);cam.up.set(0,0,1);
S.add(new THREE.HemisphereLight(0xffffff,0x444455,1.1));
const L=new THREE.DirectionalLight(0xffffff,0.9);L.position.set(1,-1.5,2);S.add(L);
const grid=new THREE.GridHelper(2,40,0x555a66,0x33373f);grid.rotation.x=Math.PI/2;S.add(grid);
const robot=new THREE.Group();S.add(robot);
D.base.forEach(p=>robot.add(mesh(p)));
const gr={};
['sx','dx'].forEach(l=>{const g=new THREE.Group();g.position.set(0,l==='sx'?D.yr:-D.yr,0);D.ruota[l].forEach(p=>g.add(mesh(p)));robot.add(g);gr[l]=g;});
const idx={sx:D.anim.giunti.indexOf('giunto_ruota_sx'),dx:D.anim.giunti.indexOf('giunto_ruota_dx')};
let th=1.0,ph=0.9,dist=0.7,tgt=new THREE.Vector3(0,0,0.06);
function setCam(){cam.position.set(tgt.x+dist*Math.cos(ph)*Math.cos(th),tgt.y+dist*Math.cos(ph)*Math.sin(th)*-1,tgt.z+dist*Math.sin(ph));cam.lookAt(tgt);}
let drag=null;canvas.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,s:e.shiftKey};canvas.setPointerCapture(e.pointerId)};
canvas.onpointerup=()=>drag=null;
canvas.onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;
 if(drag.s){tgt.x-=dx*0.0015*Math.sin(th)*-1;tgt.y-=dx*0.0015*Math.cos(th);tgt.z+=dy*0.0015}else{th-=dx*0.008;ph=Math.max(-0.2,Math.min(1.5,ph+dy*0.006));}};
canvas.onwheel=e=>{e.preventDefault();dist=Math.max(0.2,Math.min(3,dist*Math.exp(e.deltaY*0.001)));};
const F=D.anim.frames,fps=D.anim.fps,sl=document.getElementById('t'),info=document.getElementById('info');
sl.max=F.length-1;let f=0,play=true,acc=0,last=performance.now();
function pose(i){const fr=F[i];robot.position.set(fr.p[0],fr.p[1],fr.p[2]);robot.quaternion.set(fr.q[1],fr.q[2],fr.q[3],fr.q[0]);
 gr.sx.rotation.y=fr.g[idx.sx];gr.dx.rotation.y=fr.g[idx.dx];
 const [w,x,y,z]=fr.q,pitch=Math.asin(Math.max(-1,Math.min(1,2*(w*y-z*x))))*180/Math.PI;
 info.textContent='t = '+(i/fps).toFixed(2)+' s   inclinazione = '+pitch.toFixed(1)+'° (+ = in avanti)';sl.value=i;}
document.getElementById('pp').onclick=e=>{play=!play;e.target.textContent=play?'Pausa':'Riprendi'};
document.getElementById('ri').onclick=()=>{f=0;acc=0;};
sl.oninput=()=>{f=+sl.value;play=false;document.getElementById('pp').textContent='Riprendi'};
function loop(now){const dt=(now-last)/1000;last=now;if(play){acc+=dt*fps*+document.getElementById('v').value;
 while(acc>=1){acc-=1;f++;if(f>=F.length){f=F.length-1;}}}
 pose(f);const w=innerWidth,h=innerHeight;if(canvas.width!==w||canvas.height!==h){R.setSize(w,h,false);cam.aspect=w/h;cam.updateProjectionMatrix();}
 setCam();R.render(S,cam);requestAnimationFrame(loop);}
requestAnimationFrame(loop);
</script></body></html>"""

OUT.write_text(HTML.replace("__DATI__", json.dumps(dati)), encoding="utf-8")
print(f"scritto {OUT} ({OUT.stat().st_size / 1e6:.2f} MB), {len(anim['frames'])} frame")
