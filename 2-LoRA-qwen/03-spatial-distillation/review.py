#!/usr/bin/env python3
"""Build an offline annotation page with identity checks and drawable object boxes."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.predictions.read_text(encoding="utf-8"))
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    document = r'''<!doctype html><html lang="fr"><meta charset="utf-8">
<title>Revue spatiale Sharm</title><style>
body{font:16px system-ui;margin:24px;max-width:1150px;background:#f6f7f9;color:#19232d}button,input,select{font:inherit;padding:7px;margin:4px}
main{display:flex;gap:24px;flex-wrap:wrap}canvas{width:512px;max-width:100%;touch-action:none;background:repeating-conic-gradient(#bbb 0% 25%,#eee 0% 50%) 50%/24px 24px}
aside{flex:1;min-width:300px}label{display:block;margin:10px 0}pre{white-space:pre-wrap}small{display:block;color:#45515c}
</style><h1>Revue des positions</h1>
<p>Identifiez chaque objet puis tracez sa boîte visible. Laissez sans boîte un objet absent ou impossible à identifier.
Les rectangles cibles sont affichables pour comparer les positions. Chaque image doit recevoir une revue explicite.</p>
<label>Votre nom <input id="reviewer" placeholder="Identifiant du relecteur"></label>
<button id="prev">← Précédente</button><button id="next">Suivante →</button><span id="progress"></span>
<main><section><canvas id="canvas" width="512" height="512"></canvas><label><input id="targets" type="checkbox">Afficher les boîtes cibles</label></section>
<aside><p id="prompt"></p><label>Objet à annoter <select id="object"></select></label><div id="identities"></div>
<button id="remove">Effacer la boîte de cet objet</button>
<label><input id="distinct" type="checkbox">Les objets attendus sont distincts et séparés</label>
<label>Objets supplémentaires <input id="extra" type="number" min="0" value="0"></label>
<label><input id="reviewed" type="checkbox">J’ai vérifié cette image, y compris les objets manquants</label>
<small>Le numéro de méthode est masqué pour limiter le biais. Le protocole reste une revue individuelle ; utilisez plusieurs relecteurs pour estimer leur accord.</small>
</aside></main><button id="export">Exporter les annotations JSON</button><label>Reprendre un fichier <input id="import" type="file" accept="application/json"></label>
<script>const data=__DATA__;const rows=[...data.rows].sort((a,b)=>a.id.localeCompare(b.id));let index=0,start=null;
const $=id=>document.getElementById(id),annotations={},image=new Image();const ctx=$('canvas').getContext('2d');
function current(){const r=rows[index];return annotations[r.id]??={id:r.id,image_sha256:r.image_sha256,objects:[],reviewed:false,distinct:false,extra_objects:0,method:'human_identity_and_boxes',reviewer:''};}
function save(){const a=current();a.reviewer=$('reviewer').value.trim();a.distinct=$('distinct').checked;a.extra_objects=Number($('extra').value);a.reviewed=$('reviewed').checked;}
function draw(){ctx.clearRect(0,0,512,512);ctx.drawImage(image,0,0,512,512);const a=current();function box(b,color,text){ctx.strokeStyle=color;ctx.lineWidth=3;ctx.strokeRect(b[0]*512,b[1]*512,(b[2]-b[0])*512,(b[3]-b[1])*512);ctx.fillStyle=color;ctx.font='15px sans-serif';ctx.fillText(text,b[0]*512,Math.max(16,b[1]*512-3));}
if($('targets').checked)for(const o of rows[index].objects)box(o.box,'#00bbcc','cible '+o.object_id);for(const o of a.objects)box(o.box,'#ff6633','tracé '+o.object_id);}
function show(){const r=rows[index],a=current();$('progress').textContent=`${index+1}/${rows.length} — ${Object.values(annotations).filter(a=>a.reviewed).length} revues`;$('prompt').textContent=r.prompt;
$('object').replaceChildren();$('identities').replaceChildren();for(const o of r.objects){const opt=document.createElement('option');opt.value=o.object_id;opt.textContent=`${o.object_id} : ${o.label}`;$('object').append(opt);
const label=document.createElement('label'),check=document.createElement('input');check.type='checkbox';check.checked=!!a.objects.find(x=>x.object_id===o.object_id)?.identity_ok;check.onchange=()=>{const found=current().objects.find(x=>x.object_id===o.object_id);if(found)found.identity_ok=check.checked;else{check.checked=false;alert('Tracez d’abord la boîte de cet objet.');}};label.append(check,document.createTextNode(' Identité correcte : '+o.label));$('identities').append(label);}
$('distinct').checked=a.distinct;$('extra').value=a.extra_objects;$('reviewed').checked=a.reviewed;image.src=r.file_name;}
function point(e){const r=$('canvas').getBoundingClientRect();return [Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))];}
$('canvas').onpointerdown=e=>{start=point(e);$('canvas').setPointerCapture(e.pointerId);};$('canvas').onpointerup=e=>{if(!start)return;const p=point(e),box=[Math.min(start[0],p[0]),Math.min(start[1],p[1]),Math.max(start[0],p[0]),Math.max(start[1],p[1])];start=null;if(box[2]-box[0]<.005||box[3]-box[1]<.005)return;save();const a=current(),id=$('object').value;a.objects=a.objects.filter(o=>o.object_id!==id);a.objects.push({object_id:id,box,identity_ok:false});show();draw();};
image.onload=draw;$('targets').onchange=draw;$('remove').onclick=()=>{save();current().objects=current().objects.filter(o=>o.object_id!==$('object').value);show();draw();};
$('prev').onclick=()=>{save();index=Math.max(0,index-1);show();};$('next').onclick=()=>{save();index=Math.min(rows.length-1,index+1);show();};
$('export').onclick=()=>{save();if(!$('reviewer').value.trim()){alert('Renseignez votre identifiant.');return;}const blob=new Blob([JSON.stringify({prediction_signature:data.signature,rows:Object.values(annotations)},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='annotations.json';a.click();URL.revokeObjectURL(a.href);};
$('import').onchange=async e=>{const restored=JSON.parse(await e.target.files[0].text());if(restored.prediction_signature!==data.signature){alert('Ce fichier concerne d’autres générations.');return;}for(const a of restored.rows)annotations[a.id]=a;$('reviewer').value=restored.rows.find(a=>a.reviewer)?.reviewer??'';show();};show();</script></html>'''
    path = args.predictions.parent / "review.html"
    path.write_text(document.replace("__DATA__", payload), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
