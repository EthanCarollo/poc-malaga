"""Geometry and dataset helpers. No model or CUDA dependency."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

RELATIONS = ("left", "right", "above", "below")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def checked_box(box):
    if len(box) != 4 or not all(math.isfinite(float(v)) and 0 <= v <= 1 for v in box):
        raise ValueError("Boîte attendue : [x0, y0, x1, y1] entre 0 et 1.")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Une boîte doit avoir une surface positive.")
    return list(map(float, box))


def layout_prompt(prompt, objects):
    """Bind ordered labels and normalized boxes using the training convention."""
    if not isinstance(prompt, str) or not isinstance(objects, list) or not 1 <= len(objects) <= 16:
        raise ValueError("Un prompt et une liste de 1 à 16 objets sont requis.")
    boxes, labels = [], []
    for obj in objects:
        if not isinstance(obj, dict) or not isinstance(obj.get("label"), str) or not obj["label"].strip():
            raise ValueError("Chaque objet doit avoir un label explicite.")
        try:
            boxes.append(checked_box(obj.get("box", [])))
        except (TypeError, ValueError) as error:
            raise ValueError("Chaque objet requiert une boîte xyxy normalisée valide.") from error
        labels.append(obj["label"].strip())
    if "normalized object boxes xyxy:" in prompt.lower():
        raise ValueError("Fournir le prompt naturel et les boîtes séparément.")
    bound = [bool(re.search(rf"(?:Object|Objet)\s+{i+1}\s*:\s*{re.escape(label)}\.", prompt, re.IGNORECASE))
             for i, label in enumerate(labels)]
    numbered = [int(n) for n in re.findall(r"(?:Object|Objet)\s+(\d+)\s*:", prompt, re.IGNORECASE)]
    text = prompt.rstrip()
    if numbered:
        if sorted(numbered) != list(range(1, len(labels)+1)) or not all(bound):
            raise ValueError("Les labels numérotés du prompt et du layout doivent correspondre.")
    else:
        text += " "+" ".join(f"Object {i+1}: {label}." for i, label in enumerate(labels))
    text += " Normalized object boxes xyxy: "+json.dumps(boxes, separators=(",", ":"))+"."
    return text, boxes


def compose(size, objects, asset_root):
    """Place existing alpha crops inside requested boxes without overlap or clipping.

    Actual visible alpha boxes are returned: aspect ratio is preserved, so an
    object's painted box can be smaller than its requested allocation.
    """
    from PIL import Image, ImageChops
    if size < 32 or not objects:
        raise ValueError("Une résolution d'au moins 32 pixels est requise.")
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    records = []
    occupied = Image.new("L", canvas.size, 0)
    for index, obj in enumerate(objects):
        slot = checked_box(obj["box"])
        path = Path(asset_root) / obj["file_name"]
        with Image.open(path) as source:
            if source.mode != "RGBA":
                raise ValueError(f"Asset RGBA requis : {path}")
            bounds = source.getchannel("A").getbbox()
            if bounds is None:
                raise ValueError(f"Asset vide : {path}")
            tile = source.crop(bounds)
        x0, y0 = math.ceil(slot[0]*size), math.ceil(slot[1]*size)
        x1, y1 = math.floor(slot[2]*size), math.floor(slot[3]*size)
        if min(x1-x0, y1-y0) < 1:
            raise ValueError("Boîte trop petite à cette résolution.")
        scale = min((x1-x0)/tile.width, (y1-y0)/tile.height)
        tile = tile.resize((max(1, int(tile.width*scale)), max(1, int(tile.height*scale))), Image.Resampling.NEAREST)
        x, y = x0+(x1-x0-tile.width)//2, y0+(y1-y0-tile.height)//2
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        layer.paste(tile, (x, y))
        alpha = layer.getchannel("A")
        if ImageChops.multiply(occupied, alpha).getbbox() is not None:
            raise ValueError("Les objets se chevauchent dans cette composition.")
        occupied = ImageChops.lighter(occupied, alpha)
        canvas = Image.alpha_composite(canvas, layer)
        actual = [v/size for v in alpha.getbbox()]
        records.append({**obj, "object_id": str(index+1), "box": actual, "allocation": slot})
    return canvas, records


def relation_holds(a, b, relation, margin=0.0):
    a, b = checked_box(a), checked_box(b)
    if relation == "left":
        return a[2]+margin <= b[0]
    if relation == "right":
        return b[2]+margin <= a[0]
    if relation == "above":
        return a[3]+margin <= b[1]
    if relation == "below":
        return b[3]+margin <= a[1]
    raise ValueError(f"Relation inconnue : {relation}")


def box_iou(a, b):
    a, b = checked_box(a), checked_box(b)
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))
    area = lambda x: (x[2]-x[0])*(x[3]-x[1])
    return intersection/(area(a)+area(b)-intersection)


def center_error(a, b):
    return math.hypot((a[0]+a[2]-b[0]-b[2])/2, (a[1]+a[3]-b[1]-b[3])/2)


def wilson(successes, count, z=1.959963984540054):
    if count < 1:
        raise ValueError("Un score sans exemple ne peut pas être validé.")
    p = successes/count
    denominator = 1+z*z/count
    center = (p+z*z/(2*count))/denominator
    radius = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return [max(0, center-radius), min(1, center+radius)]
