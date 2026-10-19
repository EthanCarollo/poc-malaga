"""PNG export helpers that keep alpha intact and make it visible in previews."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def transparent_fraction(image: Image.Image, threshold: int = 16) -> float:
    """Return the share of pixels with a clearly transparent alpha value."""
    if image.mode != "RGBA":
        raise ValueError("Une image RGBA est requise pour mesurer l’alpha.")
    alpha = image.getchannel("A").histogram()
    return sum(alpha[:threshold + 1]) / (image.width * image.height)


def require_transparent_background(image: Image.Image) -> float:
    """Reject opaque/empty results when transparent output was requested."""
    fraction = transparent_fraction(image)
    alpha = image.getchannel("A").histogram()
    visible_fraction = sum(alpha[240:]) / (image.width * image.height)
    if fraction < 0.005 or visible_fraction < 0.01:
        raise ValueError(
            "Le rendu n’a pas de fond transparent exploitable "
            f"(pixels transparents: {fraction:.1%}, pixels visibles: {visible_fraction:.1%})."
        )
    return fraction


def checkerboard_preview(image: Image.Image, tile_size: int = 16) -> Image.Image:
    """Composite RGBA onto a neutral checkerboard for clear display in model cards."""
    if tile_size < 1:
        raise ValueError("La taille des cases du damier doit être positive.")
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (232, 232, 232, 255))
    draw = ImageDraw.Draw(background)
    for y in range(0, rgba.height, tile_size):
        for x in range(0, rgba.width, tile_size):
            if (x // tile_size + y // tile_size) % 2:
                draw.rectangle((x, y, min(x + tile_size - 1, rgba.width - 1),
                                min(y + tile_size - 1, rgba.height - 1)), fill=(184, 184, 184, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def save_png(image: Image.Image, path: str | Path, *, preview_path: str | Path | None = None,
             require_transparency: bool = False) -> Image.Image:
    """Save an image as RGBA PNG and optionally save a checkerboard preview."""
    if not isinstance(image, Image.Image):
        raise TypeError("L’image à enregistrer doit être une image PIL.")
    output = Path(path)
    if output.suffix.lower() != ".png":
        raise ValueError("Le fichier de sortie doit porter l’extension .png.")
    rgba = image.convert("RGBA")
    if require_transparency:
        require_transparent_background(rgba)
    preview = Path(preview_path) if preview_path is not None else None
    if preview is not None:
        if preview.suffix.lower() != ".png" or preview.resolve() == output.resolve():
            raise ValueError("L’aperçu doit être un autre fichier .png.")
    output.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(output, format="PNG")
    if preview is not None:
        preview.parent.mkdir(parents=True, exist_ok=True)
        checkerboard_preview(rgba).save(preview, format="PNG")
    return rgba
