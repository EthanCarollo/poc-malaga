"""PNG alpha and separate checkerboard preview regression checks."""
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from sharm_image21.image_io import checkerboard_preview, require_transparent_background, save_png


class PngExportChecks(unittest.TestCase):
    def sprite(self):
        image = Image.new("RGBA", (32, 32), (1, 2, 3, 0))
        ImageDraw.Draw(image).rectangle((10, 10, 20, 20), fill=(255, 0, 0, 255))
        image.putpixel((10, 10), (255, 0, 0, 128))
        return image

    def test_png_roundtrip_preserves_alpha_and_preview_is_separate(self):
        image = self.sprite()
        original = image.tobytes()
        with tempfile.TemporaryDirectory() as folder:
            output, preview = (Path(folder) / name for name in ("output.png", "preview.png"))
            save_png(image, output, preview_path=preview, require_transparency=True)
            with Image.open(output) as saved:
                self.assertEqual(saved.format, "PNG")
                self.assertEqual(saved.mode, "RGBA")
                self.assertEqual(saved.tobytes(), original)
            with Image.open(preview) as saved:
                self.assertEqual(saved.mode, "RGB")
                self.assertEqual(saved.getpixel((0, 0)), (232, 232, 232))
                self.assertEqual(saved.getpixel((16, 0)), (184, 184, 184))
            self.assertEqual(image.tobytes(), original)

    def test_transparency_request_rejects_rgb_opaque_and_empty_images(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "rejected.png"
            for image in (Image.new("RGB", (32, 32), "red"),
                          Image.new("RGBA", (32, 32), (255, 0, 0, 255)),
                          Image.new("RGBA", (32, 32), (0, 0, 0, 0))):
                with self.subTest(mode=image.mode, extrema=image.getextrema()):
                    with self.assertRaises(ValueError):
                        save_png(image, output, require_transparency=True)
                    self.assertFalse(output.exists())

    def test_preview_cannot_overwrite_native_png(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "native.png"
            save_png(self.sprite(), output)
            original = output.read_bytes()
            with self.assertRaises(ValueError):
                save_png(Image.new("RGB", (32, 32), "blue"), output, preview_path=output)
            self.assertEqual(output.read_bytes(), original)
            with self.assertRaises(ValueError):
                save_png(self.sprite(), Path(folder) / "wrong.jpg")

    def test_checkerboard_preserves_input_and_checks_tile_size(self):
        image = self.sprite()
        original = image.tobytes()
        self.assertGreater(require_transparent_background(image), 0.8)
        self.assertEqual(checkerboard_preview(image).mode, "RGB")
        self.assertEqual(image.tobytes(), original)
        with self.assertRaises(ValueError):
            checkerboard_preview(image, tile_size=0)


if __name__ == "__main__":
    unittest.main()
