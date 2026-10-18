#!/usr/bin/env python3
"""Patch Diffusers' Qwen DreamBooth dataset for JSONL captions and lazy image loading."""

from __future__ import annotations

import argparse
from pathlib import Path


ORIGINAL = '''        else:
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.exists():
                raise ValueError("Instance images root doesn't exists.")

            instance_images = [Image.open(path) for path in list(Path(instance_data_root).iterdir())]
            self.custom_instance_prompts = None

        self.instance_images = []
'''

PATCHED = '''        else:
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.exists():
                raise ValueError("Instance images root doesn't exists.")

            metadata_path = self.instance_data_root / "metadata.jsonl"
            if metadata_path.is_file():
                metadata = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line]
                image_paths = [self.instance_data_root / entry["file_name"] for entry in metadata]
                if not image_paths:
                    raise ValueError(f"No image captions found in {metadata_path}.")
                missing_images = [str(path) for path in image_paths if not path.is_file()]
                if missing_images:
                    raise ValueError(f"Images referenced by {metadata_path} are missing: {missing_images[:5]}")
                if any(not entry.get("text") for entry in metadata):
                    raise ValueError(f"Every metadata.jsonl row must contain a non-empty 'text' caption: {metadata_path}")
                instance_images = image_paths
                self.custom_instance_prompts = []
                for entry in metadata:
                    self.custom_instance_prompts.extend(itertools.repeat(entry["text"], repeats))
            else:
                image_paths = sorted(
                    path for path in self.instance_data_root.iterdir()
                    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
                )
                instance_images = image_paths
                self.custom_instance_prompts = None

        self.instance_images = []
'''
PATCHED_PREVIOUS = PATCHED.replace(
    "instance_images = image_paths",
    "instance_images = [Image.open(path) for path in image_paths]",
)

PIXEL_VALUES_ORIGINAL = '''        self.pixel_values = []
        self.buckets = []
        bucket_to_idx = {}
        for image in self.instance_images:
            image = exif_transpose(image)
            # RGBA, not RGB: this VAE takes and returns four channels.
            if not image.mode == "RGBA":
                image = image.convert("RGBA")

            width, height = image.size

            # Assign the image to a bucket.
            target = self._bucket_for_image(height, width)
            if target not in bucket_to_idx:
                bucket_to_idx[target] = len(self.buckets)
                self.buckets.append(target)
            bucket_idx = bucket_to_idx[target]

            # based on the bucket assignment, define the transformations
            image = self.train_transform(
                image,
                size=target,
                center_crop=args.center_crop,
                random_flip=args.random_flip,
            )
            self.pixel_values.append((image, bucket_idx))
'''

PIXEL_VALUES_PATCHED = '''        self.pixel_values = []
        self.buckets = []
        bucket_to_idx = {}
        self.random_flip = args.random_flip
        for image in self.instance_images:
            if isinstance(image, (str, Path)):
                image = Path(image)
                with Image.open(image) as source:
                    width, height = exif_transpose(source).size
            else:
                image = exif_transpose(image)
                width, height = image.size

            # Assign the image to a bucket without decoding every image into a training tensor.
            target = self._bucket_for_image(height, width)
            if target not in bucket_to_idx:
                bucket_to_idx[target] = len(self.buckets)
                self.buckets.append(target)
            bucket_idx = bucket_to_idx[target]
            self.pixel_values.append((image, bucket_idx))
'''

GETITEM_ORIGINAL = '''        instance_image, bucket_idx = self.pixel_values[index % self.num_instance_images]
        example["index"] = index
        example["instance_images"] = instance_image
'''

GETITEM_PATCHED = '''        image_source, bucket_idx = self.pixel_values[index % self.num_instance_images]
        if isinstance(image_source, (str, Path)):
            with Image.open(image_source) as source:
                image = exif_transpose(source)
                if image.mode != "RGBA":
                    image = image.convert("RGBA")
                instance_image = self.train_transform(
                    image,
                    size=self.buckets[bucket_idx],
                    center_crop=self.center_crop,
                    random_flip=self.random_flip,
                )
        else:
            image = exif_transpose(image_source)
            if image.mode != "RGBA":
                image = image.convert("RGBA")
            instance_image = self.train_transform(
                image,
                size=self.buckets[bucket_idx],
                center_crop=self.center_crop,
                random_flip=self.random_flip,
            )
        example["index"] = index
        example["instance_images"] = instance_image
'''

CACHE_SAMPLER_ORIGINAL = '''        cache_batch_sampler = BucketBatchSampler(
            train_dataset, batch_size=args.train_batch_size, drop_last=False, seed=args.seed
        )
'''

CACHE_SAMPLER_PATCHED = '''        cache_batch_size = max(args.train_batch_size, 2)
        cache_batch_sampler = BucketBatchSampler(
            train_dataset, batch_size=cache_batch_size, drop_last=False, seed=args.seed
        )
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trainer", type=Path, help="Chemin du trainer Qwen-Image-2.1 de Diffusers")
    trainer_path = parser.parse_args().trainer.resolve()
    source = trainer_path.read_text(encoding="utf-8")
    changes: list[str] = []
    if PATCHED not in source:
        if source.count(PATCHED_PREVIOUS) == 1:
            source = source.replace(PATCHED_PREVIOUS, PATCHED, 1)
            changes.append("chargement paresseux des fichiers dataset")
        elif source.count(ORIGINAL) == 1:
            source = source.replace(ORIGINAL, PATCHED, 1)
            changes.append("légendes metadata.jsonl")
        else:
            raise SystemExit(
                "La version Diffusers ne correspond pas au bloc de légendes attendu ; aucun changement appliqué. "
                "Vérifie le trainer avant de mettre à jour patch_trainer.py."
            )

    if PIXEL_VALUES_PATCHED not in source:
        if source.count(PIXEL_VALUES_ORIGINAL) != 1:
            raise SystemExit(
                "La version Diffusers ne correspond pas au bloc de chargement des images attendu ; aucun changement appliqué. "
                "Vérifie le trainer avant de mettre à jour patch_trainer.py."
            )
        source = source.replace(PIXEL_VALUES_ORIGINAL, PIXEL_VALUES_PATCHED, 1)
        changes.append("chargement des images à la demande")

    if GETITEM_PATCHED not in source:
        if source.count(GETITEM_ORIGINAL) != 1:
            raise SystemExit(
                "La version Diffusers ne correspond pas au bloc __getitem__ attendu ; aucun changement appliqué. "
                "Vérifie le trainer avant de mettre à jour patch_trainer.py."
            )
        source = source.replace(GETITEM_ORIGINAL, GETITEM_PATCHED, 1)
        changes.append("transformation par lot")

    if CACHE_SAMPLER_PATCHED not in source:
        if source.count(CACHE_SAMPLER_ORIGINAL) != 1:
            raise SystemExit(
                "La version Diffusers ne correspond pas au bloc de pré-calcul attendu ; aucun changement appliqué. "
                "Vérifie le trainer avant de mettre à jour patch_trainer.py."
            )
        source = source.replace(CACHE_SAMPLER_ORIGINAL, CACHE_SAMPLER_PATCHED, 1)
        changes.append("pré-calcul par lots de deux")

    if changes:
        trainer_path.write_text(source, encoding="utf-8")
        print(f"Trainer configuré pour {', '.join(changes)} : {trainer_path}")
    else:
        print(f"Trainer déjà configuré pour les légendes et le chargement à la demande : {trainer_path}")


if __name__ == "__main__":
    main()
