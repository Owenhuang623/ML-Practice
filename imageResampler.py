import os
from PIL import Image
from pathlib import Path

def batch_resample_nested(input_root, output_root, target_size=(500, 500)):
    input_path = Path(input_root)
    output_path = Path(output_root)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for file_path in input_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            relative_path = file_path.relative_to(input_path)
            dest_file_path = output_path / relative_path

            dest_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                with Image.open(file_path) as img:
                    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                    img_resized.save(dest_file_path)
                    print(f"Processed: {relative_path}")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

batch_resample_nested("PokemonDataRaw", "PokemonDataResampled", target_size=(500, 500))