import subprocess
import uuid
from pathlib import Path

from seg.segtext import apply_texture

ROOT = Path(__file__).resolve().parent.parent

GENERATED_DIR = ROOT / "server/generated"
FINAL_DIR = ROOT / "server/final"


def generate_pipeline(prompt, texture_path):

    uid = str(uuid.uuid4())

    generated_image = GENERATED_DIR / f"{uid}.png"
    final_image = FINAL_DIR / f"{uid}.png"

    # ---------------------------------
    # STEP 1: RUN STUDIOGAN
    # ---------------------------------

    subprocess.run([
        "python",
        "gan/generate.py",
        "--prompt",
        prompt,
        "--output",
        str(generated_image)
    ], check=True)

    # ---------------------------------
    # STEP 2: APPLY TEXTURE
    # ---------------------------------

    apply_texture(
        clothing_image=str(generated_image),
        texture_image=str(texture_path),
        output_path=str(final_image)
    )

    return str(final_image)