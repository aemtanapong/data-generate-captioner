import os
from PIL import Image

# =========================
# CONFIG
# =========================
GIF_PATH = "radar_caption/central/history/2026/20260527_020002.webp"
OUTPUT_DIR = f"radar_caption/central/history/2026/{os.path.splitext(os.path.basename(GIF_PATH))[0]}"
print(OUTPUT_DIR)
PREFIX = "n_image"

# =========================
# CREATE OUTPUT FOLDER
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD GIF
# =========================

gif = Image.open(GIF_PATH)

frame_count = gif.n_frames

print("Total frames:", frame_count)

# =========================
# EXTRACT FRAMES
# =========================
left = 279
upper = 349
right = 279 + 240
lower = 320 + 200
for frame_idx in range(frame_count):

    gif.seek(frame_idx)

    frame = gif.convert("RGB")

    filename = f"{PREFIX}{frame_idx+1:03d}.png"

    cropped_image = frame.crop((left, upper, right, lower))

    save_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    cropped_image.save(save_path)

    print("Saved:", save_path)

print("Done")