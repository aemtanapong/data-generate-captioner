from PIL import Image
from pathlib import Path

def convert_all_gifs_to_true_gif(
    input_folder,
    output_folder=None,
    duration=500,
):

    """
    Convert all gif/webp animation files
    in folder to true GIF animation
    """

    input_folder = Path(input_folder)

    # -----------------------------------
    # OUTPUT FOLDER
    # -----------------------------------
    if output_folder is None:

        output_folder = input_folder / "converted_gif"

    output_folder = Path(output_folder)

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------
    # FIND GIF FILES
    # -----------------------------------
    gif_files = list(input_folder.glob("*.gif"))

    print(f"Found {len(gif_files)} gif files")

    # -----------------------------------
    # LOOP FILES
    # -----------------------------------
    for gif_path in gif_files:

        try:

            img = Image.open(gif_path)

            frames = []

            # -----------------------------
            # EXTRACT FRAMES
            # -----------------------------
            for i in range(img.n_frames):

                img.seek(i)

                frame = img.convert("RGB")

                frames.append(frame.copy())

            # -----------------------------
            # OUTPUT PATH
            # -----------------------------
            output_path = (
                output_folder /
                gif_path.name
            )

            # -----------------------------
            # SAVE TRUE GIF
            # -----------------------------
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0,
                format="GIF"
            )

            print(
                f"✅ Converted: {gif_path.name}"
            )

        except Exception as e:

            print(
                f"❌ Failed: {gif_path.name}"
            )

            print(e)

    print("\n🎉 Done")
convert_all_gifs_to_true_gif(
    input_folder="app/rain"
)