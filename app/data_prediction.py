import cv2
import numpy as np
import imageio.v2 as imageio
import geopandas as gpd
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt
from rasterio.transform import from_bounds
import rasterio.features # Required for calculate_district_metrics
import pandas as pd # Import pandas for DataFrame creation
import matplotlib.font_manager as fm
import os
from matplotlib.patches import Patch
import math
import rasterio.features
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

GIF_PATH = "example/n006.gif"
target_colors = np.array([
    [252, 252, 255], # 66.5
    [252, 219, 255], # 64.0
    [252, 202, 255], # 61.5
    [252, 139, 255], # 59.0
    [252,   0, 255], # 56.5
    [195,   0,  85], # 54.0
    [216,   0,  71], # 51.5
    [224,   0,  85], # 49.0
    [238,   0,   0], # 46.5
    [252,  75,   0], # 44.0
    [222, 152,   0], # 41.5
    [230, 164,   0], # 39.0
    [252, 214,   0], # 36.5
    [216, 216,   0], # 34.0
    [234, 218,   0], # 31.5
    [238, 252,   0], # 29.0
    [  0, 243,   0], # 26.5
    [  0, 236,   0], # 24.0
    [  0, 214,  82], # 21.5
    [  0, 200,   0], # 19.0
    [  0, 197,   0], # 16.5
    [  0, 191,   0], # 14.0
    [  0, 176,   0], # 11.5
    [  0, 168,   0],  # 10.0
    [0,0,255]
])
values = np.array([
    66.5, 64.0, 61.5, 59.0, 56.5, 54.0, 51.5, 49.0, 46.5,
    44.0, 41.5, 39.0, 36.5, 34.0, 31.5, 29.0, 26.5, 24.0,
    21.5, 19.0, 16.5, 14.0, 11.5, 10.0, 9.5
])
radar_x = 699558.0797  # พิกัด UTM X ของจุดกึ่งกลาง (ใส่ค่าของคุณ)
radar_y = 1530232.3207 # พิกัด UTM Y ของจุดกึ่งกลาง (ใส่ค่าของคุณ)
pixel_resolution = 300     # 1 พิกเซล = กี่เมตร (ตรวจสอบค่านี้อีกครั้ง)
shapefile_path = 'mapdata/Export_Output.shp' # ชื่อไฟล์ Shapefile ของคุณ
threshold = 60
import io
# import IPython.display as pyn
import imageio.v2 as imageio

def create_in_memory_gif(frames_4d_array, fps=10):
    """
    Creates an animated GIF from a 4D NumPy array (num_frames, height, width, channels)
    and returns it as bytes in memory.

    Args:
        frames_4d_array (np.ndarray): A 4D NumPy array where the first dimension is the number of frames,
                                    and the last dimension is 3 (RGB) or 4 (RGBA).
        fps (int, optional): Frames per second for the GIF. Defaults to 10.

    Returns:
        bytes: The bytes content of the generated GIF, or None if an error occurred.
    """
    if frames_4d_array.ndim != 4:
        print(f"Error: Input array must be 4D (num_frames, height, width, channels). Got {frames_4d_array.ndim}D.")
        return None

    num_channels = frames_4d_array.shape[-1]
    frames_to_save = []

    if num_channels == 3: # RGB, convert to RGBA with opaque alpha
        for frame_rgb in frames_4d_array:
            h, w, _ = frame_rgb.shape
            frame_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            frame_rgba[:, :, :3] = frame_rgb
            frame_rgba[:, :, 3] = 255  # Fully opaque alpha
            frames_to_save.append(frame_rgba)
        print("Converting 3-channel RGB frames to 4-channel RGBA (opaque) for GIF creation.")
    elif num_channels == 4: # Already RGBA
        frames_to_save = frames_4d_array
    else:
        print(f"Error: Last dimension of input array must be 3 (RGB) or 4 (RGBA). Got {num_channels}.")
        return None

    # Use BytesIO to create an in-memory file-like object
    gif_buffer = io.BytesIO()
    try:
        # Ensure all frames are uint8 and close any open figures before saving.
        # Explicitly set the format to 'GIF' when using imageio.mimsave with BytesIO.
        imageio.mimsave(gif_buffer, [np.asarray(f, dtype=np.uint8) for f in frames_to_save], format='GIF', fps=fps, loop=0)
    except Exception as e:
        print(f"Error creating GIF: {e}")
        return None

    gif_buffer.seek(0) # Rewind the buffer to the beginning
    return gif_buffer.getvalue()

print("create_in_memory_gif function defined.")
def calculate_district_metrics(radar_frame_rgba, district_gdf, radar_extent, target_colors, values, threshold=60):
    """
    Calculates both radar coverage percentage and a 'score' for each district
    based on the radar color legend values within a single frame.

    Args:
        radar_frame_rgba (np.ndarray): A NumPy array (H, W, 4) representing one RGBA radar frame.
        district_gdf (gpd.GeoDataFrame): GeoDataFrame containing district polygons.
        radar_extent (list): [left, right, bottom, top] geographical extent of the radar_frame_rgba.
        target_colors (np.ndarray): Nx3 NumPy array of target RGB colors from the radar legend.
        values (np.ndarray): N-length NumPy array of values corresponding to target_colors.
        threshold (int): Color tolerance for matching radar pixels to target_colors.

    Returns:
        tuple: A tuple containing two dictionaries:
            - coverage_percentages (dict): Keys are district names, values are coverage percentages.
            - district_scores (dict): Keys are district names, values are average scores.
    """
    height, width, _ = radar_frame_rgba.shape
    transform = from_bounds(radar_extent[0], radar_extent[2], radar_extent[1], radar_extent[3], width, height)

    coverage_percentages = {}
    district_scores = {}

    radar_rgb = radar_frame_rgba[:, :, :3]
    alpha_channel = radar_frame_rgba[:, :, 3]

    for idx, row in district_gdf.iterrows():
        district_geometry = row['geometry']
        district_name = row['ADM3_EN'] if 'ADM3_EN' in row else row['DISTRICT_T'] # Use 'ADM3_EN' or fallback

        district_mask = rasterio.features.rasterize(
            [(district_geometry, 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        ).astype(bool)

        # --- Coverage Calculation ---
        # Identify radar pixels (where alpha > 0) within the district mask
        radar_pixels_in_district = np.sum(district_mask & (alpha_channel > 0))
        total_district_pixels = np.sum(district_mask)

        percentage = 0.0
        if total_district_pixels > 0:
            percentage = (radar_pixels_in_district / total_district_pixels) * 100
        coverage_percentages[district_name] = percentage

        # --- Score Calculation ---
        # Combine district mask with non-transparent radar pixels
        relevant_radar_pixels_mask = district_mask & (alpha_channel > 0)
        pixels_in_district = radar_rgb[relevant_radar_pixels_mask]
#         print(pixels_in_district)
        if len(pixels_in_district) == 0:
            district_scores[district_name] = 0.0
            continue

        distances = np.linalg.norm(pixels_in_district[:, np.newaxis] - target_colors[np.newaxis, :], axis=2)
        min_dist_indices = np.argmin(distances, axis=1)
        min_distances = distances[np.arange(len(pixels_in_district)), min_dist_indices]

        pixel_values = np.where(min_distances < threshold, values[min_dist_indices], 0.0)
#         print(pixel_values)
        if np.sum(pixel_values > 0) > 0:
            district_scores[district_name] = np.mean(pixel_values[pixel_values > 0])
        else:
            district_scores[district_name] = 0.0

    return coverage_percentages, district_scores
def min_pooling(img, ksize=3):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(img, kernel)
def max_pooling(img, ksize=3):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(img, kernel)
def extract_radar_frame(img, threshold = 15):
  # reshape image
  h, w, _ = img.shape
  pixels = img.reshape(-1,3)

  mask = np.zeros(len(pixels), dtype=bool)

  for color in target_colors:
      dist = np.linalg.norm(pixels - color, axis=1)
      mask |= dist < threshold

  mask = mask.reshape(h,w)

  result = img.copy()
  result[~mask] = 255
  return result
# Assuming min_pooling, max_pooling, extract_radar_frame, calculate_district_metrics
# are already defined in previous cells or accessible in the global scope.
def get_rain_data(score_dict, n_frame):

    max_score = sum(range(1, n_frame + 1))

    result = {
        "🚨 ฝนต่อเนื่อง": [],
        "🌧️ ฝนเริ่มเคลื่อนเข้า": [],
        "👀 เฝ้าระวัง": [],
        "☀️ ไม่มีฝน": []
    }

    for district, score in score_dict.items():

        normalized_score = score / max_score

        if normalized_score >= 0.80:

            result["🚨 ฝนต่อเนื่อง"].append(district)

        elif normalized_score >= 0.40:

            result["🌧️ ฝนเริ่มเคลื่อนเข้า"].append(district)

        elif normalized_score > 0:

            result["👀 เฝ้าระวัง"].append(district)

        else:

            result["☀️ ไม่มีฝน"].append(district)

    return result
def process_radar_animation_and_extract_district_values(
    gif_path,
    radar_x,
    radar_y,
    pixel_resolution,
    shapefile_path,
    target_colors,
    values,
    threshold=60, # Default from existing notebook
    draw_white_boxes=True # Option to disable the hardcoded box drawing if needed
):
    """
    Processes a radar GIF animation, extracts radar values (coverage and score)
    for each district per frame, and returns aggregated results as a Pandas DataFrame.

    Args:
        gif_path (str): Path to the input radar GIF animation.
        radar_x (float): UTM X coordinate of the radar's center.
        radar_y (float): UTM Y coordinate of the radar's center.
        pixel_resolution (int): Meters per pixel.
        shapefile_path (str): Path to the GeoJSON or Shapefile containing district polygons.
        target_colors (np.ndarray): Nx3 NumPy array of target RGB colors from the radar legend.
        values (np.ndarray): N-length NumPy array of values corresponding to target_colors.
        threshold (int, optional): Color tolerance for matching radar pixels to target_colors. Defaults to 60.
        draw_white_boxes (bool, optional): Whether to draw white boxes on frames (as per notebook logic). Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing 'Frame', 'District', 'Coverage', and 'Score' for each district and frame.
    """
    print(f"Starting to process radar animation from {gif_path}...")

    # --- 1. Read GIF frames and preprocess ---
    frames_rgb_processed = []
    img_width, img_height = 0, 0
    try:
        with Image.open(gif_path) as im:
            img_width, img_height = im.size
            for frame_idx in range(im.n_frames):
                im.seek(frame_idx)
                # Convert each frame to RGB and Numpy Array
                frame_array = np.array(im.convert('RGB'))

                if draw_white_boxes:
                    # Apply hardcoded white box drawing as seen in iTnYQGi2qYq9
                    # These coordinates might need to be dynamic or adjusted based on actual image size
                    cv2.rectangle(frame_array, (0, 0), (100, 970), (255, 255, 255), -1)
                    cv2.rectangle(frame_array, (700, 600), (1300, 970), (255, 255, 255), -1)

                # Process with extract_radar_frame and pooling operations
                processed_frame = extract_radar_frame(frame_array)
                # Set white background (255,255,255) to black (0,0,0) before pooling
                processed_frame[np.all(processed_frame == 255, axis=-1)] = 0

                # Apply pooling operations as seen in iTnYQGi2qYq9
                processed_frame = max_pooling(processed_frame, 3)
                processed_frame = min_pooling(processed_frame, 3)
                processed_frame = min_pooling(processed_frame, 3)
                processed_frame = max_pooling(processed_frame, 10)

                frames_rgb_processed.append(processed_frame)
    except FileNotFoundError:
        print(f"Error: GIF file not found at {gif_path}")
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        print(f"Error reading or processing GIF: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

    if not frames_rgb_processed:
        print("No frames were processed.")
        return pd.DataFrame() # Return empty DataFrame if no frames

    print(f"Successfully processed {len(frames_rgb_processed)} frames.")

    # --- 2. Calculate image extent ---
    # These calculations depend on img_width and img_height obtained from the GIF
    left = radar_x - (img_width / 2 * pixel_resolution)
    right = radar_x + (img_width / 2 * pixel_resolution)
    bottom = radar_y - (img_height / 2 * pixel_resolution)
    top = radar_y + (img_height / 2 * pixel_resolution)
    img_extent = [left, right, bottom, top]
    print(f"Calculated image extent: {img_extent}")

    # --- 3. Load Shapefile ---
    try:
        gdf = gpd.read_file(shapefile_path)
        print(f"Loaded shapefile with {len(gdf)} districts.")
    except FileNotFoundError:
        print(f"Error: Shapefile not found at {shapefile_path}")
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        print(f"Error loading shapefile: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

    # --- 4. Process each frame for district metrics ---
    all_frames_coverage = []
    all_frames_district_scores = []

    print("Calculating radar coverage and scores for each district per frame...")
    for i, processed_rgb_frame in enumerate(frames_rgb_processed):

        h, w, _ = processed_rgb_frame.shape
        frame_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        frame_rgba[:, :, :3] = processed_rgb_frame

        # Set alpha to 0 for black pixels (non-radar) and 255 for other colors (radar)
        black_pixels_mask = np.all(processed_rgb_frame == [0, 0, 0], axis=-1)
        frame_rgba[~black_pixels_mask, 3] = 255 # Opaque for radar pixels
        frame_rgba[black_pixels_mask, 3] = 0   # Transparent for black background pixels

        # Call the existing calculate_district_metrics function
        frame_coverage, frame_scores = calculate_district_metrics(
            frame_rgba, gdf, img_extent, target_colors, values, threshold
        )
        all_frames_coverage.append(frame_coverage)
        all_frames_district_scores.append(frame_scores)
        if (i + 1) % 10 == 0 or (i + 1) == len(frames_rgb_processed):
            print(f"  Processed frame {i+1}/{len(frames_rgb_processed)}")

    print("Finished calculating district metrics for all frames.")

    # --- 5. Convert results to DataFrame ---
    df_rows = []
    for i in range(len(all_frames_coverage)):
        frame_num = i + 1 # 1-based frame indexing
        frame_coverage = all_frames_coverage[i]
        frame_scores = all_frames_district_scores[i]
        for district_name in frame_coverage.keys():
            df_rows.append({
                'Frame': frame_num,
                'District': district_name,
                'Coverage': float(frame_coverage.get(district_name, 0.0)),
                'Score': float(frame_scores.get(district_name, 0.0))
            })

    return pd.DataFrame(df_rows)
def get_data(GIF_PATH, degree_value, update = None):
    gif_file = io.BytesIO(GIF_PATH)
    if update : update("ประมวลภาพ radar",0.1)
    df_radar_metrics = process_radar_animation_and_extract_district_values(
        gif_path=gif_file,
        radar_x=radar_x,
        radar_y=radar_y,
        pixel_resolution=pixel_resolution,
        shapefile_path=shapefile_path,
        target_colors=target_colors,
        values=values
    )

    frames = []

    average_map_frame = []
    # -----------------------------
    # READ GIF
    # -----------------------------

    if update : update("อ่านภาพ radar",0.3)
    with Image.open(gif_file) as img:
        for frame in range(img.n_frames):
            img.seek(frame)

            # Convert to RGB
            rgba_frame = img.convert('RGB')

            # Convert to numpy
            frame_array = np.array(rgba_frame)

            # -----------------------------
            # DRAW WHITE BOX
            # -----------------------------
            cv2.rectangle(frame_array, (0, 0), (100, 970), (255, 255, 255), -1)
            cv2.rectangle(frame_array, (700, 600), (1300, 970), (255, 255, 255), -1)


            # frame_array = max_pooling(frame_array, ksize=5)
            # If you want box BEFORE processing
            processed = extract_radar_frame(frame_array)
            processed[np.all(processed == 255, axis=-1)] = 0

            processed = max_pooling(processed, 3)
            processed = min_pooling(processed, 3)
            processed = min_pooling(processed, 3)
            processed = max_pooling(processed, 10)
            average_map_frame.append(processed)
            frames.append(processed)
    # These should be globally available from previous cells but are re-assigned for clarity/robustness
    # radar_x, radar_y, pixel_resolution are from 1zPY95e5iOYT
    # gif_path, shapefile_path are from 1zPY95e5iOYT

    # Load gdf if not already available or ensure it's in scope
    if 'gdf' not in globals() or gdf is None:
        print("Loading gdf globally...")
        gdf = gpd.read_file(shapefile_path)
    else:
        print("gdf is already loaded globally.")

    # Determine img_width and img_height from the processed frames (available from iTnYQGi2qYq9)
    if 'frames' in globals() and len(frames) > 0:
        img_height, img_width, _ = frames[0].shape
        print(f"img_width: {img_width}, img_height: {img_height} derived from 'frames' variable.")
    else:
        # Fallback if 'frames' is not available or empty (should not happen if iTnYQGi2qYq9 ran)
        print("Warning: 'frames' variable not found or empty. Reading GIF directly for dimensions.")
        with Image.open(gif_file) as im:
            img_width, img_height = im.size
        print(f"img_width: {img_width}, img_height: {img_height} derived from original GIF.")

    # Calculate img_extent using global variables
    left = radar_x - (img_width / 2 * pixel_resolution)
    right = radar_x + (img_width / 2 * pixel_resolution)
    bottom = radar_y - (img_height / 2 * pixel_resolution)
    top = radar_y + (img_height / 2 * pixel_resolution)
    img_extent = [left, right, bottom, top]




    print(f"img_extent calculated globally: {img_extent}")




    # Assuming 'frames' (processed radar images), 'gdf' (GeoDataFrame), and 'img_extent' are already defined.

    # Combine all district geometries into a single MultiPolygon representing the entire area
    all_geometries = [geom for geom in gdf.geometry]
    entire_shapefile_geometry = unary_union(all_geometries)

    # Prepare to store results
    overall_coverage_per_frame = []

    print("Calculating overall radar coverage percentage for each frame...")
    if update : update("🛰️ กำลังอ่านข้อมูลภาพเรดาร์...",0.5)
    for i, processed_rgb_frame in enumerate(frames):
        # Convert the RGB frame to RGBA, making black pixels transparent
        h, w, _ = processed_rgb_frame.shape
        radar_frame_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        radar_frame_rgba[:, :, :3] = processed_rgb_frame

        # Identify black pixels (background) and set their alpha to 0 (transparent)
        # Other pixels (radar data) will have alpha set to 255 (opaque)
        black_pixels_mask = np.all(processed_rgb_frame == [0, 0, 0], axis=-1)
        radar_frame_rgba[~black_pixels_mask, 3] = 255
        radar_frame_rgba[black_pixels_mask, 3] = 0

        # Get alpha channel for radar data identification
        alpha_channel = radar_frame_rgba[:, :, 3]

        # Calculate the transform for rasterization
        left, right, bottom, top = img_extent
        transform = rasterio.transform.from_bounds(left, bottom, right, top, w, h)

        # Create a single mask for the entire shapefile
        shapefile_mask = rasterio.features.rasterize(
            [(entire_shapefile_geometry, 1)],
            out_shape=(h, w),
            transform=transform,
            fill=0,
            dtype=np.uint8
        ).astype(bool)

        # Identify radar pixels (where alpha > 0) within the entire shapefile mask
        radar_pixels_in_shapefile = np.sum(shapefile_mask & (alpha_channel > 0))
        total_shapefile_pixels = np.sum(shapefile_mask)

        overall_percentage = 0.0
        if total_shapefile_pixels > 0:
            overall_percentage = (radar_pixels_in_shapefile / total_shapefile_pixels) * 100

        overall_coverage_per_frame.append(overall_percentage)
        print(f"Frame {i+1}: Overall Radar Coverage = {overall_percentage:.2f}%")

    print("\nOverall Radar Coverage for each frame:")
    coverage_percentage = []
    for i, percentage in enumerate(overall_coverage_per_frame):
        print(f"  Frame {i+1}: {percentage:.2f}%")
        coverage_percentage.append(percentage)




    # --- Font Configuration for Thai Characters ---
    # Install Thai fonts if not already installed (for Colab environment)

    # Find a Thai font available on the system
    # A common choice is 'TH Sarabun New' or 'Garuda'
    # You can inspect available fonts with `fm.findSystemFonts(fontpaths=None, fontext='ttf')`

    font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    thai_font_path = None
    for font_path in font_paths:
        if 'Sarabun' in font_path or 'Garuda' in font_path or 'Laksaman' in font_path: # Look for common Thai fonts
            thai_font_path = font_path
            break
    
    if thai_font_path:
        fm.fontManager.addfont(thai_font_path)
        plt.rcParams['font.family'] = fm.FontProperties(fname=thai_font_path).get_name()
        plt.rcParams['axes.unicode_minus'] = False # Fix minus sign if using unicode font
        print(f"Using Thai font: {plt.rcParams['font.family']} from {thai_font_path}")
    else:
        print("Warning: No suitable Thai font found. Thai characters might not render correctly.")
    # --- End Font Configuration ---


    print(f"Generating plots for {len(frames)} radar frames...")
    if update : update("🌧️ กำลังคำนวณและวิเคราะห์ข้อมูลฝนจาก Radar...",0.6)
    # Ensure district_col is defined (e.g., 'ADM3_EN' or 'DISTRICT_T')
    # This variable might be defined globally, but it's safer to ensure it here.
    district_col = 'ADM3_EN' if 'ADM3_EN' in gdf.columns else 'DISTRICT_T'

    all_raining_districts_per_frame = [] # New list to store raining districts for each frame
    plot_frames = [] # List to store individual plot images for GIF

    for i, frame_rgb in enumerate(frames):
        fig, ax = plt.subplots(figsize=(12, 10))

        # --- Start of Highlight each district logic ---
        # Get districts with rain in the current frame from df_radar_metrics
        # df_radar_metrics should be available from previous executions
        current_frame_df = df_radar_metrics[df_radar_metrics['Frame'] == (i + 1)]
        raining_districts_in_frame = current_frame_df[current_frame_df['Coverage'] > 0]['District'].tolist()
        all_raining_districts_per_frame.append(f"Frame {i+1}: {raining_districts_in_frame}") # Store for later printing

        # Separate GeoDataFrame into districts with and without radar coverage in the current frame
        districts_with_radar = gdf[gdf[district_col].isin(raining_districts_in_frame)]
        districts_without_radar = gdf[~gdf[district_col].isin(raining_districts_in_frame)]

        # Plot districts without radar coverage (default style)
        districts_without_radar.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=1)
        # Plot districts with radar coverage (highlighted style: yellow fill, black border)
        districts_with_radar.plot(ax=ax, edgecolor='black', facecolor='yellow', linewidth=2, alpha=0.5)

        # --- End of Highlight each district logic ---

        # Convert the processed RGB radar frame to RGBA for transparency
        # Black pixels (0,0,0) will be made transparent
        h, w, _ = frame_rgb.shape
        frame_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        frame_rgba[:, :, :3] = frame_rgb  # Copy RGB channels

        # Set alpha to 0 for black pixels (background) and 255 for radar pixels
        # Using np.all for exact black match, you might adjust this tolerance if needed
        black_pixels_mask = np.all(frame_rgb == [0, 0, 0], axis=-1)
        frame_rgba[~black_pixels_mask, 3] = 255 # Opaque for radar data
        frame_rgba[black_pixels_mask, 3] = 0   # Transparent for background

        # Overlay the radar image with the calculated extent and general alpha
        # The 'alpha' parameter here applies an overall transparency to the entire image
        # In addition to the pixel-level transparency set in frame_rgba's alpha channel.
        # You can adjust the overall_alpha value below.
        overall_alpha = 0.8 # Adjust this value (0.0 to 1.0) for overall radar transparency
        ax.imshow(frame_rgba, extent=img_extent, alpha=overall_alpha)

        # Plot the radar station
        ax.scatter(radar_x, radar_y, color='blue', marker='+', s=200)

        # --- Add annotations for district names (similar to 7PMwN6uMnqni) ---
        # For better readability, annotations are commented out unless specifically requested.
        # for idx, row in gdf.iterrows():
        #     district_name = row[district_col]
        #     # Get centroid coordinates for annotation
        #     x_centroid, y_centroid = row.geometry.centroid.x, row.geometry.centroid.y

        #     # Determine annotation style based on whether the district has rain in this frame
        #     if district_name in raining_districts_in_frame:
        #         bbox_style = dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.7) # Highlighted
        #         text_color = 'black'

        #         ax.annotate(district_name, xy=(x_centroid, y_centroid), xytext=(3, 3), textcoords="offset points",
        #                     ha='center', va='center', fontsize=8, color=text_color,
        #                     bbox=bbox_style)
        # --- End of annotations for district names ---

        ax.set_title(f'Radar Frame {i+1} on Map')
        ax.set_xlabel('UTM X Coordinate')
        ax.set_ylabel('UTM Y Coordinate')
        ax.grid(True, alpha=0.3)

        # Create custom legend handles
        legend_handles = [
            Patch(facecolor='yellow', edgecolor='black', linewidth=2, alpha=0.5, label='Districts with Radar Coverage'),
            plt.Line2D([0], [0], marker='+', color='blue', markersize=10, linestyle='None', label='Radar Station')
        ]
        ax.legend(handles=legend_handles)
        plt.tight_layout()

        # Capture the plot as an image and close the figure to save memory
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plot_frames.append(image_from_plot)
        plt.close(fig) # Close the figure to prevent it from displaying and to free up memory
    if update : update("🖼️ กำลังสร้างภาพเรดาร์ทั้งหมดกำลังโหลดข้อมูล...",0.7)
    print("Finished generating all radar plots.")

    # Create and display the in-memory GIF
    if plot_frames:
        print("Creating in-memory GIF...")
        in_memory_gif_bytes = create_in_memory_gif(np.array(plot_frames), fps=5)
        # if in_memory_gif_bytes:
        #     pyn.display(pyn.Image(data=in_memory_gif_bytes))
        # else:
        #     print("Failed to create in-memory GIF.")
    else:
        print("No plot frames were generated.")

    print("\nDistricts with radar coverage per frame:")
    for item in all_raining_districts_per_frame:
        print(item)
    
    # max
    max_coverage_value = max(coverage_percentage)

    # average
    avg_coverage_value = np.mean(coverage_percentage)

    # 50 percentile (median)
    coverage_p50 = np.percentile(coverage_percentage, 50)


    # Ensure 'frames' list of processed RGB radar images is available
    # Ensure 'pixel_resolution' (meters per pixel) is available
    # Ensure 'img_width', 'img_height', 'img_extent' are available
    
    if len(frames) < 2:
        print("Need at least two frames to calculate speed and direction for nowcasting.")
    else:
        frame_prev_rgb = frames[-2]
        frame_curr_rgb = frames[-1]

        # Assume a time interval between frames in minutes.
        # This value is crucial and is based on common radar data intervals.
        # Adjust this if the actual GIF frame interval in minutes is different.
        frame_interval_minutes = 5

        def get_radar_center_of_mass(img_rgb):
            # Convert to grayscale for intensity analysis
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            # Create a binary mask: pixels that are not black (0) are considered radar activity
            _, mask = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)

            # Calculate moments of the binary image to find center of mass
            M = cv2.moments(mask)

            # Calculate x,y coordinate of center of mass
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                return cX, cY
            else:
                return None # No radar activity detected in this frame

        center_prev_pixels = get_radar_center_of_mass(frame_prev_rgb)
        center_curr_pixels = get_radar_center_of_mass(frame_curr_rgb)
        speed_meters_per_minute = 0.0

        if center_prev_pixels and center_curr_pixels:
            cX_prev, cY_prev = center_prev_pixels
            cX_curr, cY_curr = center_curr_pixels
            print("cX_curr, cY_curr : ", cX_curr, cY_curr)
            # Calculate displacement in pixels
            delta_x_pixels = cX_curr - cX_prev
            # Note: In image coordinates, positive Y is typically downwards.
            # For geographic interpretation (North-up), we consider positive Y to be North.
            # So, a positive delta_y_pixels (downwards movement in image) means southward movement.
            delta_y_pixels = cY_curr - cY_prev

            # Convert displacement to meters using the pixel resolution
            delta_x_meters = delta_x_pixels * pixel_resolution
            delta_y_meters = delta_y_pixels * pixel_resolution

            # Calculate the magnitude of displacement (distance) in meters
            distance_meters = math.sqrt(delta_x_meters**2 + delta_y_meters**2)

            # Calculate speed (meters per minute)
            speed_meters_per_minute = distance_meters / frame_interval_minutes

            # Convert speed to km/h for better readability (1 km = 1000 m, 1 hour = 60 minutes)
            speed_km_per_hour = (speed_meters_per_minute * 60) / 1000

            # Calculate direction (angle in degrees, 0=North, 90=East, 180=South, 270=West)
            # We need to map image coordinates (X=East, Y=South) to a standard Cartesian system (X=East, Y=North)
            # So, the 'north_component' will be the negative of the 'delta_y_meters'.
            north_component_meters = -delta_y_meters
            east_component_meters = delta_x_meters

            # atan2(y, x) gives angle from positive X-axis (East) counter-clockwise.
            # Convert to degrees and adjust for North=0, increasing clockwise.
            angle_rad = math.atan2(north_component_meters, east_component_meters)
            angle_deg_from_east_ccw = math.degrees(angle_rad)

            # Convert to compass bearing (0=North, 90=East, 180=South, 270=West)
            # (90 - angle_from_east_ccw) maps East to 0, North to 90, West to 180, South to 270
            # Then add 360 and take modulo 360 to ensure positive values.
            direction_degrees_north_0 = (90 - angle_deg_from_east_ccw + 360) % 360

            print(f"--- Radar Cloud Nowcast ---")
            print(f"Movement detected between Frame {len(frames)-1} and Frame {len(frames)} (over {frame_interval_minutes} minutes):")
            print(f"  Speed: {speed_km_per_hour:.2f} km/h")
            print(f"  Direction: {direction_degrees_north_0:.2f} degrees (0=North, 90=East, 180=South, 270=West)")

            # --- Nowcast for a future time ---
            nowcast_minutes_ahead = 15 # Example: nowcast for 15 minutes into the future

            # Calculate projected distance based on speed
            nowcast_distance_meters = speed_meters_per_minute * nowcast_minutes_ahead

            # Calculate projected displacement vector in meters
            if distance_meters > 0: # Avoid division by zero if no movement
                proj_delta_x_meters = (nowcast_distance_meters / distance_meters) * delta_x_meters
                proj_delta_y_meters = (nowcast_distance_meters / distance_meters) * delta_y_meters
            else:
                proj_delta_x_meters = 0
                proj_delta_y_meters = 0

            # Calculate projected new center of mass in pixels
            # Add displacement to the current frame's center
            proj_cX_pixels = cX_curr + (proj_delta_x_meters / pixel_resolution)
            proj_cY_pixels = cY_curr + (proj_delta_y_meters / pixel_resolution)

            print(f"\nNowcast for {nowcast_minutes_ahead} minutes ahead:")
            print(f"  Projected center of radar cloud (pixel coordinates): ({proj_cX_pixels:.0f}, {proj_cY_pixels:.0f})")

            # Convert projected pixel coordinates to geographic UTM coordinates
            geo_left, geo_right, geo_bottom, geo_top = img_extent

            # Horizontal (X) conversion
            proj_geo_x = geo_left + (proj_cX_pixels / img_width) * (geo_right - geo_left)

            # Vertical (Y) conversion - remember image Y is inverted relative to geo Y
            proj_geo_y = geo_top - (proj_cY_pixels / img_height) * (geo_top - geo_bottom)

            print(f"  Projected center of radar cloud (geographic coordinates - UTM): ({proj_geo_x:.2f}, {proj_geo_y:.2f})")

        else:
            cX_curr, cY_curr = 0.0, 0.0
            speed_km_per_hour = 0.0
            print("Could not detect sufficient radar activity in the last two frames to calculate movement or project nowcast.")

    # Input your desired simulation parameters here
    n_frame = 14
    new_simulated_direction_degrees = degree_value  # Example: 90 degrees for East
    new_nowcast_minutes_ahead = 30      # Example: 30 minutes ahead
    if new_simulated_direction_degrees == '-':
        new_simulated_direction_degrees = 0.0
    print(f"Simulating nowcast with direction: {new_simulated_direction_degrees} degrees (0=North, 90=East) and {new_nowcast_minutes_ahead} minutes ahead.")
    
    #Reuse previously calculated speed and current radar position
    # Variables like speed_meters_per_minute, cX_curr, cY_curr, img_extent, img_width, img_height, pixel_resolution
    # are assumed to be available from previous execution of h0EGT7To0f1S and b_2sJPx7KgUJ

    # Calculate projected distance based on the existing speed
    simulated_nowcast_distance_meters = speed_meters_per_minute * new_nowcast_minutes_ahead

    # Convert simulated_direction_degrees (0=North, 90=East, increases clockwise)
    # to radians for trigonometric functions (0=East, increases counter-clockwise)
    # (90 - direction) converts North=0 to Math_Y_axis=90, East=90 to Math_X_axis=0
    print("new_simulated_direction_degrees : ", new_simulated_direction_degrees)
    simulated_angle_rad = np.deg2rad(90 - new_simulated_direction_degrees)

    # Calculate projected displacement vector components in meters
    # east_component (delta_x) corresponds to Math X-axis (cos)
    # north_component (delta_y) corresponds to Math Y-axis (sin)
    proj_delta_x_meters = simulated_nowcast_distance_meters * np.cos(simulated_angle_rad)
    proj_delta_y_meters = -simulated_nowcast_distance_meters * np.sin(simulated_angle_rad) # Negative because image Y is inverted (South is positive Y in image)

    # Calculate projected new center of mass in pixels
    # Add displacement to the current frame's center
    proj_cX_pixels = cX_curr + (proj_delta_x_meters / pixel_resolution)
    proj_cY_pixels = cY_curr + (proj_delta_y_meters / pixel_resolution)

    # Convert projected pixel coordinates to geographic UTM coordinates
    geo_left, geo_right, geo_bottom, geo_top = img_extent

    # Horizontal (X) conversion
    proj_geo_x = geo_left + (proj_cX_pixels / img_width) * (geo_right - geo_left)

    # Vertical (Y) conversion - remember image Y is inverted relative to geo Y
    proj_geo_y = geo_top - (proj_cY_pixels / img_height) * (geo_top - geo_bottom)

    # Update the global nowcast_minutes_ahead variable for consistency with plotting
    nowcast_minutes_ahead = new_nowcast_minutes_ahead

    print(f"\nSimulated Nowcast Results for {new_nowcast_minutes_ahead} minutes ahead:")
    print(f"  Projected center of radar cloud (pixel coordinates): ({proj_cX_pixels:.0f}, {proj_cY_pixels:.0f})")
    print(f"  Projected center of radar cloud (geographic coordinates - UTM): ({proj_geo_x:.2f}, {proj_geo_y:.2f})")
    print(f"  Simulated Direction: {new_simulated_direction_degrees} degrees (0=North, 90=East)")
    print(f"  Speed used for simulation: {speed_km_per_hour:.2f} km/h")

    print(f"Generating {n_frame} different nowcast data points...")

    # Define the time interval between each nowcast frame based on user's request (100 minutes / 4 frames = 25 minutes per frame)
    nowcast_interval_minutes = new_nowcast_minutes_ahead / n_frame
    print(f"Nowcast interval: {nowcast_interval_minutes} minutes")
    all_nowcast_data = []
    all_nowcast_rgba_frames = [] # New list to store RGBA frames for visualization

    # Loop to generate n_frame different nowcasts
    for i in range(n_frame):
        current_nowcast_minutes_ahead = (i + 1) * nowcast_interval_minutes

        # Calculate projected distance based on the existing speed
        proj_nowcast_distance_meters = speed_meters_per_minute * current_nowcast_minutes_ahead

        # Convert new_simulated_direction_degrees (0=East, 90=North, increases counter-clockwise)
        # to radians for trigonometric functions
        simulated_angle_rad = np.deg2rad(new_simulated_direction_degrees)

        # Calculate projected displacement vector components in meters
        # east_component (delta_x) corresponds to Math X-axis (cos)
        # north_component (delta_y) corresponds to Math Y-axis (sin)
        proj_delta_x_meters = proj_nowcast_distance_meters * np.cos(simulated_angle_rad)
        proj_delta_y_meters = -proj_nowcast_distance_meters * np.sin(simulated_angle_rad) # Negative because image Y is inverted (South is positive Y in image)

        # Calculate projected new center of mass in pixels
        # Add displacement to the current frame's center
        proj_cX_pixels_current_frame = cX_curr + (proj_delta_x_meters / pixel_resolution)
        proj_cY_pixels_current_frame = cY_curr + (proj_delta_y_meters / pixel_resolution)

        # Convert projected pixel coordinates to geographic UTM coordinates
        geo_left, geo_right, geo_bottom, geo_top = img_extent

        # Horizontal (X) conversion
        proj_geo_x_current_frame = geo_left + (proj_cX_pixels_current_frame / img_width) * (geo_right - geo_left)

        # Vertical (Y) conversion - remember image Y is inverted relative to geo Y
        proj_geo_y_current_frame = geo_top - (proj_cY_pixels_current_frame / img_height) * (geo_top - geo_bottom)

        # --- Shift the radar frame and calculate district metrics ---
        # Create a blank image to shift the radar frame onto
        shifted_radar_frame_rgb = np.zeros_like(frame_curr_rgb, dtype=np.uint8)

        # Calculate the actual pixel shift to apply
        shift_x_pixels = int(proj_cX_pixels_current_frame - cX_curr)
        shift_y_pixels = int(proj_cY_pixels_current_frame - cY_curr)

        # Create a translation matrix
        M_nowcast = np.float32([[1, 0, shift_x_pixels], [0, 1, shift_y_pixels]])

        # Apply the affine transformation (shift)
        shifted_radar_rgb = cv2.warpAffine(frame_curr_rgb, M_nowcast, (img_width, img_height), borderValue=(0,0,0))

        # Convert to RGBA for calculate_district_metrics
        shifted_radar_frame_rgba = np.zeros((img_height, img_width, 4), dtype=np.uint8)
        shifted_radar_frame_rgba[:, :, :3] = shifted_radar_rgb
        black_pixels_mask_shifted = np.all(shifted_radar_rgb == [0, 0, 0], axis=-1)
        shifted_radar_frame_rgba[~black_pixels_mask_shifted, 3] = 255
        shifted_radar_frame_rgba[black_pixels_mask_shifted, 3] = 0

        # Store the RGBA frame
        all_nowcast_rgba_frames.append(shifted_radar_frame_rgba)

        # Calculate district metrics for the nowcast frame
        nowcast_coverage, nowcast_scores = calculate_district_metrics(
            shifted_radar_frame_rgba, gdf, img_extent, target_colors, values, threshold
        )

        # Store results for the current nowcast frame
        for district_name in nowcast_coverage.keys():
            all_nowcast_data.append({
                'Nowcast_Frame_Num': i + 1,
                'Minutes_Ahead': current_nowcast_minutes_ahead,
                'District': district_name,
                'Coverage': nowcast_coverage.get(district_name, 0.0),
                'Score': nowcast_scores.get(district_name, 0.0),
                'Proj_Geo_X': proj_geo_x_current_frame,
                'Proj_Geo_Y': proj_geo_y_current_frame
            })
        print(f"  Generated nowcast data for {current_nowcast_minutes_ahead} minutes ahead (Frame {i+1}).")

    df_nowcast_output = pd.DataFrame(all_nowcast_data)

    # Ensure gdf and shapefile_path are defined
    # Assuming shapefile_path is defined in an earlier cell, e.g., 1zPY95e5iOYT
    # For robustness, explicitly load gdf here if it's not guaranteed to be in scope.
    if 'gdf' not in globals() or gdf is None:
        if 'shapefile_path' in globals():
            print(f"Loading gdf from {shapefile_path}...")
            gdf = gpd.read_file(shapefile_path)
        else:
            print("Error: 'shapefile_path' not defined. Cannot load gdf.")
            # You might want to halt execution or handle this error more gracefully

    # Ensure district_col is defined (e.g., 'ADM3_EN' or 'DISTRICT_T')
    district_col = 'ADM3_EN' if 'ADM3_EN' in gdf.columns else 'DISTRICT_T'
    if update : update("📡 กำลังแสดงผล Nowcast ข้อมูลกำลังโหลด...",0.9)
    print(f"Visualizing {n_frame} nowcast frames with movement direction...")

    gif_frames = []

    # =========================================================
    # LOOP
    # =========================================================


    nowcast_coverage = []
    num_district_data = {}

    for frame_index in range(n_frame):

        nowcast_frame_num = frame_index + 1
        current_nowcast_minutes_ahead = (frame_index + 1) * nowcast_interval_minutes

        nowcast_rgba_frame = all_nowcast_rgba_frames[frame_index]
        alpha_channel = nowcast_rgba_frame[:, :, 3]
        current_frame_nowcast_df = df_nowcast_output[
            df_nowcast_output['Nowcast_Frame_Num'] == nowcast_frame_num
        ]
        left, right, bottom, top = img_extent
        transform = rasterio.transform.from_bounds(left, bottom, right, top, w, h)
        nowcast_raining_districts_in_frame = current_frame_nowcast_df[
            (current_frame_nowcast_df['Coverage'] > 0) |
            (current_frame_nowcast_df['Score'] > 0)
        ]['District'].tolist()

        avg_coverage_current_frame = current_frame_nowcast_df['Coverage'].mean()
        avg_score_current_frame = current_frame_nowcast_df['Score'].mean()
        
        shapefile_mask = rasterio.features.rasterize(
            [(entire_shapefile_geometry, 1)],
            out_shape=(h, w),
            transform=transform,
            fill=0,
            dtype=np.uint8
        ).astype(bool)

        # Identify radar pixels (where alpha > 0) within the entire shapefile mask
        radar_pixels_in_shapefile = np.sum(shapefile_mask & (alpha_channel > 0))
        total_shapefile_pixels = np.sum(shapefile_mask)

        overall_percentage = 0.0
        if total_shapefile_pixels > 0:
            overall_percentage = (radar_pixels_in_shapefile / total_shapefile_pixels) * 100
            print("nowcast data ",overall_percentage, radar_pixels_in_shapefile, total_shapefile_pixels)
            nowcast_coverage.append(overall_percentage)
        # =====================================================
        # FIGURE
        # =====================================================
        fig, ax = plt.subplots(figsize=(12, 10))

        # -----------------------------------------------------
        # DISTRICTS WITHOUT RAIN
        # -----------------------------------------------------
        districts_without_nowcast = gdf[
            ~gdf[district_col].isin(nowcast_raining_districts_in_frame)
        ]

        districts_without_nowcast.plot(
            ax=ax,
            edgecolor='red',
            facecolor='none',
            linewidth=1
        )

        # -----------------------------------------------------
        # DISTRICTS WITH RAIN
        # -----------------------------------------------------
        districts_with_nowcast = gdf[
            gdf[district_col].isin(nowcast_raining_districts_in_frame)
        ]

        districts_with_nowcast.plot(
            ax=ax,
            edgecolor='black',
            facecolor='yellow',
            linewidth=2,
            alpha=0.5
        )

        # -----------------------------------------------------
        # RADAR IMAGE
        # -----------------------------------------------------
        ax.imshow(
            nowcast_rgba_frame,
            extent=img_extent,
            alpha=0.8
        )

        # -----------------------------------------------------
        # CURRENT RADAR
        # -----------------------------------------------------
        ax.scatter(
            radar_x,
            radar_y,
            color='blue',
            marker='+',
            s=200,
            label='Current Radar Station',
            zorder=5
        )

        # -----------------------------------------------------
        # PROJECTED CENTER
        # -----------------------------------------------------
        proj_geo_x_frame = current_frame_nowcast_df['Proj_Geo_X'].iloc[0]
        proj_geo_y_frame = current_frame_nowcast_df['Proj_Geo_Y'].iloc[0]

        ax.scatter(
            proj_geo_x_frame,
            proj_geo_y_frame,
            color='lime',
            marker='x',
            s=200,
            label='Projected Radar Center',
            zorder=5
        )

        # -----------------------------------------------------
        # MOVEMENT ARROW
        # -----------------------------------------------------
        arrow_dx = proj_geo_x_frame - radar_x
        arrow_dy = proj_geo_y_frame - radar_y

        movement_magnitude = np.sqrt(
            arrow_dx**2 + arrow_dy**2
        )

        if movement_magnitude > 0:

            arrow_display_length = 0.05 * np.sqrt(ax.get_xlim()[1] - ax.get_xlim()[0]) # A percentage of plot width

            # Convert direction (0=East, 90=North, increases counter-clockwise) to radians
            angle_rad_for_arrow = np.deg2rad(new_simulated_direction_degrees)

            # Calculate arrow components based on overall calculated direction
            arrow_component_x = arrow_display_length * np.cos(angle_rad_for_arrow)
            arrow_component_y = arrow_display_length * np.sin(angle_rad_for_arrow)

            # Draw arrow originating from projected center, pointing in the direction of movement
            ax.arrow(proj_geo_x_frame - arrow_component_x,
                    proj_geo_y_frame - arrow_component_y,
                    arrow_component_x,
                    arrow_component_y,
                    head_width=2000, head_length=3000, fc='magenta', ec='magenta',
                    linewidth=2, zorder=6, length_includes_head=True)

            speed_text = f"Speed: {speed_km_per_hour:.1f} km/h"
            direction_text = (
                f"Direction: "
                f"{new_simulated_direction_degrees:.0f}°"
            )

            ax.text(
                proj_geo_x_frame,
                proj_geo_y_frame + 5000,
                f"{speed_text}\n{direction_text}",
                color='magenta',
                fontsize=10,
                ha='center',
                va='bottom',
                bbox=dict(
                    facecolor='white',
                    alpha=0.7,
                    edgecolor='none',
                    boxstyle='round,pad=0.2'
                ),
                zorder=7
            )

        # -----------------------------------------------------
        # TITLE
        # -----------------------------------------------------
        ax.set_title(
            f'Nowcast Frame {nowcast_frame_num} '
            f'({current_nowcast_minutes_ahead:.0f} min ahead)'
        )

        ax.set_xlabel('UTM X Coordinate')
        ax.set_ylabel('UTM Y Coordinate')

        ax.grid(True, alpha=0.3)

        # -----------------------------------------------------
        # LEGEND
        # -----------------------------------------------------
        legend_handles = [
            Patch(
                facecolor='yellow',
                edgecolor='black',
                linewidth=2,
                alpha=0.5,
                label='Districts with Projected Radar Coverage'
            ),

            plt.Line2D(
                [0], [0],
                marker='+',
                color='blue',
                markersize=10,
                linestyle='None',
                label='Current Radar Station'
            ),

            plt.Line2D(
                [0], [0],
                marker='x',
                color='lime',
                markersize=10,
                linestyle='None',
                label='Projected Radar Center'
            ),

            Patch(
                facecolor='magenta',
                edgecolor='magenta',
                label='Cloud Movement Direction'
            )
        ]

        ax.legend(handles=legend_handles)

        plt.tight_layout()

        # =====================================================
        # FIGURE -> NUMPY IMAGE
        # =====================================================
        buf = io.BytesIO()

        fig.savefig(
            buf,
            format='png',
            bbox_inches='tight'
        )

        buf.seek(0)

        img_array = np.frombuffer(
            buf.getvalue(),
            dtype=np.uint8
        )

        frame_img = cv2.imdecode(
            img_array,
            cv2.IMREAD_COLOR
        )

        frame_img = cv2.cvtColor(
            frame_img,
            cv2.COLOR_BGR2RGB
        )

        gif_frames.append(frame_img)

        buf.close()
        plt.close(fig)
        # print(f"--- Metrics for Nowcast Frame {nowcast_frame_num} ({current_nowcast_minutes_ahead:.0f} min ahead) ---")
        print(f"--- Metrics for Nowcast Frame {nowcast_frame_num} ")
        # print(f"  Average Radar Coverage: {avg_coverage_current_frame:.2f}%")
        # print(f"  Average Radar Score: {avg_score_current_frame:.2f}")
        print(f"  Districts with significant radar activity: {nowcast_raining_districts_in_frame}\n")

        for district in nowcast_raining_districts_in_frame:

            num_district_data[district] = (
                num_district_data.get(district, 0) + frame_index
            )
    print(nowcast_coverage)
    # max
    max_n_coverage_value = max(nowcast_coverage)

    # average
    avg_n_coverage_value = np.mean(nowcast_coverage)

    # 50 percentile (median)
    coverage_n_p50 = np.percentile(nowcast_coverage, 50)
    # =========================================================
    # CREATE GIF IN MEMORY
    # =========================================================
    gif_frames_np = np.array(gif_frames)

    gif_prediction = create_in_memory_gif(
        gif_frames_np,
        fps=2
    )
    # if gif_prediction:
    #         pyn.display(pyn.Image(data=gif_prediction))
    # else:
    #         print("Failed to create in-memory GIF.")
    print("GIF created in memory")
    print(num_district_data)
    num_district_data = get_rain_data(num_district_data, n_frame)
    if update : update("loading ... ",1.0)
    return in_memory_gif_bytes, (max_coverage_value, avg_coverage_value, coverage_p50), gif_prediction, (max_n_coverage_value, avg_n_coverage_value, coverage_n_p50), num_district_data