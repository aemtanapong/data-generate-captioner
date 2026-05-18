import cv2
import numpy as np
import imageio.v2 as imageio
import geopandas as gpd
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt
from rasterio.transform import from_bounds
from shapely.geometry import mapping
import rasterio.features # Required for calculate_district_metrics
import pandas as pd # Import pandas for DataFrame creation
from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import os
from matplotlib.patches import Patch
import io
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
# gif_path = "/content/drive/MyDrive/radar/radar (2).gif"
gif_path = "./example/n006.gif"
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
SHP_PATH = "./mapdata/Export_Output.shp" # ชื่อไฟล์ Shapefile ของคุณ
threshold = 60
def get_data(GIF_PATH, update = None):
    if update: update("🌧️ กำลังโหลดข้อมูลเรดาร์ฝน...", 0.25)
    # =========================================================
    # LOAD DATA
    # =========================================================
    gdf = gpd.read_file(SHP_PATH)
    gif = imageio.mimread(GIF_PATH)
    df_radar_metrics = process_radar_animation_and_extract_district_values(
        gif_path=gif_path,
        radar_x=radar_x,
        radar_y=radar_y,
        pixel_resolution=pixel_resolution,
        shapefile_path=SHP_PATH,
        target_colors=target_colors,
        values=values
    )
    if update: update("🗺️ กำลังประมวลผลข้อมูลเขต...", 0.50)
    df_radar_metrics
    

    # Assuming df_radar_metrics is available from previous cells
    # Filter for districts that actually had some rain (score > 0 at some point)
    raining_districts_df = df_radar_metrics[df_radar_metrics['Score'] > 0]

    districts_with_upward_trend = []

    # Iterate through each unique district that experienced rain
    for district_name in raining_districts_df['District'].unique():
        district_data = raining_districts_df[raining_districts_df['District'] == district_name]

        # Ensure there's enough data points for linear regression (at least 2 frames)
        if len(district_data) > 1:
            # Perform linear regression: Score (y) vs. Frame (x)
            slope, intercept, r_value, p_value, std_err = linregress(district_data['Frame'], district_data['Score'])

            # Consider an upward trend if the slope is positive and statistically significant (p-value < 0.05)
            # A higher slope threshold can be added for a 'stronger' trend, e.g., slope > 0.5
            if slope > 0 and p_value < 0.05:
                districts_with_upward_trend.append({
                    'District': district_name,
                    'Slope': slope,
                    'P_Value': p_value,
                    'R_Value': r_value
                })

    if districts_with_upward_trend:
        trend_df = pd.DataFrame(districts_with_upward_trend)
        trend_df = trend_df.sort_values(by='Slope', ascending=False)
        print("Districts with a significant upward linear trend in rain level:")
        for index, row in trend_df.iterrows():
            print(f"- {row['District']}: Slope={row['Slope']:.2f}, R-squared={row['R_Value']**2:.2f}, P-value={row['P_Value']:.3f}")
    else:
        print("No districts found with a significant upward linear trend in rain level.")

    frames = []

    if update: update("🎞️ สร้างภาพแต่ละ frame...", 0.60)
    average_map_frame = []
    # -----------------------------
    # READ GIF
    # -----------------------------
    with Image.open(gif_path) as img:
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
    if update: update("📊 กำลังสร้างภาพข้อมูล...", 0.70)
    # Load gdf if not already available or ensure it's in scope
    if 'gdf' not in globals() or gdf is None:
        print("Loading gdf globally...")
        gdf = gpd.read_file(SHP_PATH)
    else:
        print("gdf is already loaded globally.")

    # Determine img_width and img_height from the processed frames (available from iTnYQGi2qYq9)
    if 'frames' in globals() and len(frames) > 0:
        img_height, img_width, _ = frames[0].shape
        print(f"img_width: {img_width}, img_height: {img_height} derived from 'frames' variable.")
    else:
        # Fallback if 'frames' is not available or empty (should not happen if iTnYQGi2qYq9 ran)
        print("Warning: 'frames' variable not found or empty. Reading GIF directly for dimensions.")
        with Image.open(gif_path) as im:
            img_width, img_height = im.size
        print(f"img_width: {img_width}, img_height: {img_height} derived from original GIF.")

    # Calculate img_extent using global variables
    left = radar_x - (img_width / 2 * pixel_resolution)
    right = radar_x + (img_width / 2 * pixel_resolution)
    bottom = radar_y - (img_height / 2 * pixel_resolution)
    top = radar_y + (img_height / 2 * pixel_resolution)
    img_extent = [left, right, bottom, top]
    print(f"img_extent calculated globally: {img_extent}")
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
    # =========================================================
    # AUTO DETECT DISTRICT COLUMN
    # =========================================================
    print(gdf.columns)

    possible_cols = [

        "ADM3_EN",
        "ADM3_TH",
        "DISTRICT_T",
        "DISTRICT",
        "NAME",
        "AMPHOE_T",
        "AMP_NAME",
        "เขต"
    ]

    district_col = None

    for col in possible_cols:

        if col in gdf.columns:

            district_col = col
            break

    if district_col is None:

        raise ValueError(
            f"No district column found.\nAvailable columns:\n{gdf.columns}"
        )

    print("Using district column:", district_col)
    
    # =========================================================
    # IMAGE SIZE
    # =========================================================
    h, w, _ = frames[0].shape

    # =========================================================
    # TRANSFORM
    # =========================================================
    left, right, bottom, top = img_extent

    transform = from_bounds(
        left,
        bottom,
        right,
        top,
        w,
        h
    )

    # =========================================================
    # STORE COVERAGE
    # =========================================================
    district_coverage_summary = {}

    all_raining_districts_per_frame = []
    gif_data = []
    # =========================================================
    # PROCESS EACH FRAME
    # =========================================================
    for i, frame_rgb in enumerate(frames):

        print(f"\n========================")
        print(f"FRAME {i+1}")
        print(f"========================")

        # =====================================================
        # CREATE FIGURE
        # =====================================================
        fig, ax = plt.subplots(figsize=(12, 10))

        # =====================================================
        # RGBA CONVERSION
        # =====================================================
        frame_rgba = np.zeros((h, w, 4), dtype=np.uint8)

        frame_rgba[:, :, :3] = frame_rgb

        black_pixels_mask = np.all(
            frame_rgb == [0, 0, 0],
            axis=-1
        )

        frame_rgba[~black_pixels_mask, 3] = 255
        frame_rgba[black_pixels_mask, 3] = 0

        # =====================================================
        # RADAR MASK
        # =====================================================
        alpha_channel = frame_rgba[:, :, 3]

        radar_mask = alpha_channel > 0

        # =====================================================
        # CURRENT RAINING DISTRICT
        # =====================================================
        raining_districts_in_frame = []

        rain_persistance = {}
        # =====================================================
        # LOOP DISTRICT
        # =====================================================
        for idx, row in gdf.iterrows():

            district_name = row[district_col]

            # -------------------------------------------------
            # CREATE DISTRICT MASK
            # -------------------------------------------------
            district_mask = rasterio.features.rasterize(
                [(mapping(row.geometry), 1)],
                out_shape=(h, w),
                transform=transform,
                fill=0,
                dtype=np.uint8
            ).astype(bool)
            combined_mask = district_mask & radar_mask
            # -------------------------------------------------
            # PIXEL COUNT
            # -------------------------------------------------
            total_pixels = np.sum(district_mask)

            radar_pixels = np.sum(combined_mask)

            # -------------------------------------------------
            # COVERAGE %
            # -------------------------------------------------
            coverage = 0.0

            if total_pixels > 0:

                coverage = (
                    radar_pixels / total_pixels
                ) * 100
            print(radar_pixels ," / ", total_pixels)
            # -------------------------------------------------
            # SAVE COVERAGE
            # -------------------------------------------------
            if district_name not in district_coverage_summary:

                district_coverage_summary[district_name] = []

            district_coverage_summary[district_name].append(
                coverage
            )

            # -------------------------------------------------
            # CHECK RAIN
            # -------------------------------------------------
            if coverage > 0:

                raining_districts_in_frame.append(
                    district_name
                )

            print(
                district_name,
                f"Coverage = {coverage:.2f}%"
            )
            # =====================================================
            # GET RGB PIXELS INSIDE DISTRICT
            # =====================================================
            district_pixels = frame_rgb[combined_mask]

            # =====================================================
            # STORE dBZ VALUES
            # =====================================================
            # =====================================================
            # AVERAGE INTENSITY
            # =====================================================
            average_intensity = 0.0

            if len(district_pixels) > 0:

                # -------------------------------------------------
                # RGB DIFFERENCE
                # shape:
                # (num_pixels, num_colors, 3)
                # -------------------------------------------------
                diff = (
                    district_pixels[:, None, :].astype(np.float32)
                    -
                    target_colors[None, :, :].astype(np.float32)
                )

                # -------------------------------------------------
                # COLOR DISTANCE
                # -------------------------------------------------
                distances = np.linalg.norm(
                    diff,
                    axis=2
                )

                # -------------------------------------------------
                # NEAREST RADAR COLOR
                # -------------------------------------------------
                nearest_indices = np.argmin(
                    distances,
                    axis=1
                )

                # -------------------------------------------------
                # CONVERT TO dBZ
                # -------------------------------------------------
                pixel_dbz_values = values[
                    nearest_indices
                ]

                # -------------------------------------------------
                # AVERAGE dBZ
                # -------------------------------------------------
                average_intensity = np.mean(
                    pixel_dbz_values
                )

            print(
                district_name,
                f"Average Intensity = {average_intensity:.2f} dBZ"
            )
            # =====================================================
            # NORMALIZE COVERAGE
            # =====================================================
            normalized_coverage = np.clip(
                coverage / 100.0,
                0,
                1
            )

            # =====================================================
            # NORMALIZE INTENSITY
            # =====================================================
            normalized_intensity = np.clip(
                average_intensity / 70.0,
                0,
                1
            )

            # =====================================================
            # RAIN SEVERITY SCORE
            # =====================================================
            # rain_severity_score = (
            #     normalized_coverage * 0.4
            #     +
            #     normalized_intensity * 0.6
            # ) * 100
            # =====================================================
            # RAIN SEVERITY SCORE
            # Composite Formula
            # =====================================================

            raw_score = (
                coverage
                *
                average_intensity
            )

            # =====================================================
            # MAX POSSIBLE SCORE
            # =====================================================

            max_possible_score = (
                100.0
                *
                np.max(values)
            )

            # =====================================================
            # NORMALIZE TO 0-100
            # =====================================================

            rain_severity_score = (
                raw_score
                /
                max_possible_score
            ) * 100
            print(
                district_name,
                f"Rain Severity Score = "
                f"{rain_severity_score:.2f}"
            )
            if district_name not in rain_persistance:
                rain_persistance[district_name] = 0.0
            rain_persistance[district_name] += i * rain_severity_score
        
        # =====================================================
        # rain_severity_score
        # =====================================================
        # STORE RAINING DISTRICT
        # =====================================================
        all_raining_districts_per_frame.append({
            "frame": i + 1,
            "districts": raining_districts_in_frame
        })

        # =====================================================
        # SPLIT DISTRICT
        # =====================================================
        districts_with_radar = gdf[
            gdf[district_col].isin(
                raining_districts_in_frame
            )
        ]

        districts_without_radar = gdf[
            ~gdf[district_col].isin(
                raining_districts_in_frame
            )
        ]

        # =====================================================
        # PLOT DISTRICT
        # =====================================================
        districts_without_radar.plot(
            ax=ax,
            edgecolor='red',
            facecolor='none',
            linewidth=1
        )

        districts_with_radar.plot(
            ax=ax,
            edgecolor='black',
            facecolor='yellow',
            linewidth=2,
            alpha=0.5
        )

        # =====================================================
        # OVERLAY RADAR
        # =====================================================
        ax.imshow(
            frame_rgba,
            extent=img_extent,
            alpha=0.8
        )

        # =====================================================
        # RADAR LOCATION
        # =====================================================
        ax.scatter(
            radar_x,
            radar_y,
            color='blue',
            marker='+',
            s=200
        )

        # =====================================================
        # DISTRICT LABEL
        # =====================================================
        for idx, row in gdf.iterrows():

            district_name = row[district_col]

            if district_name in raining_districts_in_frame:

                centroid_x = row.geometry.centroid.x
                centroid_y = row.geometry.centroid.y

                ax.annotate(
                    district_name,
                    xy=(centroid_x, centroid_y),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=8,
                    color='black',
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        fc="yellow",
                        alpha=0.7
                    )
                )

        # =====================================================
        # TITLE
        # =====================================================
        ax.set_title(
            f'Radar Frame {i+1}'
        )

        ax.set_xlabel(
            'UTM X Coordinate'
        )

        ax.set_ylabel(
            'UTM Y Coordinate'
        )

        ax.grid(True, alpha=0.3)

        # =====================================================
        # LEGEND
        # =====================================================
        legend_handles = [

            Patch(
                facecolor='yellow',
                edgecolor='black',
                linewidth=2,
                alpha=0.5,
                label='Districts with Radar Coverage'
            ),

            plt.Line2D(
                [0],
                [0],
                marker='+',
                color='blue',
                markersize=10,
                linestyle='None',
                label='Radar Station'
            )
        ]

        ax.legend(
            handles=legend_handles
        )

        plt.tight_layout()
        # Capture the plot as an image and close the figure to save memory
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
        gif_data.append(image_from_plot)
        plt.close(fig)
    gif_buffer = io.BytesIO()

    imageio.mimsave(
        gif_buffer,
        gif_data,
        format="GIF",
        duration=1.0,
        loop=0
    )

    gif_buffer.seek(0)

    # ==========================================
    # RETURN BYTES
    # ==========================================
    gif_bytes = gif_buffer.getvalue()
    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n===================================")
    print("DISTRICT COVERAGE SUMMARY")
    print("===================================")

    district_statistics = {}

    for district_name, coverage_list in district_coverage_summary.items():

        avg_coverage = np.mean(
            coverage_list
        )

        max_coverage = np.max(
            coverage_list
        )

        percentile_90 = np.percentile(
            coverage_list,
            90
        )

        district_statistics[district_name] = {

            "average": avg_coverage,

            "maximum": max_coverage,

            "p90": percentile_90
        }

        print(
            district_name,
            f"AVG={avg_coverage:.2f}%",
            f"MAX={max_coverage:.2f}%",
            f"P90={percentile_90:.2f}%"
        )

    # =========================================================
    # DISTRICTS PER FRAME
    # =========================================================
    print("\n===================================")
    print("RAINING DISTRICT PER FRAME")
    print("===================================")
    print(rain_persistance)
    for item in all_raining_districts_per_frame:

        print(
            f"Frame {item['frame']} : "
            f"{item['districts']}"
        )
    # Prepare Thai labels for output
    thai_intensity_labels = {
        'none': 'ไม่มีฝน',
        'light': 'ฝนเบา',
        'medium': 'ฝนปานกลาง',
        'heavy': 'ฝนหนัก'
    }
    def get_rain_level(score):

        if score >= 400:
            return "heavy"

        elif score >= 100:
            return "medium"

        elif score > 0:
            return "light"

        return "none"
    grouped_district_name = {
        level: []
        for level in thai_intensity_labels
    }
    # -------------------------------------------------
    # GROUP DISTRICT
    # -------------------------------------------------
    for district_name, rain_value in rain_persistance.items():

        level = get_rain_level(
            rain_value
        )

        grouped_district_name[level].append(
            district_name
        )

        print(
            district_name,
            rain_value,
            "->",
            thai_intensity_labels[level]
        )
    if update: update("📈...", 1.0)
    
    return gif_bytes, grouped_district_name

if __name__ == "__main__":
    print(get_data(gif_path))