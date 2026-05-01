import cv2
import numpy as np
import imageio.v2 as imageio
import geopandas as gpd
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt
from rasterio.transform import from_bounds
import rasterio.features # Required for calculate_district_metrics
from scipy.stats import linregress
import pandas as pd # Import pandas for DataFrame creation
from rasterio.transform import from_bounds
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
def extract_radar_frame(img):
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
    draw_white_boxes=True, # Option to disable the hardcoded box drawing if needed
    update=None
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
    if update: update("🌧️ กำลังประมวลผล...", 0.3)
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
        if update: update("🌧️ กำลังตรวจสอบพื้นที่ที่มีฝน...", 0.4)
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
    if update: update("🌧️ การคำนวนภาพรวม...", 0.4)
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
    if update: update("🌧️ การคำนวนภาพรวมแต่ละเขตเวลา...", 0.4)
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
    if update: update("🌧️ การคำนวนภาพรวมแต่ละเขตเวลา...", 0.4)
    return pd.DataFrame(df_rows)
gif_path = "/content/drive/MyDrive/radar/n006.gif"
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
# df_radar_metrics = process_radar_animation_and_extract_district_values(
#     gif_path=gif_path,
#     radar_x=radar_x,
#     radar_y=radar_y,
#     pixel_resolution=pixel_resolution,
#     shapefile_path=shapefile_path,
#     target_colors=target_colors,
#     values=values
# )

# Assuming df_radar_metrics is available from previous cells
# Filter for districts that actually had some rain (score > 0 at some point)

def classify_rain_intensity(score):
    """
    Classifies rain intensity based on radar score.
    These thresholds are illustrative and can be adjusted.
    """
    if score > 30: # Example threshold for heavy rain
        return 'Heavy'
    elif score > 15: # Example threshold for medium rain
        return 'Medium'
    elif score > 0: # Any score above 0 but less than or equal to 15
        return 'Light'
    else:
        return 'No Rain'


# display(avg_score_by_intensity.head(20))
def generate_data(gif_path, update = None):
    data = {}
    if update: update("🖼️ กำลัง สร้างรายงานไฟล์...", 0.1)
    df_radar_metrics = process_radar_animation_and_extract_district_values(
            gif_path=gif_path,
            radar_x=radar_x,
            radar_y=radar_y,
            pixel_resolution=pixel_resolution,
            shapefile_path=shapefile_path,
            target_colors=target_colors,
            values=values,
            update=update
    )
    data['data_rain_level_frame'] = df_radar_metrics

    if update: update("🌧️ กำลังตรวจสอบพื้นที่ที่มีฝน...", 0.3)
    raining_districts_df = df_radar_metrics[df_radar_metrics['Score'] > 0]
    data['notift_district_name'] = raining_districts_df

    if update: update("📈 กำลังวิเคราะห์แนวโน้มฝน...", 0.5)
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
    
    if update: update("🔥 กำลังจัดระดับความรุนแรงของฝน...", 0.7)
    # Apply the classification function to the 'Score' column
    df_radar_metrics['Rain_Intensity'] = df_radar_metrics['Score'].apply(classify_rain_intensity)
    data['Rain Intensity Classification per District and Frame'] = df_radar_metrics[['Frame', 'District', 'Score', 'Rain_Intensity']]
    # Display the results, grouped by district and rain intensity
    print("Rain Intensity Classification per District and Frame:")
    # display(df_radar_metrics[['Frame', 'District', 'Score', 'Rain_Intensity']].head(20))
    if update: update("📊 กำลังสรุปผลรายเขต...", 0.75)
    print("\nSummary of Rain Intensity per District:")
    intensity_summary = df_radar_metrics.groupby(['District', 'Rain_Intensity']).size().unstack(fill_value=0)
    data['Rain Intensity District Name'] = df_radar_metrics.groupby(['District', 'Rain_Intensity']).size().unstack(fill_value=0)
    # display(intensity_summary.head(20))

    # You can also see the average score per intensity group for each district
    print("\nAverage Score by Rain Intensity per District:")
    avg_score_by_intensity = df_radar_metrics[df_radar_metrics['Score'] > 0].groupby(['District', 'Rain_Intensity'])['Score'].mean().unstack()

    data['average_intensity'] = avg_score_by_intensity
    if update: update("✅ สร้างรายงานเสร็จสิ้น", 0.9)
    return data
if __name__ == "__main__":
    generate_data("example/n006.gif")
    