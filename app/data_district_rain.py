import cv2
import imageio
import numpy as np
import geopandas as gpd
import io
import rasterio.features
import matplotlib.pyplot as plt
import os

from rasterio.transform import from_bounds
from shapely.geometry import mapping
import matplotlib as mpl

mpl.font_manager.fontManager.addfont('tahoma.ttf') # 3.2+
mpl.rc('font', family='Tahoma')
# =========================================================
# INPUT
# =========================================================
district_summary = {}
GIF_PATH = "./example/n006.gif"
SHP_PATH = "./mapdata/Export_Output.shp"
OUTPUT_DIR = "/content/rain_intensity_debug"
DEBUG = False
THRESHOLD_COLOR = 15
# os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# RADAR INFO
# =========================================================
radar_x = 699558.0797
radar_y = 1530232.3207
pixel_resolution = 300

# =========================================================
# RADAR COLORS + VALUES
# =========================================================
target_colors = np.array([
     [252,252,255],[252,219,255],[252,202,255],[252,139,255],
    [252,0,255],[195,0,85],[216,0,71],[224,0,85],[238,0,0],
    [252,75,0],[222,152,0],[230,164,0],[252,214,0],[216,216,0],
    [234,218,0],[238,252,0],[0,243,0],[0,236,0],[0,214,82],
    [0,200,0],[0,197,0],[0,191,0],[0,176,0],[0,168,0],[0,0,255]
], dtype=np.uint8)

values = np.array([
    66.5,64.0,61.5,59.0,56.5,54.0,51.5,49.0,46.5,
    44.0,41.5,39.0,36.5,34.0,31.5,29.0,26.5,24.0,
    21.5,19.0,16.5,14.0,11.5,10.0,9.5
])

# =========================================================
# HSV TABLE
# =========================================================
target_hsv = cv2.cvtColor(
    target_colors.reshape(-1,1,3),
    cv2.COLOR_RGB2HSV
).reshape(-1,3).astype(np.float32)


def get_data(GIF_PATH, update = None):
    if update: update("🌧️ กำลังโหลดข้อมูลเรดาร์ฝน...", 0.25)
    # =========================================================
    # LOAD DATA
    # =========================================================
    gdf = gpd.read_file(SHP_PATH)
    gif = imageio.mimread(GIF_PATH)

    H, W = gif[0].shape[:2]
    
    if update: update("🗺️ กำลังประมวลผลข้อมูลเขต...", 0.50)
    # =========================================================
    # RADAR EXTENT
    # =========================================================
    left = radar_x - (W / 2) * pixel_resolution
    right = radar_x + (W / 2) * pixel_resolution
    bottom = radar_y - (H / 2) * pixel_resolution
    top = radar_y + (H / 2) * pixel_resolution

    transform = from_bounds(left, bottom, right, top, W, H)

    # =========================================================
    # SHAPE MASK (ALL DISTRICTS)
    # =========================================================
    shape_mask = rasterio.features.rasterize(
        [(mapping(g), 1) for g in gdf.geometry],
        out_shape=(H, W),
        transform=transform,
        fill=0,
        dtype=np.uint8
    ).astype(bool)

    # =========================================================
    # FRAME LOOP
    # =========================================================
    if update: update("🎞️ กำลังวิเคราะห์เฟรมฝน...", 0.75)
    
    for frame_id, frame in enumerate(gif):

        print(f"\n================ FRAME {frame_id} ================")

        rgb = frame[:, :, :3] if frame.shape[-1] == 4 else frame.copy()

        # =====================================================
        # CONVERT IMAGE → RAIN VALUE MAP (IMPORTANT)
        # =====================================================
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hsv_pixels = hsv.reshape(-1,3).astype(np.float32)

        distances = np.linalg.norm(
            hsv_pixels[:,None,:] - target_hsv[None,:,:],
            axis=2
        )

        min_idx = distances.argmin(axis=1)
        min_dist = distances.min(axis=1)

        pixel_values = np.where(min_dist < THRESHOLD_COLOR, values[min_idx], 0.0)
        pixel_values = pixel_values.reshape(H, W)

        # =====================================================
        # FRAME OUTPUT DIR
        # =====================================================
        frame_dir = os.path.join(OUTPUT_DIR, f"frame_{frame_id:04d}")
        # os.makedirs(frame_dir, exist_ok=True)

        # =====================================================
        # DISTRICT LOOP
        # =====================================================
        for idx, row in gdf.iterrows():

            name = row.get("ADM3_EN") or row.get("DISTRICT_T") or f"d_{idx}"
            name = str(name).replace("/", "_")

            district_mask = rasterio.features.rasterize(
                [(mapping(row.geometry), 1)],
                out_shape=(H, W),
                transform=transform,
                fill=0,
                dtype=np.uint8
            ).astype(bool)

            if np.sum(district_mask) == 0:
                continue
            ys, xs = np.where(district_mask)

            ymin, ymax = ys.min(), ys.max()
            xmin, xmax = xs.min(), xs.max()

            margin = 20
            ymin = max(0, ymin - margin)
            ymax = min(H, ymax + margin)
            xmin = max(0, xmin - margin)
            xmax = min(W, xmax + margin)
            # print("Crop box:")
            # print(xmin, xmax, ymin, ymax)
            # =================================================
            # RAIN INSIDE DISTRICT
            # =================================================
            vals = pixel_values[district_mask]

            rain_pixels = np.sum(vals > 0)
            total_pixels = np.sum(district_mask)

            coverage = rain_pixels / total_pixels if total_pixels > 0 else 0
            
            # =================================================
            # INTENSITY
            # =================================================
            rain_vals = vals[vals > 0]

            if len(rain_vals) == 0:

                mean_v = 0.0
                p90_v = 0.0

                mean_norm = 0.0
                p90_norm = 0.0

            else:

                # raw value
                mean_v = np.mean(rain_vals)
                p90_v = np.percentile(rain_vals, 90)

                # normalize 0-1
                mean_norm = (
                    mean_v - values.min()
                ) / (
                    values.max() - values.min()
                )

                p90_norm = (
                    p90_v - values.min()
                ) / (
                    values.max() - values.min()
                )

            # =================================================
            # FINAL INTENSITY SCORE
            # =================================================
            intensity = (
                0.6 * p90_norm +
                0.4 * mean_norm
            )

            # =================================================
            # FINAL RAIN SCORE
            # =================================================
            rain_score = (
                0.7 * coverage +
                0.3 * intensity
            )

            # =================================================
            # CLASSIFICATION
            # =================================================
            if rain_score > 0.7:
                level = "heavy"
            elif rain_score > 0.4:
                level = "medium"

                # break
            elif rain_score > 0.01:
                level = "light"
                print("district name",name)
                print("rain_score", rain_score)
                print("coverage", coverage)
                print(p90_norm, mean_norm)
                print(p90_v, mean_v)
            else:
                level = "none"
            if name not in district_summary:
                district_summary[name] = {
                    "heavy": 0,
                    "medium": 0,
                    "light": 0,
                    "none": 0,
                    "max_score": 0,
                    "sum_score": 0,
                    "frames": 0
                }

            district_summary[name][level] += 1

            district_summary[name]["max_score"] = max(
            district_summary[name]["max_score"],
            rain_score
            )

            district_summary[name]["sum_score"] += rain_score
            district_summary[name]["frames"] += 1
            if DEBUG:
                # =================================================
                # OUTPUT FOLDER
                # =================================================
                ddir = os.path.join(frame_dir, name)
                os.makedirs(ddir, exist_ok=True)

                # =================================================
                # VISUALS
                # =================================================
                cropped = np.zeros_like(rgb)
                cropped[district_mask] = rgb[district_mask]

                rain_mask_vis = (pixel_values > 0).astype(np.uint8) * 255
                rain_mask_vis = rain_mask_vis * district_mask

                highlight = cropped.copy()
                highlight[district_mask] = rgb[district_mask]
                highlight[district_mask & (pixel_values > 0)] = [255, 0, 0]

                # =================================================
                # SAVE 1 ORIGINAL
                # =================================================
                plt.figure(figsize=(6,6))
                plt.imshow(rgb)
                plt.title(f"{name} | {level} | score={rain_score:.1f}")
                plt.axis("off")
                plt.savefig(os.path.join(ddir, "01_original.png"))
                plt.close()

                # =================================================
                # SAVE 2 MASK
                # =================================================
                plt.figure(figsize=(6,6))
                plt.imshow(district_mask[ymin:ymax, xmin:xmax], cmap="gray")
                plt.title("mask")
                plt.axis("off")
                plt.savefig(os.path.join(ddir, "02_mask.png"))
                plt.close()

                # =================================================
                # SAVE 3 CROPPED
                # =================================================
                plt.figure(figsize=(6,6))
                plt.imshow(cropped[ymin:ymax, xmin:xmax])
                plt.title("cropped")
                plt.axis("off")
                plt.savefig(os.path.join(ddir, "03_cropped.png"))
                plt.close()

                # =================================================
                # SAVE 4 RAIN INTENSITY
                # =================================================
                plt.figure(figsize=(6,6))
                plt.imshow(rain_mask_vis[ymin:ymax, xmin:xmax], cmap="hot")
                plt.title(f"rain intensity | {level}")
                plt.axis("off")
                plt.savefig(os.path.join(ddir, "04_rain.png"))
                plt.close()

                # =================================================
                # SAVE 5 HIGHLIGHT
                # =================================================
                plt.figure(figsize=(6,6))
                plt.imshow(highlight[ymin:ymax, xmin:xmax])
                # plt.title(f"{name}\n{level} | p90={score:.1f}")
                plt.title(
                    f"HIGHLIGHT\n"
                    f"{name}\n{level} | p90={rain_score:.1f}\n"
                    f"Cov={coverage:.2f}% | Rain={rain_pixels} / {total_pixels}"
                )
                print( f"HIGHLIGHT\n"
                    f"{name}\n{level} | p90={rain_score:.1f}\n"
                    f"Cov={coverage:.2f}% | Rain={rain_pixels} / {total_pixels}")
                plt.axis("off")
                plt.savefig(os.path.join(ddir, "05_highlight.png"))
                plt.close()

    print("DONE")

    print("\nรายชื่อเขตตามกลุ่มข้อมูลความเข้มฝน:")
    if update: update("📊 กำลังจัดกลุ่มระดับความรุนแรงฝน...", 0.9)
    # Define levels for determining dominant rain intensity
    levels = ["heavy", "medium", "light", "none"]

    # Create a dictionary to hold districts grouped by rain level
    grouped_districts = {
        'heavy': [],
        'medium': [],
        'light': [],
        'none': []
    }

    for district, info in district_summary.items():
        # Determine the dominant rain level for each district
        dominant = max(
            levels,
            key=lambda x: info[x]
        )
        grouped_districts[dominant].append(district)

    # Prepare Thai labels for output
    thai_intensity_labels = {
        'none': 'ไม่มีฝน',
        'light': 'ฝนเบา',
        'medium': 'ฝนปานกลาง',
        'heavy': 'ฝนหนัก'
    }

    # Print the grouped districts
    data_district_rain = {}
    for level in levels:
        thai_label = thai_intensity_labels.get(level, level)
        districts_in_group = grouped_districts[level]
        if districts_in_group:
            data_district_rain[level] = []
            print(f"\n--- {thai_label} ({len(districts_in_group)} เขต) ---")
            for district_name in districts_in_group:
                print(f"- {district_name}")
                data_district_rain[level].append(district_name)


    # import matplotlib.pyplot as plt
    # import geopandas as gpd
    # import pandas as pd

    # Define levels for determining dominant rain intensity
    levels = ["heavy", "medium", "light", "none"]

    # Prepare district_rain_levels dictionary from district_summary
    district_rain_levels = {}
    for district, info in district_summary.items():
        # Determine the dominant rain level for each district
        dominant = max(
            levels,
            key=lambda x: info[x]
        )
        district_rain_levels[district] = dominant

    # Define colors for rain levels
    color_map = {
        'none': '#D3D3D3',   # Light gray
        'light': '#90EE90',  # Light green
        'medium': '#ADD8E6', # Light blue
        'heavy': '#FF6347'   # Tomato red
    }

    # Prepare a list to hold GeoSeries for the districts to be plotted
    districts_to_plot = []

    # Iterate through the districts and their determined rain levels
    for user_district_name, rain_level in district_rain_levels.items():
        # Find the matching row in the original gdf
        matching_rows = gdf[(gdf['DISTRICT_T'] == user_district_name) | (gdf['DISTRICT_E'] == user_district_name)]

        if not matching_rows.empty:
            # Take the first match (assuming unique district names)
            found_row = matching_rows.iloc[0].copy()
            found_row['rain_level_user'] = rain_level # Add the user-defined rain level
            districts_to_plot.append(found_row)
        else:
            print(f"Warning: District '{user_district_name}' not found in GeoDataFrame.")

    # Create a new GeoDataFrame containing only the districts to be plotted with their assigned rain levels
    if not districts_to_plot:
        print("No matching districts found for plotting.")
    else:
        gdf_filtered_for_plot = gpd.GeoDataFrame(districts_to_plot, crs=gdf.crs)

        # Plotting
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        # Plot the base map of all districts in a very light grey for context
        gdf.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5, alpha=0.5)

        # Plot the filtered districts with specified colors based on user input
        # Iterate through unique rain levels present in the filtered data to ensure a correct legend
        unique_levels_in_plot = gdf_filtered_for_plot['rain_level_user'].unique()
        for level_name in unique_levels_in_plot:
            subset = gdf_filtered_for_plot[gdf_filtered_for_plot['rain_level_user'] == level_name]
            if not subset.empty:
                subset.plot(ax=ax, color=color_map.get(level_name, '#CCCCCC'), edgecolor='black', linewidth=0.8, legend=True, label=level_name)

        ax.set_title('Rain Intensity Map by District (from district_data)', fontdict={'fontsize': 20, 'fontweight': 'medium'})
        ax.set_axis_off()

        # Create a custom legend using the unique levels actually plotted
        # Filter color_map to only include levels present in the plot
        legend_handles = [plt.Rectangle((0,0),1,1, color=color_map.get(level, '#CCCCCC')) for level in sorted(unique_levels_in_plot)]
        legend_labels = [level.capitalize() for level in sorted(unique_levels_in_plot)]
        ax.legend(legend_handles, legend_labels, title="Rain Intensity", loc='lower left')

        # plt.show()

        buf = io.BytesIO()

        fig.savefig(
            buf,
            format='png',
            bbox_inches='tight',
            dpi=150
        )

        buf.seek(0)

        img_array = np.frombuffer(
            buf.getvalue(),
            dtype=np.uint8
        )

        plot_img = cv2.imdecode(
            img_array,
            cv2.IMREAD_COLOR
        )

        plot_img = cv2.cvtColor(
            plot_img,
            cv2.COLOR_BGR2RGB
        )

        buf.close()
        plt.close(fig)
        if update: update("📈...", 1.0)
        # =========================
        # RETURN NUMPY IMAGE
        # =========================
        return plot_img, data_district_rain