import os
import cv2
import imageio
import numpy as np
import geopandas as gpd
import rasterio.features
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from rasterio.transform import from_bounds
from shapely.geometry import mapping
import io

# =========================================================
# INPUT
# =========================================================
GIF_PATH = "/content/drive/MyDrive/radar/n006.gif"
SHP_PATH = "./mapdata/Export_Output.shp"

OUTPUT_DIR = "/content/rain_debug_split"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# RADAR INFO
# =========================================================
radar_x = 699558.0797
radar_y = 1530232.3207
pixel_resolution = 300

# =========================================================
# RADAR COLORS
# =========================================================
target_colors = np.array([
    [252,252,255],
    [252,219,255],
    [252,202,255],
    [252,139,255],
    [252,0,255],
    [195,0,85],
    [216,0,71],
    [224,0,85],
    [238,0,0],
    [252,75,0],
    [222,152,0],
    [230,164,0],
    [252,214,0],
    [216,216,0],
    [234,218,0],
    [238,252,0],
    [0,243,0],
    [0,236,0],
    [0,214,82],
    [0,200,0],
    [0,197,0],
    [0,191,0],
    [0,176,0],
    [0,168,0],
    [0,0,255]
], dtype=np.uint8)

# =========================================================
# RGB -> HSV
# =========================================================
target_hsv = cv2.cvtColor(
    target_colors.reshape(-1,1,3),
    cv2.COLOR_RGB2HSV
).reshape(-1,3)

def get_data(GIF_PATH):
    data = {}
    # =========================================================
    # LOAD SHAPEFILE
    # =========================================================
    gdf = gpd.read_file(SHP_PATH)

    print("CRS:", gdf.crs)

    # =========================================================
    # LOAD GIF
    # =========================================================
    gif = imageio.mimread(GIF_PATH)

    print("Frames:", len(gif))

    H, W = gif[0].shape[:2]

    print("Image size:", W, H)

    # =========================================================
    # RADAR EXTENT
    # =========================================================
    left = radar_x - (W / 2) * pixel_resolution
    right = radar_x + (W / 2) * pixel_resolution

    bottom = radar_y - (H / 2) * pixel_resolution
    top = radar_y + (H / 2) * pixel_resolution

    transform = from_bounds(
        left,
        bottom,
        right,
        top,
        W,
        H
    )

    # =========================================================
    # CREATE SHAPE MASK
    # =========================================================
    all_shapes = [(mapping(g), 1) for g in gdf.geometry]

    shape_mask = rasterio.features.rasterize(
        all_shapes,
        out_shape=(H, W),
        transform=transform,
        fill=0,
        dtype=np.uint8
    ).astype(bool)

    # =========================================================
    # BORDER
    # =========================================================
    contours, _ = cv2.findContours(
        shape_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # =========================================================
    # PROCESS EACH FRAME
    # =========================================================
    for frame_id, frame in enumerate(gif):

        print(f"FRAME {frame_id}")

        # -----------------------------------------------------
        # RGB
        # -----------------------------------------------------
        if frame.shape[-1] == 4:
            rgb = frame[:, :, :3]
        else:
            rgb = frame.copy()

        # =====================================================
        # HSV MATCHING
        # =====================================================
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        hsv_pixels = hsv.reshape(-1,3).astype(np.float32)

        distances = np.linalg.norm(
            hsv_pixels[:,None,:] -
            target_hsv[None,:,:].astype(np.float32),
            axis=2
        )

        min_dist = distances.min(axis=1)

        threshold = 40

        matched = min_dist < threshold

        rain_mask = matched.reshape(H,W)

        # =====================================================
        # APPLY SHAPE MASK
        # =====================================================
        final_rain_mask = rain_mask & shape_mask

        # =====================================================
        # COVERAGE
        # =====================================================
        total_pixels = np.sum(shape_mask)

        rain_pixels = np.sum(final_rain_mask)

        coverage = 0.0

        if total_pixels > 0:
            coverage = (rain_pixels / total_pixels) * 100

        print(f"coverage = {coverage:.2f}%")
        ys, xs = np.where(shape_mask)

        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()

        # small margin
        margin = 20

        ymin = max(0, ymin - margin)
        ymax = min(H, ymax + margin)

        xmin = max(0, xmin - margin)
        xmax = min(W, xmax + margin)

        print("Crop box:")
        print(xmin, xmax, ymin, ymax)


        # =====================================================
        # 1 ORIGINAL
        # =====================================================
   

        # =====================================================
        # 2 SHAPE MASK
        # =====================================================
        mask_vis = np.zeros((H,W), dtype=np.uint8)
        mask_vis[shape_mask] = 255

        # =====================================================
        # 3 CROPPED
        # =====================================================
        cropped = np.zeros_like(rgb)
        cropped[shape_mask] = rgb[shape_mask]


        # =====================================================
        # 4 RAIN MASK
        # =====================================================
        rain_vis = np.zeros((H,W), dtype=np.uint8)

        rain_vis[final_rain_mask] = 255

    

        # =====================================================
        # 5 HIGHLIGHT
        # =====================================================
        highlight = cropped.copy()

        # red rain cloud
        highlight[final_rain_mask] = [255,0,0]

        # white border
        cv2.drawContours(
            highlight,
            contours,
            -1,
            (255,255,255),
            2
        )
        data[frame_id] = {'coverage_value':coverage , "rain_pixel": rain_pixels, 'total_pixel':total_pixels}

    # Extract frame IDs and coverage values from the 'data' dictionary
    print("data : ",data)
    frame_ids = np.array(list(data.keys()))
    coverage_values = np.array([data[f_id]['coverage_value'] for f_id in frame_ids])
    

    # =========================
    # LINEAR REGRESSION
    # =========================
    slope, intercept, r_value, p_value, std_err = linregress(
        frame_ids,
        coverage_values
    )

    regress_line = slope * frame_ids + intercept

    # Interpret the trend
    if slope > 0.05:
        trend = "increasing"
    elif slope < -0.05:
        trend = "decreasing"
    else:
        trend = "relatively constant"

    # =========================
    # PLOT
    # =========================
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(frame_ids, coverage_values,
            label='Rainfall Coverage per Frame', color='blue')

    ax.plot(frame_ids, regress_line, color='red', label=f'y = {slope:.2f}x + {intercept:.2f}')

    ax.set_title('Linear Trend Analysis')
    ax.set_xlabel('Frame ID')
    ax.set_ylabel('Coverage (%)')
    ax.legend()
    ax.grid(True)

    # =========================
    # FIG -> IMAGE
    # =========================
    buf = io.BytesIO()

    fig.savefig(buf, format='png', bbox_inches='tight')

    buf.seek(0)

    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)

    plotimg = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    plotimg = cv2.cvtColor(plotimg, cv2.COLOR_BGR2RGB)

    buf.close()
    plt.close(fig)
    
    average_coverage = np.mean(coverage_values)
    data_coverage_and_trend_rain = average_coverage, trend
        # plot_img is numpy image
    return rgb,  mask_vis , cropped[ymin:ymax, xmin:xmax], rain_vis[ymin:ymax, xmin:xmax], highlight[ymin:ymax, xmin:xmax], plotimg, data_coverage_and_trend_rain
