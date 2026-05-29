import cv2
import io
import numpy as np
import imageio.v2 as imageio
import geopandas as gpd
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt
from rasterio.transform import from_bounds
from shapely.geometry import mapping
import rasterio.features # Required for calculate_district_metrics
import pandas as pd # Import pandas for DataFrame creation
import matplotlib.font_manager as fm
import os
from matplotlib.patches import Patch
from functools import cache
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from tqdm import tqdm
from shapely.geometry import Point
from rasterio.transform import from_bounds
from rasterio.features import geometry_mask
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
    print(thai_font_path)
# gif_path = '/content/drive/MyDrive/radar/20260525_040000.webp'
gif_path = r'radar_caption\central\history\2026\20260525_040000.webp'
radar_x = 699558.0797  # พิกัด UTM X ของจุดกึ่งกลาง (ใส่ค่าของคุณ)
radar_y = 1530232.3207 # พิกัด UTM Y ของจุดกึ่งกลาง (ใส่ค่าของคุณ)
pixel_resolution = 300     # 1 พิกเซล = กี่เมตร (ตรวจสอบค่านี้อีกครั้ง)
shapefile_path = './../mapdata/Export_Output.shp' # ชื่อไฟล์ Shapefile ของคุณ
threshold = 10
district_names = [
    "ดุสิต",
    "หนองจอก",
    "พระนคร",
    "บางรัก",
    "บางเขน",
    "บางกะปิ",
    "ปทุมวัน",
    "ป้อมปราบศัตรูพ่าย",
    "พระโขนง",
    "มีนบุรี",
    "ลาดกระบัง",
    "ยานนาวา",
    "สัมพันธวงศ์",
    "พญาไท",
    "ธนบุรี",
    "บางกอกใหญ่",
    "ห้วยขวาง",
    "คลองสาน",
    "ตลิ่งชัน",
    "บางกอกน้อย",
    "บางขุนเทียน",
    "ภาษีเจริญ",
    "หนองแขม",
    "ราษฎร์บูรณะ",
    "บางพลัด",
    "ดินแดง",
    "บึงกุ่ม",
    "สาทร",
    "บางซื่อ",
    "จตุจักร",
    "บางคอแหลม",
    "ประเวศ",
    "คลองเตย",
    "สวนหลวง",
    "จอมทอง",
    "ดอนเมือง",
    "ราชเทวี",
    "ลาดพร้าว",
    "วัฒนา",
    "บางแค",
    "หลักสี่",
    "สายไหม",
    "คันนายาว",
    "สะพานสูง",
    "วังทองหลาง",
    "คลองสามวา",
    "บางนา",
    "ทวีวัฒนา",
    "ทุ่งครุ",
    "บางบอน"
]


target_colors = np.array([

    [195,   0,  85],
    [216,   0,  71],
    [224,   0,  85],
    [238,   0,   0],
    [252,  75,   0],
    [222, 152,   0],
    [230, 164,   0],
    [252, 214,   0],
    [216, 216,   0],
    [234, 218,   0],
    [238, 252,   0],
    [0, 243,   0],
    [0, 236,   0],
    [0, 214,  82],
    [0, 200,   0],
    [0, 197,   0],
    [0, 191,   0],
    [0, 176,   0],
    [0, 168,   0],

], dtype=np.uint8)

# dBZ values

values = np.array([

    66.5, 64.0, 61.5, 59.0,
    56.5, 54.0, 51.5, 49.0,
    46.5, 44.0, 41.5, 39.0,
    36.5, 34.0, 31.5, 29.0,
    26.5, 24.0, 21.5

])
left = 279
upper = 339

right = 279 + 270
lower = 339 + 260
# =========================
# LOAD DATA
# =========================
@cache
def generate_data(gif_path, verbose = False, mode = 'central'):
    if mode == 'central':
        left = 279
        upper = 339

        right = 279 + 270
        lower = 339 + 260
        radar_x = 699558.0797  # พิกัด UTM X ของจุดกึ่งกลาง (ใส่ค่าของคุณ)
        radar_y = 1530232.3207 # พิกัด UTM Y ของจุดกึ่งกลาง (ใส่ค่าของคุณ)
    elif mode == 'north':
        left = 449
        upper = 306

        right = 449 + 270
        lower = 306 + 260
        radar_x = 646930.0623
        radar_y = 1519140.397
    gif = Image.open(gif_path)

    gdf = gpd.read_file(shapefile_path)

    frames = [
        frame.copy()
        for frame in ImageSequence.Iterator(gif)
    ]

    print(f"Total frames: {len(frames)}")

    # =========================
    # CRS CHECK
    # =========================

    print("Shapefile CRS:", gdf.crs)

    # Example if needed
    # gdf = gdf.to_crs("EPSG:32647")

    # =========================
    # LOOP ALL FRAMES
    # =========================

    all_results = []
    data = []
    for frame_idx, frame in tqdm(enumerate(frames)):
        if verbose:
            print("=" * 80)
            print(f"Processing frame {frame_idx+1}")

        # =====================
        # RGB IMAGE
        # =====================

        frame_rgb = frame.convert("RGB")
        cropped_image = frame_rgb.crop((left, upper, right, lower))
        # resized_image = cropped_image.resize((224, 224))
        dir_frame_data = f"radar_caption/{mode}/history/2026/{os.path.splitext(os.path.basename(gif_path))[0]}/"
        os.makedirs(dir_frame_data, exist_ok=True)
        cropped_image.save(dir_frame_data +f"n_image{frame_idx + 1:03d}.png")
        if verbose:
            print(f"radar_caption/{mode}/history/2026/{os.path.splitext(os.path.basename(gif_path))[0]}/" +f"n_image{frame_idx + 1:03d}.png")
        image_np = np.array(frame_rgb)

        height, width, _ = image_np.shape

        # =====================
        # CREATE RAIN VALUE MAP
        # =====================

        rain_value_map = np.zeros(
            (height, width),
            dtype=np.float32
        )

        threshold = 70

        for color, value in zip(target_colors, values):

            diff = np.linalg.norm(
                image_np.astype(np.int16)
                - color.astype(np.int16),
                axis=2
            )

            matched = diff < threshold

            rain_value_map[matched] = value

        # =====================
        # REMOVE BLUE REGION
        # =====================

        hsv_image = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2HSV
        )

        blue_lower = np.array([90, 50, 50])
        blue_upper = np.array([140, 255, 255])

        blue_mask = cv2.inRange(
            hsv_image,
            blue_lower,
            blue_upper
        )

        rain_value_map[blue_mask > 0] = 0
        
        # =====================
        # CALCULATE EXTENT
        # =====================

        half_width_meter = (
            width * pixel_resolution
        ) / 2

        half_height_meter = (
            height * pixel_resolution
        ) / 2

        xmin = radar_x - half_width_meter
        xmax = radar_x + half_width_meter

        ymin = radar_y - half_height_meter
        ymax = radar_y + half_height_meter

        extent = [
            xmin,
            xmax,
            ymin,
            ymax
        ]

        # =====================
        # RESULT GDF
        # =====================

        rain_gdf = gdf.copy()

        rain_gdf["severity"] = 0.0
        rain_gdf["max_dbz"] = 0.0
        rain_gdf["rain_level"] = "ไม่มีฝน"
        rain_gdf["rain_pixels"] = 0

        # =====================
        # PROCESS EACH DISTRICT
        # =====================

        for idx, row in rain_gdf.iterrows():

            geom = row.geometry

            district_mask = np.zeros(
                (height, width),
                dtype=np.uint8
            )

            # =====================
            # HANDLE GEOMETRY
            # =====================

            if geom.geom_type == "Polygon":

                polygons = [geom]

            elif geom.geom_type == "MultiPolygon":

                polygons = geom.geoms

            else:

                continue

            # =====================
            # DRAW DISTRICT MASK
            # =====================

            for poly in polygons:

                exterior = np.array(
                    poly.exterior.coords
                )

                pixel_coords = []

                for x, y in exterior:

                    px = int(
                        (x - xmin)
                        / (xmax - xmin)
                        * width
                    )

                    py = int(
                        (ymax - y)
                        / (ymax - ymin)
                        * height
                    )

                    pixel_coords.append([px, py])

                pixel_coords = np.array(
                    pixel_coords,
                    dtype=np.int32
                )

                cv2.fillPoly(
                    district_mask,
                    [pixel_coords],
                    255
                )

            # =====================
            # EXTRACT DISTRICT RAIN
            # =====================

            district_values = rain_value_map[
                district_mask > 0
            ]

            district_values = district_values[
                district_values > 0
            ]

            rain_pixels = len(district_values)

            # =====================
            # CALCULATE SEVERITY
            # =====================

            if rain_pixels > 0:

                severity = np.mean(
                    district_values
                )

                max_dbz = np.max(
                    district_values
                )

            else:

                severity = 0

                max_dbz = 0

            # =====================
            # CLASSIFY RAIN LEVEL
            # =====================

            if severity < 10:

                rain_level = "ไม่มีฝน"
                rain_level_data = 0
            elif severity < 30:

                rain_level = "ฝนเล็กน้อย"
                rain_level_data = 1
            elif severity < 45:

                rain_level = "ฝนปานกลาง"
                rain_level_data = 2
            else:

                rain_level = "ฝนหนัก"
                rain_level_data = 3
            # =====================
            # SAVE RESULT
            # =====================

            rain_gdf.at[idx, "severity"] = severity

            rain_gdf.at[idx, "max_dbz"] = max_dbz

            rain_gdf.at[idx, "rain_level"] = rain_level
            
            rain_gdf.at[idx, "rain_pixels"] = rain_pixels
            
            rain_gdf.at[idx, "rain_level_data"] = rain_level_data
            # =====================
            # SAVE TABLE
            # =====================

            all_results.append({

                "frame": frame_idx,

                "district": row["DISTRICT_T"],

                "severity": severity,

                "max_dbz": max_dbz,

                "rain_pixels": rain_pixels,

                "rain_level": rain_level

            })

        # =====================
        # COLOR FUNCTION
        # =====================

        def severity_color(level):

            if level == "ฝนหนัก":
                return "red"

            elif level == "ฝนปานกลาง":
                return "orange"

            elif level == "ฝนเล็กน้อย":
                return "yellow"

            return "none"

        # =====================
        # PLOT
        # =====================
        if verbose:
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(18, 9)
            )

            # =====================
            # ORIGINAL IMAGE
            # =====================

            axes[0].imshow(
                image_np,
                extent=extent,
                origin="upper"
            )

            rain_gdf.plot(
                ax=axes[0],
                facecolor=rain_gdf["rain_level"].map(
                    severity_color
                ),
                edgecolor="black",
                linewidth=1,
                alpha=0.5
            )

            axes[0].set_title(
                f"Radar Rain Severity {frame_idx+1}"
            )

            # =====================
            # dBZ MAP
            # =====================

            im = axes[1].imshow(
                rain_value_map,
                cmap="turbo",
                extent=extent,
                origin="upper"
            )

            rain_gdf.boundary.plot(
                ax=axes[1],
                edgecolor="black",
                linewidth=0.5
            )

            
        
            axes[1].set_title(
                f"dBZ Intensity Map {frame_idx+1}"
            )
            plt.colorbar(
                im,
                ax=axes[1],
                label="dBZ"
            )

            plt.tight_layout()

            plt.show()

            # =====================
            # PRINT RESULT
            # =====================

            print("\nRain Summary:\n")

        raining = rain_gdf[
            rain_gdf["rain_level"] != "ไม่มีฝน"
        ]
        example_data = {"original_image":gif_path,"image_path": f"radar_caption/{mode}/history/2026/{os.path.splitext(os.path.basename(gif_path))[0]}/" + f"n_image{frame_idx+1:03d}.png"}
        for n in range(len(district_names)):
                example_data[f'{district_names[n]}'] = 0
        for _, row in raining.iterrows():
            
            example_data[row['DISTRICT_T']] = row['rain_level_data']
            if verbose:
                print(
                    f"{row['DISTRICT_T']:20}"
                    f"| {row['rain_level']:12}"
                    f"| severity={row['severity']:.2f}"
                    f"| max={row['max_dbz']:.2f}"
                    f"| pixels={row['rain_pixels']}"
                )
        data.append(example_data)
    # =========================
    # EXPORT CSV
    # =========================

    result_df = pd.DataFrame(all_results)

    datadf = pd.DataFrame(data)
    return datadf, rain_gdf
# datadf.to_csv("train_data.csv", index=False)