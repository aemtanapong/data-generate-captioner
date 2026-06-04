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
from PIL import Image
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping
import matplotlib.pyplot as plt
import cv2 # Ensure cv2 is imported for color conversion
import pandas as pd # Ensure pandas is imported for DataFrame
from functools import cache
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
@cache
def generate_data(gif_path, mode = 'central'):
    if mode == 'central':

        radar_x = 699558.0797  # พิกัด UTM X ของจุดกึ่งกลาง (ใส่ค่าของคุณ)
        radar_y = 1530232.3207 # พิกัด UTM Y ของจุดกึ่งกลาง (ใส่ค่าของคุณ)
    elif mode == 'north':

        radar_x = 646930.0623
        radar_y = 1519140.397
    # 1. Get file name with extension
    filename = os.path.basename(gif_path)
    print(filename)
    # Output: report.pdf

    # 2. Get file name WITHOUT extension
    filename_only = os.path.splitext(filename)[0]
    print(filename_only)
    # Output base folder for all cropped districts
    save_base_folder = f"cropped_districts_by_frame/{mode}/{filename_only}"
    os.makedirs(save_base_folder, exist_ok=True)

    # Define the Coordinate Reference System (CRS) - Assuming UTM Zone 47N for Thailand
    # This should be consistent with previous cells, ensuring gdf is in this CRS.
    image_crs = 'EPSG:32647'

    # Load the shapefile if not already in the kernel (should be from 3d8243d2)
    # Added for robustness, but assumes gdf might be available.
    if 'gdf' not in locals():
        gdf = gpd.read_file(shapefile_path)
        if gdf.crs != image_crs:
            print(f"Reprojecting shapefile from {gdf.crs} to {image_crs}")
            gdf = gdf.to_crs(image_crs)

    gif = Image.open(gif_path)
    frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]


    print(f"Starting extraction for {len(frames)} frames and {len(gdf)} districts.")

    # Initialize list to store all rain severity results
    all_results = []

    for frame_idx, frame_pil in enumerate(frames):
        print(f"\n--- Processing Frame {frame_idx + 1}/{len(frames)} ---")

        # Create a subfolder for each frame
        frame_save_folder = os.path.join(save_base_folder, f"frame_{frame_idx:03d}")
        os.makedirs(frame_save_folder, exist_ok=True)

        # Convert PIL Image frame to NumPy array (full frame)
        frame_np = np.array(frame_pil)

        img_height, img_width, num_bands = frame_np.shape

        # Calculate top-left corner coordinates (west, north)
        # The radar_x, radar_y are center coordinates.
        half_width_meters = (img_width / 2) * pixel_resolution
        half_height_meters = (img_height / 2) * pixel_resolution

        west = radar_x - half_width_meters
        north = radar_y + half_height_meters

        # Define the affine transform for the image
        transform = rasterio.transform.from_origin(west, north, pixel_resolution, pixel_resolution)

        # Create a MemoryFile for the current frame to treat it as a rasterio dataset
        with rasterio.MemoryFile() as memfile:
            with memfile.open(
                driver='GTiff',
                height=img_height,
                width=img_width,
                count=num_bands,
                dtype=frame_np.dtype,
                crs=image_crs,
                transform=transform
            ) as src:
                # Write the actual image data to the MemoryFile
                src.write(frame_np.transpose((2, 0, 1))) # rasterio expects (bands, height, width)

                # --- Start Rain Severity Calculation for the FULL FRAME ---
                # Convert the full frame to RGB and NumPy array for rain calculations
                frame_rgb_full = frame_pil.convert("RGB")
                image_np_full_frame = np.array(frame_rgb_full)
                full_frame_height, full_frame_width, _ = image_np_full_frame.shape

                # Create rain value map (dBZ values)
                rain_value_map = np.zeros((full_frame_height, full_frame_width), dtype=np.float32)
                color_match_threshold = 70 # Specific threshold for color matching from pJCA04qgL0LK

                for color, value in zip(target_colors, values):
                    diff = np.linalg.norm(image_np_full_frame.astype(np.int16) - color.astype(np.int16), axis=2)
                    matched = diff < color_match_threshold
                    rain_value_map[matched] = value

                # Remove blue regions (e.g., radar artifacts or background)
                hsv_image = cv2.cvtColor(image_np_full_frame, cv2.COLOR_RGB2HSV)
                blue_lower = np.array([90, 50, 50])
                blue_upper = np.array([140, 255, 255])
                blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)
                rain_value_map[blue_mask > 0] = 0

                # Calculate extent of the full frame for geographic to pixel conversion
                half_width_meter_full = (full_frame_width / 2) * pixel_resolution
                half_height_meter_full = (full_frame_height / 2) * pixel_resolution
                xmin_full_frame = radar_x - half_width_meter_full
                xmax_full_frame = radar_x + half_width_meter_full
                ymin_full_frame = radar_y - half_height_meter_full
                ymax_full_frame = radar_y + half_height_meter_full
                # --- End Rain Severity Calculation for the FULL FRAME ---

                for district_idx, row in gdf.iterrows():
                    district_name = row['DISTRICT_T'] # Assuming 'DISTRICT_T' column has Thai names
                    district_geometry = [mapping(row.geometry)]

                    try:
                        # Crop the radar image using the district geometry
                        out_image, out_transform = mask(src, district_geometry, crop=True)

                        # Check if the cropped image is empty or fully transparent
                        is_empty = False
                        if out_image.size == 0: # Check for completely empty array
                            is_empty = True
                        elif num_bands == 4: # RGBA image
                            # Check if all RGB values are near zero and alpha is near zero for all pixels
                            if np.all(out_image[:3, :, :] < 5) and np.all(out_image[3, :, :] < 5):
                                is_empty = True
                        elif num_bands == 1: # Grayscale image
                            if np.all(out_image < 5):
                                is_empty = True

                        if is_empty:
                            # print(f"  District '{district_name}' in Frame {frame_idx:03d} resulted in an empty image (no overlap or data).")
                            pass # Don't save empty image, but still process severity as 0
                        else:
                            # Convert the cropped image (bands, height, width) to (height, width, bands) for PIL
                            cropped_image_np = np.transpose(out_image, (1, 2, 0))

                            # Convert to PIL Image
                            cropped_image_pil = Image.fromarray(cropped_image_np)

                            # Save the cropped image
                            output_filename = os.path.join(frame_save_folder, f"{district_name}.png")
                            # frame_rgb_np = cropped_image_pil.convert("RGB") # This line is not needed for the original cropped_image_pil.save
                            # image_np = np.array(frame_rgb_np) # This image_np is for the cropped image, not the full frame

                            # height, width, _ = image_np.shape # These refer to the cropped image dimensions
                            # print(height, width)

                            cropped_image_pil.save(output_filename)
                            # print(f"  Saved '{district_name}' from Frame {frame_idx:03d} to {output_filename}")

                    except ValueError as e:
                        # This typically happens if the mask operation results in no valid pixels
                        # print(f"  District '{district_name}' in Frame {frame_idx:03d} resulted in an empty image (no overlap or data). Error: {e}")
                        is_empty = True # Treat as empty for severity calculation
                    except Exception as e:
                        print(f"  Error processing district '{district_name}' in Frame {frame_idx:03d}: {e}")
                        is_empty = True # Treat as empty for severity calculation

                    # --- Start Rain Severity Calculation for the current district --- (using full frame's rain_value_map)
                    district_mask_for_rain = np.zeros((full_frame_height, full_frame_width), dtype=np.uint8)
                    current_geom = row.geometry

                    polygons_to_mask = []
                    if current_geom.geom_type == "Polygon":
                        polygons_to_mask = [current_geom]
                    elif current_geom.geom_type == "MultiPolygon":
                        polygons_to_mask = current_geom.geoms

                    if polygons_to_mask:
                        for poly in polygons_to_mask:
                            exterior = np.array(poly.exterior.coords)
                            pixel_coords = []
                            for x, y in exterior:
                                px = int((x - xmin_full_frame) / (xmax_full_frame - xmin_full_frame) * full_frame_width)
                                py = int((ymax_full_frame - y) / (ymax_full_frame - ymin_full_frame) * full_frame_height)
                                pixel_coords.append([px, py])

                            pixel_coords = np.array(pixel_coords, dtype=np.int32)
                            cv2.fillPoly(district_mask_for_rain, [pixel_coords], 255)

                        district_values = rain_value_map[district_mask_for_rain > 0]
                        district_values = district_values[district_values > 0] # Only consider pixels with rain

                        rain_pixels = len(district_values)

                        if rain_pixels > 0:
                            severity = np.mean(district_values)
                            max_dbz = np.max(district_values)
                        else:
                            severity = 0.0
                            max_dbz = 0.0
                    else:
                        # Handle non-polygon geometries or empty polygons for rain calculation
                        severity = 0.0
                        max_dbz = 0.0
                        rain_pixels = 0

                    # Classify rain level
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

                    all_results.append({
                        "original_image_path": output_filename,
                        "frame": frame_idx,
                        "district": row["DISTRICT_T"],
                        "severity": severity,
                        "max_dbz": max_dbz,
                        "rain_pixels": rain_pixels,
                        "rain_level": rain_level,
                        "rain_level_data": rain_level_data
                    })
                    # --- End Rain Severity Calculation for the current district ---

    print("\nAll frames and districts processed.")

    # Convert all_results to DataFrame after processing all frames and districts
    df_rain_severity = pd.DataFrame(all_results)
    print("\n--- Rain Severity Results per District and Frame ---")
    return df_rain_severity

if __name__ == "__main__":
    print(generate_data(gif_path))
