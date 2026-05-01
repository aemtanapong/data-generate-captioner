import cv2
import numpy as np
import imageio
import io
from PIL import Image
from scipy.spatial.distance import cdist
import time

def data_radar(callback=None):
    steps = [
        "โหลด GIF",
        "แยกเฟรม",
        "คำนวณทิศทาง",
        "วิเคราะห์ฝน",
        "สร้างผลลัพธ์"
    ]

    for i, step in enumerate(steps):
        time.sleep(1)

        if callback:
            callback(step, (i + 1) / len(steps))

    return "done"
# ==================================================
# CONFIG
# ==================================================
target_colors = np.array([
    [252,252,255],[252,219,255],[252,202,255],[252,139,255],
    [252,0,255],[195,0,85],[216,0,71],[224,0,85],
    [238,0,0],[252,75,0],[222,152,0],[230,164,0],
    [252,214,0],[216,216,0],[234,218,0],[238,252,0],
    [0,243,0],[0,236,0],[0,214,82],[0,200,0],
    [0,197,0],[0,191,0],[0,176,0],[0,168,0],
    [0,0,255]
], dtype=np.uint8)

values = np.array([
    66.5,64.0,61.5,59.0,56.5,54.0,51.5,49.0,46.5,
    44.0,41.5,39.0,36.5,34.0,31.5,29.0,26.5,24.0,
    21.5,19.0,16.5,14.0,11.5,10.0,9.5
])

threshold = 60


# ==================================================
# BASIC OPS
# ==================================================
def min_pooling(img, ksize=3):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(img, kernel)

def max_pooling(img, ksize=3):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(img, kernel)

# ==================================================
# EXTRACT RADAR COLORS
# ==================================================
def extract_radar_frame(img):
    h, w, _ = img.shape
    pixels = img.reshape(-1, 3)

    mask = np.zeros(len(pixels), dtype=bool)

    for color in target_colors:
        dist = np.linalg.norm(pixels - color, axis=1)
        mask |= dist < threshold

    mask = mask.reshape(h, w)

    result = img.copy()
    result[~mask] = 255
    return result


# ==================================================
# COLOR -> DBZ
# ==================================================
def map_to_dbz(image):
    h, w, _ = image.shape
    pixels = image.reshape(-1, 3)

    dist = cdist(pixels, target_colors)
    idx = np.argmin(dist, axis=1)

    return values[idx].reshape(h, w)


# ==================================================
# QUANTIZE
# ==================================================
def quantize_dbz(dbz):
    levels = np.zeros_like(dbz, dtype=np.uint8)

    levels[(dbz >= 10) & (dbz < 20)] = 1
    levels[(dbz >= 20) & (dbz < 30)] = 2
    levels[(dbz >= 30) & (dbz < 40)] = 3
    levels[(dbz >= 40)] = 4

    return levels


# ==================================================
# CLUSTERS
# ==================================================
def get_clusters(levels):
    clusters = []

    for lvl in np.unique(levels):
        if lvl == 0:
            continue

        mask = (levels == lvl).astype(np.uint8) * 255
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]

            if area < 100:
                continue

            clusters.append({
                "bbox": (x, y, w, h),
                "centroid": centroids[i],
                "level": int(lvl),
                "mask": (labels == i)
            })

    return clusters


# ==================================================
# FLOW
# ==================================================
def compute_flow(prev_gray, curr_gray):
    return cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )


def get_cluster_direction(flow, cluster_mask):
    ys, xs = np.where(cluster_mask)

    if len(xs) < 20:
        return None

    fx = flow[ys, xs, 0]
    fy = flow[ys, xs, 1]

    mean_fx = np.mean(fx)
    mean_fy = np.mean(fy)

    angle = np.degrees(np.arctan2(mean_fy, mean_fx))
    speed = np.sqrt(mean_fx**2 + mean_fy**2)

    return angle, speed


# ==================================================
# DRAW
# ==================================================
def draw_clusters_on_frame(frame, clusters, average_angle):
    img = frame.copy()
    number_data = 0
    for c in clusters:
        x, y, w, h = c["bbox"]
        cx, cy = c["centroid"]
        level = c["level"]

        x, y, w, h = map(int, [x, y, w, h])
        cx, cy = map(int, [cx, cy])

        if (w > 300 and h > 300) or level not in [2]:
            continue

        color_map = {
            1:(0,255,0),
            2:(0,200,255),
            3:(0,100,255),
            4:(0,0,255)
        }

        color = color_map.get(level, (255,255,255))

        cv2.putText(
            img,
            f"L{level}",
            (x, y-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

        dx = int(40 * np.cos(np.radians(average_angle)))
        dy = int(40 * np.sin(np.radians(average_angle)))

        move_caption_data = get_direction_from_points(cx, cy, cx+dx, cy+dy)
        number_data +=1
        cv2.arrowedLine(
            img,
            (cx, cy),
            (cx+dx, cy+dy),
            (221,81,62),
            3
        )
    try:
        return img, move_caption_data
    except:
        print("Error")
        return img, ("-", "ไม่มี")

def draw(frame, clusters, flow):
    out = frame.copy()

    for c in clusters:
        x, y, w, h = c["bbox"]
        cx, cy = map(int, c["centroid"])

        cv2.rectangle(out, (x,y), (x+w,y+h), (0,255,0), 2)

        if flow is not None:
            res = get_cluster_direction(flow, c["mask"])
            if res is not None:
                angle, speed = res

                dx = int(40 * np.cos(np.radians(angle)))
                dy = int(40 * np.sin(np.radians(angle)))

                cv2.arrowedLine(out, (cx,cy), (cx+dx, cy+dy), (255,0,0), 2)

                cv2.putText(out,
                            f"L{c['level']} {angle:.1f}",
                            (x-5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.25, (0,0,0), 1)

    return out
# ==================================================
# MAIN (NO FILE SAVE / SAME OLD RESULT)
# ==================================================
def get_direction_from_points(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1

    # OpenCV: y ลง = positive → ต้อง flip เพื่อให้ตรงกับเข็มทิศ
    angle = np.degrees(np.arctan2(-dy, dx))
    angle = (angle + 360) % 360
    if 337.5 <= angle or angle < 22.5:
        direction = "ทิศตะวันออก"
    elif 22.5 <= angle < 67.5:
        direction = "ทิศตะวันออกเฉียงเหนือ"
    elif 67.5 <= angle < 112.5:
        direction = "ทิศเหนือ"
    elif 112.5 <= angle < 157.5:
        direction = "ทิศตะวันตกเฉียงเหนือ"
    elif 157.5 <= angle < 202.5:
        direction = "ทิศตะวันตก"
    elif 202.5 <= angle < 247.5:
        direction = "ทิศตะวันตกเฉียงใต้"
    elif 247.5 <= angle < 292.5:
        direction = "ทิศใต้"
    else:
        direction = "ทิศตะวันออกเฉียงใต้"

    return angle, direction
MAIN_DIRECTION_DEGREE = None
MOVE_CAPTION = None
def radar_pipeline(input_gif_path, update=None):
    raw_frames = []
    processed_frames = []
    output_frames = []
    # -----------------------------------------
    # READ ORIGINAL GIF
    # -----------------------------------------
    if update: update("🖼️ กำลัง Upload ไฟล์...", 0.1)
    with Image.open(input_gif_path) as gif:
        for i in range(gif.n_frames):
            gif.seek(i)

            frame = np.array(gif.convert("RGB"))
            raw_frames.append(frame.copy())

            # same as old code
            cv2.rectangle(frame, (0, 0), (100, 970), (255,255,255), -1)
            cv2.rectangle(frame, (700,600), (1300,970), (255,255,255), -1)

            frame = extract_radar_frame(frame)
            frame[np.all(frame == 255, axis=-1)] = 0

            frame = max_pooling(frame, 3)
            frame = min_pooling(frame, 3)
            frame = min_pooling(frame, 3)
            frame = max_pooling(frame, 10)

            processed_frames.append(frame)
    if update: update("🎞️ สร้าง GIF ชั่วคราว...", 0.3)
    # -----------------------------------------
    # IMPORTANT:
    # simulate old temp.gif in memory
    # -----------------------------------------
    gif_buffer = io.BytesIO()

    imageio.mimsave(
        gif_buffer,
        processed_frames,
        format="GIF",
        fps=10,
        loop=0
    )

    gif_buffer.seek(0)

    frames = imageio.mimread(gif_buffer, format="GIF")

    # -----------------------------------------
    # FLOW ANALYSIS
    # -----------------------------------------
    if update: update("🧭 วิเคราะห์การเคลื่อนที่...", 0.6)
    prev_gray = None
    results = []

    for frame in frames:
        frame = np.array(frame)

        if frame.shape[-1] == 4:
            frame = frame[:, :, :3]

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        dbz = map_to_dbz(frame)
        levels = quantize_dbz(dbz)
        clusters = get_clusters(levels)

        flow = None
        if prev_gray is not None:
            flow = compute_flow(prev_gray, gray)

        frame_result = []

        for c in clusters:
            direction = None
            if flow is not None:
                direction = get_cluster_direction(flow, c["mask"])

            frame_result.append({
                "bbox": c["bbox"],
                "centroid": c["centroid"],
                "level": c["level"],
                "direction": direction
            })
            if c["level"] == 1 and direction: 
                MAIN_DIRECTION_DEGREE = direction[0]
           
        results.append(frame_result)
        visual_data = draw(frame, clusters, flow)
        output_frames.append(visual_data)

        prev_gray = gray

        
    # -----------------------------------------
    # AVERAGE ANGLE
    # -----------------------------------------
    all_angles = []
    cloud_speed = [] 
    for frame_results in results:
        for cluster in frame_results:
            if cluster["direction"] is not None:
                angle, speed = cluster["direction"]
                cloud_speed.append(speed)
                all_angles.append(angle)

    # average_angle = float(np.mean(all_angles))

    average_cloud_speed = float(np.mean(cloud_speed))
    # print("Average rain average_cloud_speed : ",average_cloud_speed)
    # -----------------------------------------
    # RENDER OUTPUT IMAGE
    # -----------------------------------------
    if update: update("🖼️ สร้างภาพผลลัพธ์...", 1.0)
    render_data, move_caption_data = draw_clusters_on_frame(
        raw_frames[0],
        results[1],
        MAIN_DIRECTION_DEGREE
    )
    # if update: update("✅ เสร็จสิ้น", 1.0)
    # print("render_data : ",render_data.shape)
    # imageio.mimsave("animation.gif", output_frames, duration=1000, loop=0)
    # print("Direction : ",move_caption_data)
    
    return move_caption_data, render_data, output_frames


# ==================================================
# USE
# ==================================================
if __name__ == "__main__":

    average_angle, render_data, output_frames = radar_pipeline("C:/Users/BMA_01/Documents/ขอข้อมูล/2026-04-22-จัดการ NLG/example/n006.gif")

    print("average_angle =", average_angle)

    cv2.imwrite(
        "example/result.png",
        cv2.cvtColor(render_data, cv2.COLOR_RGB2BGR)
    )