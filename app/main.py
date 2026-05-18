import streamlit as st
import pandas as pd
import numpy as np
import base64
import data_moving as data_radar
import data_intensity_v3 as data_intensity_v3
import matplotlib.pyplot as plt
import cv2
import imageio
import io
import geopandas as gpd
import time
import model_main
import plotly.express as px
import random
from datetime import datetime
import re
import data_coverage
import data_district_rainv3
import data_prediction
from datetime import datetime
import uuid
plt.rcParams['font.family'] = 'Tahoma'

st.title("Radar Animation")
st.set_page_config( page_title="Radar Detail")
def render_district_grid_html(
    title,
    districts,
    color1,
    color2,
    emoji="🌧️"
):

    # =====================================================
    # UNIQUE CLASS กัน style ทับกันหลาย component
    # =====================================================
    uid = str(uuid.uuid4()).replace("-", "")

    container_class = f"rain_container_{uid}"
    title_class = f"rain_title_{uid}"
    grid_class = f"district_grid_{uid}"
    card_class = f"district_card_{uid}"

    if not districts:
        districts = ["ไม่มีข้อมูล"]

    cards_html = ""

    for district in districts:

        cards_html += f"""
        <div class="{card_class}">
            {district}
        </div>
        """

    html = f"""
    <style>

    .{container_class} {{
        padding:20px;
        border-radius:24px;

        background:linear-gradient(
            135deg,
            {color1},
            {color2}
        );

        margin-bottom:24px;

        box-shadow:0 8px 24px rgba(0,0,0,0.18);
    }}

    .{title_class} {{
        color:white;
        font-size:28px;
        font-weight:800;
        margin-bottom:18px;
    }}

    .{grid_class} {{
        display:grid;

        grid-template-columns:
            repeat(auto-fill,minmax(180px,1fr));

        gap:14px;
    }}

    .{card_class} {{

        background:rgba(255,255,255,0.15);

        border:1px solid rgba(255,255,255,0.18);

        backdrop-filter:blur(6px);

        padding:14px;

        border-radius:16px;

        color:white;

        font-size:17px;

        font-weight:700;

        text-align:center;

        transition:all 0.2s ease;
    }}

    .{card_class}:hover {{

        transform:translateY(-3px);

        background:rgba(255,255,255,0.22);
    }}

    </style>

    <div class="{container_class}">

        <div class="{title_class}">
            {emoji} {title}
        </div>

        <div class="{grid_class}">
            {cards_html}
        </div>

    </div>
    """

    st.html(html)
with st.status("generate caption", expanded=True) as n:
    uploaded_file = st.file_uploader("Upload Radar GIF", type=["gif", "webp"])
    # =========================================================
    # MODULE SELECTION UI
    # =========================================================
    st.subheader("🧠 Select Analysis Modules")

    col1, col_3 = st.columns(2)

    with col1:
        module_direction_data = st.checkbox("📊 Direction Calculator", value=True)
        module_coverage_data = st.checkbox("🗺️ Coverage คำนวนคลอบคลุมฝน", value=True)

    with col_3:
        module_district_data = st.checkbox("📈 Rain District Detector", value=True)
        module_prediction_data = st.checkbox("🖼️ Rain Prediction", value=True)
def ui_update(text, p):
    status.write(text)
    progress.progress(p)
def rain_district_name_ui_update(text, p):
    rain_district_name_status.write(text)
    progress.progress(p)
def rain_ui_update(text, p):
    rain_status.write(text)
    progress.progress(p)
def show_gif(gif_bytes, width=500):
    b64 = base64.b64encode(gif_bytes).decode()
    st.markdown(
        f'<div style="text-align:center;">'
        f'<img src="data:image/gif;base64,{b64}" width="{width}">'
        f'</div>',
        unsafe_allow_html=True
    )

    
if uploaded_file is not None:
    gif_bytes = uploaded_file.read()

  
    if module_direction_data:
        st.success("Radar uploaded ✅")
        progress = st.progress(0)
        with st.status("🚀 Radar Animation", expanded=True) as status:
            # ui_update("กำลัง Upload ไฟล์",0) 
            print("data")
            # 👉 Replace with real computed values later
            direction = "NE"
            with st.spinner("กำลังประมวลผล..."):
                move_caption_data, render_data, output_frame = data_radar.radar_pipeline(
                    uploaded_file,
                    update=ui_update
                )
                print(direction)
                # -------------------------
                # Row 1: Information
                # -------------------------
                st.subheader("📊 Radar Information")
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("🧭 Direction", move_caption_data[1])

                with col2:
                    try:
                        st.metric("📐 Degree", f"{move_caption_data[0]:.0f}°")
                    except:
                        st.metric("📐 Degree", f"{move_caption_data[0]}°")

                st.divider()

                # -------------------------
                # Row 2: Radar A
                # -------------------------
                st.subheader("Radar (Original)")
                show_gif(gif_bytes)
                

                # -------------------------
                # Row 3: Radar B
                # -------------------------
                st.subheader("Radar (Prediction)")
                # แปลงเป็น RGB ถ้ามาจาก OpenCV
                render_rgb = cv2.cvtColor(render_data, cv2.COLOR_BGR2RGB)

                # แปลงเป็น base64
                _, buffer = cv2.imencode(".png", render_rgb)
                b64 = base64.b64encode(buffer).decode()

                st.markdown(
                    f"""
                    <div style="text-align:center;">
                        <img src="data:image/png;base64,{b64}" width="500">
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # -------------------------
                # Row 4: Radar C
                # -------------------------
                st.subheader("Radar Detection Cluster")

                gif_buffer = io.BytesIO()

                imageio.mimsave(
                    gif_buffer,
                    output_frame,
                    format="GIF",
                    fps=10,
                    loop=0
                )

                gif_buffer.seek(0)

                st.image(gif_buffer.read(), caption="Radar Animation")
                # show_gif(gif_bytes)
                # data_intensity_v3.get_district_name_rain(gif_buffer)
                print("data")
                st.success("Done!")
    if module_coverage_data:
        with st.status("Coverage คำนวนคลอบคลุมฝน", expanded=True) as status:
            rgb, mask_vis ,cropped, rain_vis, highlight, plot_img, data_coverage_and_trend_rain = data_coverage.get_data(uploaded_file)
            average_coverage, trend = data_coverage_and_trend_rain
            print("data")
            st.subheader("📊 Radar Information")
            coverage_1 ,trend_1 = st.columns(2)
            with coverage_1:
                st.metric("🧭 Coverage", f"{average_coverage:.2f}%")
            with trend_1:
                if trend == 'increasing':
                    trend_text = 'เพิ่มขึ้น'
                elif trend == 'decreasing':
                    trend_text = 'ลดลง'
                else:
                    trend_text = 'คงที่'
                st.metric("🧭 Trend", f"{trend_text}")
            descibe_coverage_mask, rain_data_debug = st.columns(2)
            
            with descibe_coverage_mask:
                st.image(rgb, caption="RGB",  use_container_width=True)
                st.image(mask_vis, caption="Mask", use_container_width=True)
                st.image(cropped, caption="Cropped", use_container_width=True)
            with rain_data_debug:
                st.image(rain_vis, caption="Rain Visualization", use_container_width=True)
                st.image(highlight, caption="Highlight", use_container_width=True)
                st.image(plot_img, caption="plot_img", use_container_width=True)
            
            st.success(f"Done!")
    
    if module_district_data:
        progress = st.progress(0)
        with st.status("🌧️ Rain District Analysis", expanded=True) as rain_district_name_status:
            with st.spinner("กำลังประมวลผล..."):
                plot_img, rain_district_data = data_district_rainv3.get_data(gif_bytes, update = rain_district_name_ui_update)

            # =========================
            # HEADER
            # =========================
            st.subheader("📊 Rain District Overview")

            st.image(
                plot_img,
                caption="Rain Intensity Map",
                use_container_width=True
            )

            # =========================
            # COUNTS
            # =========================
            heavy_n = len(rain_district_data.get("heavy", []))
            medium_n = len(rain_district_data.get("medium", []))
            light_n = len(rain_district_data.get("light", []))
            none_n = len(rain_district_data.get("none", []))
            c1, c2, c3, c4 = st.columns(4)

            c1.metric("🔴 Heavy", heavy_n)
            c2.metric("🟠 Medium", medium_n)
            c3.metric("🟢 Light", light_n)
            c4.metric("🔵 None", none_n)
            st.divider()

            rain_district_number, coverage_district_grid = st.columns(2)
            rain_district_number.metric("📊 เขตที่มีฝน", heavy_n + medium_n + light_n)
            coverage_number_text = (heavy_n + medium_n + light_n) / 50.0 * 100
            coverage_district_grid.metric("🧭 Coverage District Grid", f"{coverage_number_text}%")
            st.divider()

            # =========================
            # TABS
            # =========================
            tab1, tab2, tab3, tab4 = st.tabs([
                "🔴 Heavy Rain",
                "🟠 Medium Rain",
                "🟢 Light Rain",
                "🔵 No Rain"

            ])

            # -------------------------
            # HEAVY
            # -------------------------
            with tab1:

                heavy_list = rain_district_data.get("heavy", [])

                if heavy_list:

                    st.error(f"พบ {len(heavy_list)} เขต")

                    cols = st.columns(3)

                    for i, district in enumerate(heavy_list):
                        cols[i % 3].markdown(
                            f"""
                            <div style="
                                padding:10px;
                                border-radius:14px;
                                background:linear-gradient(135deg, #ff5252, #d50000);
                                color:white;
                                margin-bottom:10px;
                                text-align:center;
                                font-weight:700;
                                font-size:16px;
                                box-shadow:0 4px 10px rgba(0,0,0,0.15);
                            ">
                                ⛈️ {district}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.success("ไม่พบพื้นที่ฝนหนัก")

            # -------------------------
            # MEDIUM
            # -------------------------
            with tab2:

                medium_list = rain_district_data.get("medium", [])

                if medium_list:

                    st.warning(f"พบ {len(medium_list)} เขต")

                    cols = st.columns(3)

                    for i, district in enumerate(medium_list):
                        cols[i % 3].markdown(
                            f"""
                            <div style="
                                padding:10px;
                                border-radius:14px;
                                background:linear-gradient(135deg, #ffb74d, #fb8c00);
                                color:white;
                                margin-bottom:10px;
                                text-align:center;
                                font-weight:700;
                                font-size:16px;
                                box-shadow:0 4px 10px rgba(0,0,0,0.15);
                            ">
                                🌧️ {district}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.success("ไม่พบพื้นที่ฝนปานกลาง")

            # -------------------------
            # LIGHT
            # -------------------------
            with tab3:

                light_list = rain_district_data.get("light", [])

                if light_list:

                    st.info(f"พบ {len(light_list)} เขต")

                    cols = st.columns(3)

                    for i, district in enumerate(light_list):
                        cols[i % 3].markdown(
                            f"""
                            <div style="
                                padding:10px;
                                border-radius:14px;
                                background:linear-gradient(135deg, #66bb6a, #2e7d32);
                                color:white;
                                margin-bottom:10px;
                                text-align:center;
                                font-weight:700;
                                font-size:16px;
                                box-shadow:0 4px 10px rgba(0,0,0,0.15);
                            ">
                                🌦️ {district}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.success("ไม่พบพื้นที่ฝนเบา")
            # =========================
            # NO RAIN
            # =========================
            with tab4:

                no_rain_list = rain_district_data.get("none", [])

                if no_rain_list:

                    st.success(f"พบ {len(no_rain_list)} เขต")

                    cols = st.columns(3)

                    for i, district in enumerate(no_rain_list):
                        cols[i % 3].markdown(
                            f"""
                            <div style="
                                padding:10px;
                                border-radius:14px;
                                background:linear-gradient(135deg, #64b5f6, #42a5f5);
                                color:white;
                                margin-bottom:10px;
                                text-align:center;
                                font-weight:700;
                                font-size:16px;
                                box-shadow:0 4px 10px rgba(0,0,0,0.15);
                            ">
                                ☀️ {district}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info("ทุกพื้นที่มีฝน")
    

    if module_prediction_data:
        progress = st.progress(0)
        with st.status("🚀 Prediction", expanded=True) as rain_status:
            with st.spinner("กำลังประมวลผล..."):
                print(f"Prediction ({move_caption_data[0]})")
                image, (max_coverage_value, average_coverage_value, coverage_percentile_value), gif_prediction, (max_n_coverage_value, average_n_coverage_value, coverage_n_percentile_value), num_district_data = data_prediction.get_data(gif_bytes, move_caption_data[0], update = rain_ui_update)
                st.subheader("🌧️ Current Radar Information")
                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "📐 Avg Coverage",
                    f"{average_coverage_value:.1f}%"
                )

                c2.metric(
                    "📈 Peak Coverage",
                    f"{max_coverage_value:.1f}%"
                )

                c3.metric(
                    "⚡ Coverage P90",
                    f"{coverage_percentile_value:.1f}%"
                )
                    
                # st.write(coverage_value)
                st.image(image)

                st.subheader("📊 Prediction Radar Information")
                average_coverage, maxmimum_coverage, coverage_percentile = st.columns(3)
                with average_coverage:
                    st.metric("📐 Average Coverage : ", f"{max_n_coverage_value:.1f}%")
                with maxmimum_coverage:
                    st.metric("📐 Maximum Coverage : ", f"{average_n_coverage_value:.1f}%")
                with coverage_percentile:
                    st.metric("📐 Maximum Coverage : ", f"{coverage_n_percentile_value:.1f}%")
                st.image(gif_prediction)

                persistent_n = len(num_district_data["🚨 ฝนต่อเนื่อง"])
                incoming_n = len(num_district_data["🌧️ ฝนเริ่มเคลื่อนเข้า"])
                watch_n = len(num_district_data["👀 เฝ้าระวัง"])

                st.subheader("🚨 Rain Notify")

                a1, a2, a3 = st.columns(3)

                a1.metric("🚨 Persistent", persistent_n)
                a2.metric("🌧️ Incoming", incoming_n)
                a3.metric("👀 Watch", watch_n)

                top_districts = sorted(
                    num_district_data.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                st.subheader("📍 High Risk Districts")

                render_district_grid_html(
                    title="ฝนต่อเนื่อง",
                    districts=num_district_data["🚨 ฝนต่อเนื่อง"],
                    color1="#ff5252",
                    color2="#b71c1c",
                    emoji="🚨"
                )
                render_district_grid_html(
                    title="ฝนเริ่มเคลื่อนเข้า",
                    districts=num_district_data["🌧️ ฝนเริ่มเคลื่อนเข้า"],
                    color1="#ffb74d",
                    color2="#ef6c00",
                    emoji="🌧️"
                )
                render_district_grid_html(
                    title="เฝ้าระวัง",
                    districts=num_district_data["👀 เฝ้าระวัง"],
                    color1="#64b5f6",
                    color2="#1565c0",
                    emoji="👀"
                )
        if module_direction_data and module_coverage_data and module_district_data and module_prediction_data:
            progress = st.progress(0)
            with st.status("🌧️ Caption ", expanded=True) as caption_data:
                data_caption = []
                

                thai_months = [
                    "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน",
                    "พฤษภาคม", "มิถุนายน", "กรกฎาคม", "สิงหาคม",
                    "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"
                ]

                now = datetime.now()

                # ปัดนาทีให้เป็นช่วงละ 5 นาที
                rounded_minute = (now.minute // 5) * 5

                thai_year = now.year + 543
                thai_month = thai_months[now.month - 1]

                datetime_text = (
                    f"🗓️วันที่ {now.day} "
                    f"{thai_month} "
                    f"{thai_year} "
                    f"เวลา {now.hour:02d}.{rounded_minute:02d} น."
                )

                

                datetime_text = (
                    f"🗓️วันที่ {now.day} "
                    f"{thai_month} "
                    f"{thai_year} "
                    f"เวลา {now.hour:02d}.{now.minute:02d} น."
                )
                data_caption.append(datetime_text + "  \n")
                if heavy_n + medium_n + light_n > 0:
                    
                    data_caption.append(f"☔️กรุงเทพมหานครและปริมณฑล : มีฝนฟ้าคะนอง ร้อยละ {coverage_number_text:.0f} ของพื้นที่  \n")
                    if len(rain_district_data.get("heavy", [])):
                        data_caption.append(f"⛈️ฝนหนักเขต{', '.join(rain_district_data.get("light", []))}  \n")
                    if len(rain_district_data.get("medium", [])):
                        data_caption.append(f"🌧️ฝนปานกลางเขต{', '.join(rain_district_data.get("light", []))}  \n")
                    if len(rain_district_data.get("light", [])):
                        data_caption.append(f"🌦️ฝนเล็กน้อยเขต{', '.join(rain_district_data.get("light", []))}  \n")

                    data_caption.append(f"➡️เคลื่อนตัวทิศ{move_caption_data[1]} แนวโน้ม{trend_text}  \n\n")

                    if len(num_district_data["🚨 ฝนต่อเนื่อง"]):
                        data_caption.append(f"🔄พบฝนตกต่อเนื่องในเขต{', '.join(num_district_data['🚨 ฝนต่อเนื่อง'])} ของกรุงเทพมหานคร  \n")
                    if len(num_district_data["🌧️ ฝนเริ่มเคลื่อนเข้า"]):
                        data_caption.append(f"⛈️กลุ่มฝนกำลังเคลื่อนตัวเข้าสู่พื้นที่เขต{', '.join(num_district_data['🌧️ ฝนเริ่มเคลื่อนเข้า'])}  \n")
                    if len(num_district_data["👀 เฝ้าระวัง"]):
                        data_caption.append(f"⚠️ควรเฝ้าระวังฝนตกสะสมในพื้นที่เขต{', '.join(num_district_data['👀 เฝ้าระวัง'])}  \n\n")

                    data_caption.append(f"⛈️ ท่านสามารถติดตามคาดการณ์ฝนล่วงหน้า 3 ชั่วโมงได้ที่  \n")
                    data_caption.append(f"👉 https://dds.bangkok.go.th/nowcast/  \n\n")
                    
                    data_caption.append("🧐 เช็กสภาพอากาศกทม. แบบเรียลไทม์ได้ที่  \n")
                    data_caption.append("📱facebook ศูนย์ป้องกันน้ำท่วมกรุงเทพมหานคร  \n")
                    data_caption.append("👉 https://www.facebook.com/bkk.best?mibextid=LQQJ4d  \n\n")

                    data_caption.append("📡 สถานีเรดาร์หนองจอก  \n")
                    data_caption.append("👉 https://weather.bangkok.go.th/Radar/RadarAnimation.aspx  \n\n")

                    data_caption.append("📡 สถานีเรดาร์หนองแขม  \n")
                    data_caption.append("👉 https://weather.bangkok.go.th/Radar/RadarAnimationNk.aspx  \n\n")
                    st.write("".join(data_caption))
                else:
                    tempurature = 30
                    r_humidity_data = 79
                    data_caption.append(f"☁️ พื้นที่ กทม. ไม่พบกลุ่มฝน  \n")
                    data_caption.append(f"🌡️ อุณหภูมิที่สำนักการระบายน้ำ {tempurature} องศาเซลเซียส  \n")
                    data_caption.append(f"💧 ความชื้นสัมพัทธ์ {r_humidity_data}%  \n\n")

                    if len(num_district_data["🌧️ ฝนเริ่มเคลื่อนเข้า"]):
                        data_caption.append(f"⛈️กลุ่มฝนกำลังเคลื่อนตัวเข้าสู่พื้นที่เขต{', '.join(num_district_data["🌧️ ฝนเริ่มเคลื่อนเข้า"])}  \n")
                    if len(num_district_data["👀 เฝ้าระวัง"]):
                        data_caption.append(f"⚠️ควรเฝ้าระวังฝนตกสะสมในพื้นที่เขต{', '.join(num_district_data["👀 เฝ้าระวัง"])}  \n\n")
                        
                    st.write("".join(data_caption))

